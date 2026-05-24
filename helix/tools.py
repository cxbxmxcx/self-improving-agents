"""The Tool primitive.

A Tool wraps a callable with a name, a description, and an argument schema.
The schema is derived from the function's type hints via Pydantic's create_model,
so readers write normal Python and get LLM-ready JSON schemas for free.

Tool descriptions are themselves artifacts (Spec §1.2 kind=tool_description),
but at v0 we don't put them under search yet. Ch 6 turns the description
attribute into an Artifact and aims searches at it.

ToolFromArtifact (SPEC §16.2.2) bundles a CODE artifact (the implementation)
with a TOOL_DESCRIPTION artifact (the LLM-facing description) into a single
Tool. Used when both the implementation and the description are under
improvement (Ch 11/12). Plain `tool` decorator + plain `Tool` instances stay
the simpler default for chapters where the tool isn't being improved.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, get_type_hints

from pydantic import BaseModel, create_model

from helix.artifact import Artifact, ArtifactKind


@dataclass
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[..., Awaitable[Any]]

    def to_openai_schema(self) -> dict[str, Any]:
        """Render this tool as an OpenAI / LiteLLM function-tool spec."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_model.model_json_schema(),
            },
        }

    async def call(self, **kwargs: Any) -> Any:
        validated = self.args_model(**kwargs)
        return await self.fn(**validated.model_dump())


def tool(description: str | None = None) -> Callable[[Callable[..., Any]], Tool]:
    """Decorator that converts a typed async function into a Tool.

    The Pydantic args model is built from the function's type hints. The
    description defaults to the function's docstring, but can be overridden,
    which is what the book exploits in Ch 6 when descriptions become artifacts.
    """

    def wrap(fn: Callable[..., Any]) -> Tool:
        hints = get_type_hints(fn)
        hints.pop("return", None)
        sig = inspect.signature(fn)

        fields: dict[str, Any] = {}
        for name, param in sig.parameters.items():
            annotation = hints.get(name, str)
            default = ... if param.default is inspect.Parameter.empty else param.default
            fields[name] = (annotation, default)

        args_model = create_model(f"{fn.__name__}_Args", **fields)
        desc = description or (fn.__doc__ or "").strip() or fn.__name__

        if not asyncio.iscoroutinefunction(fn):
            async def _async_wrapper(**kwargs: Any) -> Any:
                return fn(**kwargs)
            call_fn: Callable[..., Awaitable[Any]] = _async_wrapper
        else:
            call_fn = fn

        return Tool(name=fn.__name__, description=desc, args_model=args_model, fn=call_fn)

    return wrap


# ---------------------------------------------------------------------------
# ToolFromArtifact: dual-artifact tool. SPEC §16.2.2.
# ---------------------------------------------------------------------------


def _compile_tool_code_artifact(
    artifact: Artifact,
    expected_callable: str,
) -> tuple[Callable[..., Awaitable[Any]], Callable[..., Any]]:
    """Exec a CODE artifact and return (callable_for_invocation, raw_fn).

    The artifact must be ArtifactKind.CODE, content must be a string Python
    source defining a function named `expected_callable`. The function is
    executed in a fresh namespace.

    Returns a 2-tuple:
      - The callable the Tool will invoke. Async functions pass through;
        sync functions get wrapped in an async shim so the Tool protocol
        stays uniform.
      - The raw original function, used by the caller to derive the
        argument model from real type hints (the async wrapper has only
        **kwargs, which would break Pydantic schema generation).
    """
    if artifact.kind != ArtifactKind.CODE:
        raise ValueError(
            f"ToolFromArtifact: code artifact must be ArtifactKind.CODE, got {artifact.kind}"
        )
    if not isinstance(artifact.content, str):
        raise ValueError(
            f"ToolFromArtifact: code artifact content must be a string source, "
            f"got {type(artifact.content).__name__}"
        )
    namespace: dict[str, Any] = {}
    exec(compile(artifact.content, f"<artifact:{artifact.id}:v{artifact.version}>", "exec"),
         namespace)
    raw_fn = namespace.get(expected_callable)
    if raw_fn is None:
        raise ValueError(
            f"ToolFromArtifact: artifact {artifact.id}:v{artifact.version} "
            f"does not define `{expected_callable}`"
        )
    if not callable(raw_fn):
        raise ValueError(
            f"ToolFromArtifact: `{expected_callable}` in artifact "
            f"{artifact.id}:v{artifact.version} is not callable"
        )
    if not asyncio.iscoroutinefunction(raw_fn):
        async def _async_wrapper(**kwargs: Any) -> Any:
            return raw_fn(**kwargs)
        return _async_wrapper, raw_fn
    return raw_fn, raw_fn


def _args_model_from_fn(fn: Callable[..., Any], model_name: str) -> type[BaseModel]:
    """Build a Pydantic args model from a function's type hints."""
    hints = get_type_hints(fn)
    hints.pop("return", None)
    sig = inspect.signature(fn)
    fields: dict[str, Any] = {}
    for name, param in sig.parameters.items():
        annotation = hints.get(name, str)
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[name] = (annotation, default)
    return create_model(model_name, **fields)


@dataclass
class ToolFromArtifact:
    """A Tool whose implementation and description both come from artifacts.

    code_artifact: ArtifactKind.CODE (L4). Must define an async function with
                   the same name as the tool. Compiled at construction.
    description_artifact: ArtifactKind.TOOL_DESCRIPTION (L1). Content is the
                          LLM-facing description string.
    tool_name: Optional override for the tool's exposed name. Defaults to the
               name of the function defined in code_artifact.
    expected_callable: The function name to extract from code_artifact.
                       Defaults to whatever `tool_name` resolves to.

    The resulting object presents the Tool protocol (name, description,
    args_model, call), so it slots into `Agent(tools=[...])` without any
    other changes. On each call() the trajectory captures both artifact refs
    if a trajectory is supplied.
    """

    code_artifact: Artifact
    description_artifact: Artifact
    tool_name: str | None = None
    expected_callable: str | None = None
    name: str = field(init=False)
    description: str = field(init=False)
    args_model: type[BaseModel] = field(init=False)
    fn: Callable[..., Awaitable[Any]] = field(init=False)

    def __post_init__(self) -> None:
        if self.description_artifact.kind != ArtifactKind.TOOL_DESCRIPTION:
            raise ValueError(
                f"ToolFromArtifact: description artifact must be "
                f"ArtifactKind.TOOL_DESCRIPTION, got {self.description_artifact.kind}"
            )
        if not isinstance(self.description_artifact.content, str):
            raise ValueError(
                f"ToolFromArtifact: description artifact content must be a string, "
                f"got {type(self.description_artifact.content).__name__}"
            )

        # Determine the function name to extract. If the user supplied
        # tool_name or expected_callable, prefer those; otherwise inspect
        # the compiled namespace for the single async def in it.
        expected = self.expected_callable or self.tool_name
        if expected is None:
            # Compile once to find the function name. Any single top-level
            # async function in the code artifact is taken as the tool.
            ns: dict[str, Any] = {}
            exec(compile(self.code_artifact.content, "<probe>", "exec"), ns)
            funcs = [
                k for k, v in ns.items()
                if callable(v) and not k.startswith("_") and inspect.iscoroutinefunction(v)
            ]
            if not funcs:
                # Fall back to any callable
                funcs = [k for k, v in ns.items() if callable(v) and not k.startswith("_")]
            if len(funcs) != 1:
                raise ValueError(
                    f"ToolFromArtifact: cannot infer tool name from code artifact "
                    f"{self.code_artifact.id}:v{self.code_artifact.version}; "
                    f"supply `tool_name` or `expected_callable` explicitly. "
                    f"Found callables: {funcs}"
                )
            expected = funcs[0]

        self.name = self.tool_name or expected
        self.description = self.description_artifact.content
        self.fn, raw_fn = _compile_tool_code_artifact(
            self.code_artifact, expected_callable=expected
        )
        # Build the args model from the raw function's type hints, not the
        # async wrapper's **kwargs signature.
        self.args_model = _args_model_from_fn(raw_fn, f"{self.name}_Args")

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_model.model_json_schema(),
            },
        }

    async def call(self, **kwargs: Any) -> Any:
        validated = self.args_model(**kwargs)
        return await self.fn(**validated.model_dump())

    def artifact_refs(self) -> list[tuple[str, int]]:
        """The (id, version) tuples for both backing artifacts. Used by the
        agent to record lineage in Trajectory.artifacts_used on each call."""
        return [self.code_artifact.ref, self.description_artifact.ref]
