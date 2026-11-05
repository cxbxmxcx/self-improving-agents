"""Function tools, shaped like the OpenAI Agents SDK's `@function_tool`.

A FunctionTool keeps the user's function and the argument model the model
sees. The argument model is built exactly as helix's `tool` decorator builds
it (a pydantic model named `<function>_Args` from the type hints), so a tool
defined here sends the same schema helix would, and recorded runs replay.

A first parameter annotated `RunContextWrapper` receives the run's context
and is left out of the schema. `read_only=True` marks a tool learned
procedures may call through their `api` object (chapter 6's rule: procedures
compute, the agent acts).

Errors: with no `failure_error_function`, an exception reaches helix's loop,
which returns `{"error": "<message>"}` to the model as the tool result. With
one, its return value is the result instead.
"""

from __future__ import annotations

import inspect
import typing
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, get_type_hints

from pydantic import BaseModel, create_model

from metacog.run_context import RunContextWrapper


def _is_context(annotation: Any) -> bool:
    if annotation is None:
        return False
    origin = typing.get_origin(annotation) or annotation
    return inspect.isclass(origin) and issubclass(origin, RunContextWrapper)


@dataclass
class FunctionTool:
    name: str
    description: str
    params_model: type[BaseModel]
    fn: Callable[..., Any]
    takes_context: bool = False
    read_only: bool = False
    failure_error_function: Callable[[RunContextWrapper[Any], Exception], Any] | None = None

    @property
    def params_json_schema(self) -> dict[str, Any]:
        return self.params_model.model_json_schema()

    def to_openai_schema(self) -> dict[str, Any]:
        return {"type": "function",
                "function": {"name": self.name, "description": self.description,
                             "parameters": self.params_json_schema}}

    async def invoke(self, ctx: RunContextWrapper[Any] | None, **arguments: Any) -> Any:
        """Validate the arguments and call the function, awaiting it if it is async."""
        validated = self.params_model.model_validate(arguments)
        kwargs = {name: getattr(validated, name) for name in type(validated).model_fields}
        try:
            result = self.fn(ctx, **kwargs) if self.takes_context else self.fn(**kwargs)
            if inspect.isawaitable(result):
                result = await result
            return result
        except Exception as exc:
            if self.failure_error_function is None:
                raise
            handled = self.failure_error_function(ctx, exc)  # type: ignore[arg-type]
            return await handled if inspect.isawaitable(handled) else handled

    def bind(self, ctx: RunContextWrapper[Any], executor: Callable[..., Any] | None = None) -> Any:
        """This tool as a helix Tool for one run, calling with that run's context.

        An `executor(tool, ctx, arguments)` runs the call instead, as a replayed
        world does (metacog.meta.replay).
        """
        from helix.tools import Tool

        async def call(**arguments: Any) -> Any:
            if executor is not None:
                return await executor(self, ctx, arguments)
            return await self.invoke(ctx, **arguments)

        return Tool(name=self.name, description=self.description, args_model=self.params_model, fn=call)


def function_tool(
    func: Callable[..., Any] | None = None,
    *,
    name_override: str | None = None,
    description_override: str | None = None,
    read_only: bool = False,
    failure_error_function: Callable[[RunContextWrapper[Any], Exception], Any] | None = None,
) -> Any:
    """Turn a typed function, sync or async, into a tool. Usable bare or with arguments.

    The description is `description_override`, else the docstring, else the
    function's name, as helix's decorator chooses it.
    """

    def build(fn: Callable[..., Any]) -> FunctionTool:
        hints = get_type_hints(fn)
        hints.pop("return", None)
        params = list(inspect.signature(fn).parameters.items())
        takes_context = bool(params) and _is_context(hints.get(params[0][0]))
        if takes_context:
            params = params[1:]
        fields: dict[str, Any] = {}
        for name, param in params:
            default = ... if param.default is inspect.Parameter.empty else param.default
            fields[name] = (hints.get(name, str), default)
        model = create_model(f"{fn.__name__}_Args", **fields)
        description = description_override or (fn.__doc__ or "").strip() or fn.__name__
        return FunctionTool(name=name_override or fn.__name__, description=description, params_model=model, fn=fn,
                            takes_context=takes_context, read_only=read_only,
                            failure_error_function=failure_error_function)

    if func is not None:
        return build(func)
    return build
