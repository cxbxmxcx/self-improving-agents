"""The Tool primitive.

A Tool wraps a callable with a name, a description, and an argument schema.
The schema is derived from the function's type hints via Pydantic's create_model,
so readers write normal Python and get LLM-ready JSON schemas for free.

Tool descriptions are themselves artifacts (Spec §1.2 kind=tool_description),
but at v0 we don't put them under search yet. Ch 6 turns the description
attribute into an Artifact and aims searches at it.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, get_type_hints

from pydantic import BaseModel, create_model


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
