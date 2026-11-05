"""metacog: an agent SDK whose agents learn to think less.

The public surface follows the OpenAI Agents SDK (Agent, Runner,
function_tool, ModelSettings). Every run compiles to the helix agent loop
from *Self-Improving Agents*, and metacognition (callable procedures,
reasoning floors, and the online recorder, controller, detector, learner,
and gate) is added as hooks around that loop, never as a change to it.

See HARNESS_PLAN.md for the design and milestones.
"""

from metacog.agent import Agent
from metacog.exceptions import (
    AgentsException,
    InputGuardrailTripwireTriggered,
    MaxTurnsExceeded,
    ModelBehaviorError,
    OutputGuardrailTripwireTriggered,
    UserError,
)
from metacog.guardrail import (
    GuardrailFunctionOutput,
    InputGuardrail,
    InputGuardrailResult,
    OutputGuardrail,
    OutputGuardrailResult,
    input_guardrail,
    output_guardrail,
)
from metacog.items import ItemHelpers, MessageOutputItem, ModelResponse, ToolCallItem, ToolCallOutputItem
from metacog.lifecycle import AgentHooks, RunHooks
from metacog.model_settings import ModelSettings, Reasoning
from metacog.recording import Recording
from metacog.result import RunResult
from metacog.run import DEFAULT_MAX_TURNS, DEFAULT_MODEL, RunConfig, Runner
from metacog.run_context import RunContextWrapper, Usage
from metacog.session import Session, SQLiteSession
from metacog.stream import AgentUpdatedStreamEvent, RunItemStreamEvent, RunResultStreaming
from metacog.tool import FunctionTool, function_tool
from metacog.tracing import enable_console_export, get_current_trace, trace

__version__ = "0.0.1"

__all__ = [
    "DEFAULT_MAX_TURNS",
    "DEFAULT_MODEL",
    "Agent",
    "AgentHooks",
    "AgentUpdatedStreamEvent",
    "AgentsException",
    "FunctionTool",
    "GuardrailFunctionOutput",
    "InputGuardrail",
    "InputGuardrailResult",
    "InputGuardrailTripwireTriggered",
    "ItemHelpers",
    "MaxTurnsExceeded",
    "MessageOutputItem",
    "ModelBehaviorError",
    "ModelResponse",
    "ModelSettings",
    "OutputGuardrail",
    "OutputGuardrailResult",
    "OutputGuardrailTripwireTriggered",
    "Reasoning",
    "Recording",
    "RunConfig",
    "RunContextWrapper",
    "RunHooks",
    "RunItemStreamEvent",
    "RunResult",
    "RunResultStreaming",
    "Runner",
    "SQLiteSession",
    "Session",
    "ToolCallItem",
    "ToolCallOutputItem",
    "Usage",
    "UserError",
    "enable_console_export",
    "function_tool",
    "get_current_trace",
    "input_guardrail",
    "output_guardrail",
    "trace",
]
