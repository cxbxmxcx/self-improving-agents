"""Description helpers: render Signals, Searches, EvalSources as one-liners.

The chat UI uses these in improver cards to make a complex configuration
legible at a glance. Composite signals, swap-and-agree wrappers, strategy
chains all get rendered as readable strings rather than hidden behind
class names.

These walk runtime objects (no introspection magic) and pull out the most
useful identifying parameters. New Signal / Search classes get reasonable
defaults via fallback to type(obj).__name__.
"""

from __future__ import annotations

from typing import Any


def describe_signal(signal: Any) -> str:
    """Render a Signal (possibly wrapped or composite) as a human string.

    Examples:
        SwapAndAgree(PairwiseJudge, model=claude-sonnet-4-6)
        CompositeSignal[mean] parts=[LiveTrajectoryJudge, ImplicitFeedbackSignal]
        LiveTrajectoryJudge(model=claude-sonnet-4-6)
    """
    name = type(signal).__name__

    # SwapAndAgree wraps an inner signal
    inner = getattr(signal, "inner", None)
    if inner is not None:
        return f"{name}({describe_signal(inner)})"

    # CompositeSignal has a `signals` list and an aggregator
    sub_signals = getattr(signal, "signals", None)
    aggregator = getattr(signal, "aggregator", None)
    if sub_signals and aggregator:
        parts = ", ".join(describe_signal(s) for s in sub_signals)
        weights = getattr(signal, "weights", None)
        wstr = ""
        if weights and len(weights) == len(sub_signals):
            wstr = f" weights={[round(w, 2) for w in weights]}"
        return f"{name}[{aggregator}{wstr}] parts=[{parts}]"

    # Leaf signals: pull the model name if present
    model = getattr(signal, "model", None)
    if model:
        return f"{name}(model={model})"

    return name


def describe_search(search: Any) -> str:
    """Render a Search as a human string.

    Examples:
        SPO(proposer_model=claude-sonnet-4-6, rounds=1)
        StrategyChain[max_failures=1] strategies=[SPO, GEPA]
        GEPA(proposer=claude-sonnet-4-6, pop=4, gens=2)
    """
    name = type(search).__name__

    # StrategyChain wraps a list of strategies
    strategies = getattr(search, "strategies", None)
    max_failures = getattr(search, "max_failures", None)
    if strategies is not None and max_failures is not None:
        parts = ", ".join(describe_search(s) for s in strategies)
        return f"{name}[max_failures={max_failures}] strategies=[{parts}]"

    parts: list[str] = []
    proposer_model = getattr(search, "proposer_model", None)
    if proposer_model:
        parts.append(f"proposer_model={proposer_model}")

    if hasattr(search, "rounds"):
        parts.append(f"rounds={search.rounds}")
    if hasattr(search, "population_size"):
        parts.append(f"pop={search.population_size}")
    if hasattr(search, "generations"):
        parts.append(f"gens={search.generations}")

    return f"{name}({', '.join(parts)})" if parts else name


def describe_eval_source(eval_source: Any) -> str:
    """Render an EvalSource as a human string.

    Examples:
        FixedEvalSet(24 questions)
        RecentTrajectorySource(buffer=30, current=12)
    """
    name = type(eval_source).__name__

    # Use the source's own label() if available
    label = getattr(eval_source, "label", None)
    if isinstance(label, str):
        return label
    if callable(label):
        try:
            return str(label())
        except Exception:
            pass

    return name


def describe_archive(archive: Any) -> str:
    """Render an Archive as a human string."""
    name = type(archive).__name__
    path = getattr(archive, "path", None)
    if path:
        return f"{name}({path})"
    return name


def describe_policy(policy: Any) -> dict[str, Any]:
    """Return a flat dict of policy fields for tabular rendering."""
    out: dict[str, Any] = {}
    for field in (
        "schedule", "interval_sec", "jitter_sec", "max_in_flight",
        "questions_per_round", "promote_threshold_win_rate",
        "max_concurrent_questions", "max_iterations_per_question",
        "max_tool_calls_per_question", "quiesce_on_empty_eval",
    ):
        if hasattr(policy, field):
            val = getattr(policy, field)
            # Unwrap enums to their value
            value_attr = getattr(val, "value", None)
            out[field] = value_attr if value_attr is not None else val
    return out
