"""Chapter 6's learning mechanisms as source: the seeds of chapter 7's search.

Each file is one mechanism in the form `helix.memory.mechanism_runtime`
runs: its writer prompts as constants and one `async def learn(lab)`. They
are read as text and run under the mechanism guard, never imported for use,
because the search treats every mechanism, these included, as source a
designer may rewrite.
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).parent
# The order chapter 6's search proposed them in, chapter 5's RL first.
SEEDS = ("value_learning", "reflection", "success_distillation", "transition_distillation", "skill_synthesis")
CONTROL = "no_experience"


def seed_source(name: str) -> str:
    return (HERE / f"{name}.py").read_text(encoding="utf-8")
