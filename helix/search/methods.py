"""Build a code search by method name, so GEPA and DGM can be swapped on any search.

Both methods take the same evaluator, a function from a candidate to its
measurement, and report through the same archive: with an evaluator the search
measures every candidate itself and records it. The caller names a method and
otherwise configures the search once.

GEPA evolves a population and needs a crossover prompt. DGM keeps every variant
in its archive, forks from any of them, and has no use for crossover, which is
why it suits artifacts too large to splice, such as a whole learning component.
"""

from __future__ import annotations

import random
from collections.abc import Awaitable, Callable
from typing import Any

from helix.search.base import Search
from helix.search.dgm import DGMSearch, ReflectiveMutator
from helix.search.gepa import GEPA

METHODS = ("gepa", "dgm")


class NoQuestions:
    """GEPA asks an evaluation source for questions; an evaluator brings its own cases."""

    label = "the evaluator's own cases"

    async def get_questions(self, n: int | None = None) -> list[Any]:
        return []


def build_search(
    method: str,
    *,
    proposer_model: str,
    evaluator: Callable[[Any, Any], Awaitable[Any]],
    mutation_prompt: str,
    crossover_prompt: str | None = None,
    validate: Callable[[str], str | None] | None = None,
    proposer_params: dict[str, Any] | None = None,
    population: int = 4,
    generations: int = 2,
    objectives: list[str] | dict[str, str] | None = None,
    rng: random.Random | None = None,
    distinct_proposals: bool = True,
) -> Search:
    """A configured search. DGM runs `population * generations` rounds, the candidates GEPA would make.

    `distinct_proposals` tells each proposal it is a separate one. Identical prompts would otherwise draw one
    answer from a recording or a cached model, and a population would be a single design.
    """
    if method == "gepa":
        if crossover_prompt is None:
            raise ValueError("GEPA needs a crossover prompt")
        return GEPA(proposer_model=proposer_model, eval_source=NoQuestions(), population_size=population,
                    generations=generations, mutation_prompt=mutation_prompt, crossover_prompt=crossover_prompt,
                    evaluator=evaluator, validate=validate, objectives=objectives, proposer_params=proposer_params,
                    rng=rng, distinct_proposals=distinct_proposals)
    if method == "dgm":
        mutator = ReflectiveMutator(model=proposer_model, prompt=mutation_prompt, params=proposer_params,
                                    validate=validate)
        return DGMSearch(mutator=mutator, rounds=population * generations, evaluator=evaluator, rng=rng)
    raise ValueError(f"unknown search method {method!r}; expected one of {METHODS}")
