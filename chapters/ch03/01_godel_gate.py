"""Chapter 3: the Godel machine's proof gate, in plain Python (no LLM, no cost).

Schmidhuber's Godel machine rewrites its own code only when it can PROVE the
rewrite raises utility. Real agents cannot prove anything useful about
themselves, so the rest of this book trades the proof for a measurement. This
demo shows exactly what that trade gives up.

A proof gate is certain but almost never available; an evidence gate is always
available but only as trustworthy as the sample it saw. We "self-modify" a sort
routine and judge each candidate two ways: a Godel gate that checks the whole
input space, and an evidence gate that checks one sample. Run:

    python chapters/ch03/01_godel_gate.py
"""
from __future__ import annotations

from itertools import permutations

# The whole input space for length-4 lists: 24 permutations. Small enough to
# check exhaustively, which is the only reason a proof is possible here.
ALL_INPUTS = [list(p) for p in permutations(range(4))]


def bubble(xs: list[int]) -> tuple[list[int], int]:
    """The current implementation. Always does the full n*(n-1) comparisons."""
    xs, comps = list(xs), 0
    for _ in range(len(xs)):
        for k in range(len(xs) - 1):
            comps += 1
            if xs[k] > xs[k + 1]:
                xs[k], xs[k + 1] = xs[k + 1], xs[k]
    return xs, comps


def bubble_early_exit(xs: list[int]) -> tuple[list[int], int]:
    """A GENUINELY better candidate: correct everywhere, and it stops early once a
    pass makes no swaps, so it never does more comparisons and often does fewer."""
    xs, comps = list(xs), 0
    for _ in range(len(xs)):
        swapped = False
        for k in range(len(xs) - 1):
            comps += 1
            if xs[k] > xs[k + 1]:
                xs[k], xs[k + 1] = xs[k + 1], xs[k]
                swapped = True
        if not swapped:
            break
    return xs, comps


def one_pass(xs: list[int]) -> tuple[list[int], int]:
    """A deliberately deceptive candidate: a single bubble pass. It is cheap and
    correct on already-sorted input, but wrong on most inputs. It will fool an
    evidence gate that only ever looks at the easy sample."""
    xs, comps = list(xs), 0
    for k in range(len(xs) - 1):
        comps += 1
        if xs[k] > xs[k + 1]:
            xs[k], xs[k + 1] = xs[k + 1], xs[k]
    return xs, comps


def provably_better(current, candidate) -> bool:
    """The proof gate: accept the candidate only if it is provably better over the
    entire input space, correct everywhere and never more comparisons, strictly
    fewer on at least one input. No sample can fool this, but you can almost never
    build it for real code."""
    strictly = False
    for xs in ALL_INPUTS:               # walk every input, not a sample
        out, cand_comps = candidate(xs)
        _, cur_comps = current(xs)
        if out != sorted(xs):           # wrong on even one input: no proof
            return False
        if cand_comps > cur_comps:      # worse on even one input: no proof
            return False
        if cand_comps < cur_comps:
            strictly = True
    return strictly


def empirically_better(current, candidate, sample: list[int]) -> bool:
    """The evidence gate: accept the candidate if it looks better on one sample.
    Always available, costs a single run, and trusts whatever that sample shows."""
    out, cand_comps = candidate(sample)  # run one input, not the whole space
    _, cur_comps = current(sample)
    return out == sorted(sample) and cand_comps < cur_comps


def main() -> None:
    sorted_sample = [0, 1, 2, 3]  # the easy case the evidence gate happens to see

    print("Candidate A: bubble_early_exit (genuinely better)")
    print(f"  Godel gate    -> {'ACCEPT' if provably_better(bubble, bubble_early_exit) else 'REJECT'}")
    print(f"  evidence gate -> {'ACCEPT' if empirically_better(bubble, bubble_early_exit, sorted_sample) else 'REJECT'}")
    print()
    print("Candidate B: one_pass (cheap on the sample, wrong in general)")
    print(f"  Godel gate    -> {'ACCEPT' if provably_better(bubble, one_pass) else 'REJECT'}")
    print(f"  evidence gate -> {'ACCEPT' if empirically_better(bubble, one_pass, sorted_sample) else 'REJECT'}")
    print()
    print("The Godel gate rejects B because it is wrong on inputs the sample never showed.")
    print("The evidence gate accepts B, fooled by the one easy case. That gap, proof you")
    print("cannot have versus evidence you can, is the trade every search in this book makes.")


if __name__ == "__main__":
    main()
