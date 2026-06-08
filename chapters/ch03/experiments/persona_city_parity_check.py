"""Parity gate for the persona-city substrate: every city must be equally
winnable before any method run, or a held-out city is a hardness lottery, not a
generalization test. For each of the five cities we score the genesis policy
(generic, should be low) and the oracle policy (knows the persona strategy,
should reach ~0.8) across that city's five personas, at temperature 0 so the
numbers are clean. Pass criterion: oracle is high and similar on every city and
clears genesis by a wide margin everywhere.

Run:
    python chapters/ch03/experiments/persona_city_parity_check.py
"""
import asyncio
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_spec = importlib.util.spec_from_file_location(
    "mor", Path(__file__).parent / "persona_methods_over_runs.py")
mor = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mor)

from agents.travel_persona import POLICY_GENESIS_PERSONA, POLICY_ORACLE_PERSONA


def out(s):
    sys.stdout.buffer.write((str(s) + "\n").encode("ascii", "replace")); sys.stdout.flush()


async def main_async():
    tasks = mor.TRAIN + mor.TEST
    by_city = {}
    for t in tasks:
        by_city.setdefault(t.required["city"], []).append(t)

    out(f"\n=== per-city parity (temp {mor.AGENT_TEMPERATURE}, {len(tasks)} tasks) ===")
    out(f"  {'city':<6}{'genesis':>9}{'oracle':>9}{'gap':>7}")
    oracle_scores = []
    for city in sorted(by_city):
        ct = by_city[city]
        g = await mor.absolute(POLICY_GENESIS_PERSONA, ct, 1)
        o = await mor.absolute(POLICY_ORACLE_PERSONA, ct, 1)
        oracle_scores.append(o)
        out(f"  {city:<6}{g:>9.2f}{o:>9.2f}{o - g:>7.2f}")
    lo, hi = min(oracle_scores), max(oracle_scores)
    out(f"\n  oracle range across cities: {lo:.2f}-{hi:.2f} (spread {hi - lo:.2f})")
    out("  PASS if oracle is high (~0.8) and tight across cities; a wide spread means the cities are not parallel.")


if __name__ == "__main__":
    asyncio.run(main_async())
