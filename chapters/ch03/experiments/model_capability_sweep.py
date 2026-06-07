"""Find an agent model in the sweet spot: the genesis policy fails (real headroom)
but the oracle policy executes to a high ceiling. Runs genesis vs oracle on the
persona train tasks for a list of candidate models.

Known reference points: haiku-4-5 has large headroom but a low ceiling (~0.5);
sonnet-4-6 is too strong (genesis ~0.61, no headroom); sonnet-4-5 is the sweet
spot (genesis ~0.35, oracle ~0.73). Edit CANDIDATES to try others (an open model
via a configured LiteLLM provider, for a cross-family robustness check).
"""
import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helix.env import load_env

load_env()

from agents.travel import build_travel_agent  # noqa: E402
from agents.travel_persona import (  # noqa: E402
    PERSONAS, POLICY_GENESIS_PERSONA, POLICY_ORACLE_PERSONA, build_persona_policy,
    load_persona_tasks, persona_descriptions, score_persona,
)
from agents.travel_sim import reconstruct_trip  # noqa: E402

TOOL_DEFAULTS = {"default_cabin": "economy", "default_rate_code": "Q"}
DESCS = persona_descriptions()
TRAIN = load_persona_tasks(Path(__file__).parent.parent / "travel_persona_tasks.json", "train")

CANDIDATES = ["claude-haiku-4-5", "claude-sonnet-4-5", "claude-sonnet-4-6"]


async def mean(policy, model):
    a = build_travel_agent(DESCS, system_prompt=build_persona_policy(policy), model=model, tool_defaults=TOOL_DEFAULTS)
    total = 0.0
    for t in TRAIN:
        _, traj = await a.run(t.request)
        total += score_persona(reconstruct_trip(traj), PERSONAS[t.persona], t.required)[0]
    return total / len(TRAIN)


async def main_async():
    print(f"\n{'model':<22}{'genesis':>9}{'oracle':>9}{'headroom':>10}")
    for m in CANDIDATES:
        try:
            g = await mean(POLICY_GENESIS_PERSONA, m)
            o = await mean(POLICY_ORACLE_PERSONA, m)
            print(f"{m:<22}{g:>9.2f}{o:>9.2f}{o - g:>10.2f}")
        except Exception as e:  # noqa: BLE001
            print(f"{m:<22}  FAILED: {type(e).__name__}: {str(e)[:80]}")
    print("\nsweet spot = low genesis (headroom) AND high oracle (executable ceiling)")


if __name__ == "__main__":
    asyncio.run(main_async())
