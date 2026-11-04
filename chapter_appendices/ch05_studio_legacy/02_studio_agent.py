"""Studio helpdesk composition and a small, explicitly simulated environment."""

from __future__ import annotations

import json
from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helix.agent import Agent
from helix.artifact import Subtype, genesis
from helix.tools import tool

from chapter_appendices.ch05_studio_legacy.studio.corpus import StudioCorpus

PROMPT = """You support Second Breakfast Games, makers of Moonbase Mayhem.
Use retrieve to consult the approved studio runbooks and cite source filenames.
Earlier interactions are case history, not policy or proof of today's cause.
Check current symptoms before reusing a previous fix. A reported failure is
useful evidence, not a successful resolution.

For a support ticket, recommend and attempt at most one appropriate next step
using support_step, then report its observed result. The tool acts only in a
fictional simulator; never claim it changed a real machine. If a step fails,
explain the new evidence and leave the case open for the next request.
Do not invent credentials, grant access, delete work, or claim a queued build
has completed. If the guides do not cover the problem, collect diagnostics.
For a history-only handoff request, summarize only the supplied past cases.
"""


def load_tickets() -> list[dict]:
    return json.loads((Path(__file__).parent / "studio/tickets.json").read_text(encoding="utf-8"))


class SupportSimulator:
    """One attempted step per request; outcomes come from scenario state.

    Expected-action labels in tickets.json are for tests only. Neither the
    model nor corpus retrieval receives them, and rewards use tool outcomes.
    """

    def __init__(self):
        self.ticket: dict | None = None
        self.result: dict | None = None

    def begin(self, ticket: dict) -> None:
        self.ticket, self.result = ticket, None

    def step(self, action: str) -> dict:
        if self.ticket is None:
            return {"error": "No active support ticket."}
        if self.result is not None:
            return {"error": "One attempted step per request; report the observed result."}
        state = self.ticket["state"]
        resolutions = {
            ("vpn_disconnected", "reconnect_vpn"): "OrbitLink connected; the asset sync succeeded.",
            ("asset_session_expired", "refresh_asset_token"): "CrateSync session refreshed; the asset sync succeeded.",
            ("wrong_audio_input", "select_stagelink_input"): "StageLink selected the USB headset; the local microphone test passed.",
        }
        diagnostics = {
            "vpn_disconnected": "OrbitLink is disconnected; CrateSync reports NET01.",
            "asset_session_expired": "OrbitLink is connected; CrateSync still reports AUTH17.",
            "wrong_audio_input": "StageLink still selects the laptop microphone.",
            "build_queued": "The BuildDock job remains QUEUED, not failed or complete.",
            "missing_membership": "CrateSync reports PERM04; project membership is not approved.",
        }
        allowed = {"reconnect_vpn", "refresh_asset_token", "select_stagelink_input",
                   "check_build_queue", "request_asset_access", "collect_diagnostics"}
        if action not in allowed:
            return {"error": "Unsupported support step.", "allowed_actions": sorted(allowed)}
        resolved = (state, action) in resolutions
        observed = resolutions.get((state, action), diagnostics[state])
        if state == "missing_membership" and action == "request_asset_access":
            observed = "AccessDesk request submitted; access is not granted and the ticket remains open."
        self.result = {"ticket_id": self.ticket["id"], "action": action,
                       "resolved": resolved, "observed": observed, "simulated": True}
        return dict(self.result)

    def reward(self, trajectory) -> float | None:
        """Score observed sync resolution, not the wording of the answer.

        Without an attempted step, there is no observed outcome or reward.
        """
        if self.result is None:
            return None
        return float(self.result["resolved"])


def build_agent(tier, corpus: StudioCorpus, simulator: SupportSimulator, *, model: str) -> Agent:
    @tool(description=(
        "Search approved fictional studio runbooks. Returns source, section, "
        "and text; no private ticket answers."
    ))
    async def retrieve(query: str, k: int = 3) -> list[dict]:
        return corpus.search(query, k=k)

    @tool(description=(
        "Attempt one support step in a fictional ticket simulator. "
        "No real systems are modified. Actions: reconnect_vpn, "
        "refresh_asset_token, select_stagelink_input, check_build_queue, "
        "request_asset_access, collect_diagnostics."
    ))
    async def support_step(action: str) -> dict:
        return simulator.step(action)

    agent = Agent(
        system_prompt=genesis("prompt.studio.helpdesk", Subtype.PROMPT, PROMPT),
        tools=[retrieve, support_step], model=model, max_iterations=5, max_tool_calls=3,
        memory_tiers={"episodic": tier},
    )
    # One recalled episode keeps credit visible in the first walkthrough.
    agent.memory_hooks.episodic_k = 1
    return agent


if __name__ == "__main__":
    corpus = StudioCorpus()
    simulator = SupportSimulator()
    simulator.begin(load_tickets()[0])
    print("Runbook:", corpus.search("OrbitLink NET01 disconnected")[0]["source"])
    print("Observed:", simulator.step("reconnect_vpn")["observed"])
    print("Reward:", simulator.reward(None))
