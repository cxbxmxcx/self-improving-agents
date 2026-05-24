"""Improver sidebar panel — multi-improver view.

What this panel does:
  1. When the spec declares improvers but none are attached, lists them
     as previews plus an "Attach improvers from spec" button.
  2. When improvers are attached, shows one card per improver with status,
     a description (signal/search/eval source/archive/policy), and per-
     improver controls (pause/resume/trigger/detach).
  3. If the spec declares no improvers at all, points the user at the spec
     file. Improvers come from code, not the UI; this panel only surfaces
     and operates what the spec has already committed to.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

from helix.agents import load_improvers
from helix.chat_ui._introspect import (
    describe_archive,
    describe_eval_source,
    describe_policy,
    describe_search,
    describe_signal,
)
from helix.improvement.background import (
    attach_runner,
    detach_runner,
    get_runner,
)

if TYPE_CHECKING:
    from helix.agent import Agent
    from helix.improvement.improver import OfflineImprover
    from helix.improvement.online_improver import OnlineImprover
    ImproverLike = OfflineImprover | OnlineImprover


def render_improver_panel(agent: "Agent", agent_name: str) -> None:
    """Render the multi-improver sidebar panel."""
    st.sidebar.subheader("🔧 Self-improvement")

    attached_ids = [i.improver_id for i in agent.improvers]

    try:
        spec_improvers = load_improvers(agent_name, agent)
    except Exception as exc:
        st.sidebar.warning(f"Could not load spec improvers: {exc}")
        spec_improvers = []

    spec_ids = [i.improver_id for i in spec_improvers]
    unattached_spec_ids = [i for i in spec_ids if i not in attached_ids]

    # ---------------- nothing attached path ----------------
    if not attached_ids:
        if spec_improvers:
            st.sidebar.caption(
                f"This agent's spec declares **{len(spec_improvers)} improver"
                f"{'s' if len(spec_improvers) != 1 else ''}**. None attached yet."
            )
            for imp in spec_improvers:
                _render_spec_preview(imp)
            if st.sidebar.button(
                "▶ Attach improvers from spec",
                type="primary",
                use_container_width=True,
            ):
                _attach_and_start_many(agent, spec_improvers)
                st.toast(f"Attached {len(spec_improvers)} improver(s)", icon="🔧")
                st.rerun()
        else:
            st.sidebar.info(
                f"This agent's spec doesn't declare any improvers. "
                f"To enable improvement, add a `build_improvers()` function to "
                f"`agents/{agent_name}.py`."
            )
        return

    # ---------------- attached improvers as stacked cards ----------------
    st.sidebar.caption(
        f"**{len(attached_ids)} improver{'s' if len(attached_ids) != 1 else ''}** "
        f"attached to this agent."
    )

    for improver in agent.improvers:
        _render_improver_card(agent, improver, in_spec=improver.improver_id in spec_ids)

    if unattached_spec_ids:
        st.sidebar.markdown("---")
        with st.sidebar.expander(
            f"📋 {len(unattached_spec_ids)} more declared improver(s) available"
        ):
            for imp in spec_improvers:
                if imp.improver_id in unattached_spec_ids:
                    st.markdown(f"**{imp.improver_id}** — `{imp.target_artifact_id}`")
            if st.button("Attach these too", use_container_width=True, key="attach_remaining"):
                missing = [i for i in spec_improvers if i.improver_id in unattached_spec_ids]
                _attach_and_start_many(agent, missing)
                st.toast(f"Attached {len(missing)} more improver(s)", icon="🔧")
                st.rerun()


# ---------------------------------------------------------------------------
# Card per attached improver
# ---------------------------------------------------------------------------

def _render_improver_card(agent: "Agent", improver: "ImproverLike", *, in_spec: bool) -> None:
    """One stacked card per attached improver."""
    from helix.improvement.online_improver import OnlineImprover

    runner = get_runner(improver.improver_id)
    status = runner.status() if runner else None

    if status is None:
        badge = "⚪ Not running"
    elif status.paused:
        badge = "⏸ Paused"
    elif status.running:
        badge = "🟢 Running"
    else:
        badge = "🔴 Stopped"

    # Mode by class identity, per SPEC §15 (OfflineImprover) and §17
    # (OnlineImprover). Shield for offline (deploy gate), bolt for online
    # (auto-apply).
    is_online = isinstance(improver, OnlineImprover)
    mode_badge = "⚡ online" if is_online else "🛡 offline"

    # All attached improvers should come from spec now; tag remains for
    # robustness in case an external caller attaches one directly.
    spec_tag = " · *from spec*" if in_spec else " · *external*"

    with st.sidebar.container():
        st.markdown(
            f"**{improver.improver_id}** {badge} · *{mode_badge}*<br>"
            f"<span style='font-size:0.85em;'>target: `{improver.target_artifact_id}`{spec_tag}</span>",
            unsafe_allow_html=True,
        )

        eval_line = ""
        if hasattr(improver, "eval_source"):
            eval_line = f"\n\neval: `{describe_eval_source(improver.eval_source)}`"
        st.caption(
            f"signal: `{describe_signal(improver.signal)}`\n\n"
            f"search: `{describe_search(improver.search)}`{eval_line}"
        )

        if status is not None:
            c1, c2 = st.columns(2)
            c1.metric("Rounds", status.rounds_completed)
            c2.metric("In flight", status.rounds_in_flight)
            if status.last_round_result is not None:
                r = status.last_round_result
                cscore = f"{r.candidate_score:.3f}" if r.candidate_score is not None else "n/a"
                tag = "👑 PROMOTED" if r.promoted else "kept reference"
                st.caption(f"Last round (v{r.candidate_version}): {cscore} — {tag}")
            if status.last_error:
                st.warning(f"⚠ {status.last_error[:200]}")

        # Offline promote gate: if the last round produced a winner and that
        # winner is not currently the live champion, offer a single button.
        # Online improvers auto-promote on win, so they never show this.
        if not is_online:
            _render_offline_promote_gate(improver, status)

        with st.expander("Details"):
            st.markdown("**Signal chain:**")
            st.code(describe_signal(improver.signal), language="text")
            st.markdown("**Search:**")
            st.code(describe_search(improver.search), language="text")
            if hasattr(improver, "eval_source"):
                st.markdown("**Eval source:**")
                st.code(describe_eval_source(improver.eval_source), language="text")
            st.markdown("**Archive:**")
            st.code(describe_archive(improver.archive), language="text")
            st.markdown("**Policy:**")
            st.json(describe_policy(improver.policy))

        if status is not None and status.thread_alive:
            c1, c2, c3 = st.columns(3)
            with c1:
                if status.paused:
                    if st.button("▶", key=f"resume_{improver.improver_id}",
                                 help="Resume", use_container_width=True):
                        runner.resume()
                        st.rerun()
                else:
                    if st.button("⏸", key=f"pause_{improver.improver_id}",
                                 help="Pause", use_container_width=True):
                        runner.pause()
                        st.rerun()
            with c2:
                if st.button("⚡", key=f"trigger_{improver.improver_id}",
                             help="Trigger round now", use_container_width=True):
                    try:
                        runner.trigger_round()
                        st.toast("Round triggered", icon="⚡")
                    except RuntimeError as exc:
                        st.error(f"Trigger failed: {exc}")
                    st.rerun()
            with c3:
                if st.button("✖", key=f"detach_{improver.improver_id}",
                             help="Detach this improver entirely",
                             use_container_width=True):
                    detach_runner(improver.improver_id)
                    agent.detach_improver(improver.improver_id)
                    st.toast("Improver detached", icon="✖")
                    st.rerun()
        else:
            if st.button("▶ Start runner", key=f"start_{improver.improver_id}",
                         use_container_width=True):
                runner = attach_runner(improver)
                runner.start()
                st.toast("Improver started", icon="🔧")
                st.rerun()

        st.markdown("---")


# ---------------------------------------------------------------------------
# Offline promote gate: one button per pending winning candidate
# ---------------------------------------------------------------------------

def _render_offline_promote_gate(improver: "ImproverLike", status) -> None:
    """If an offline improver has a winning candidate that's not live, show
    a single 'Promote v{N} → live' button.

    A candidate is promotable when:
      - The most recent round result has promoted=True (candidate beat
        reference by score), AND
      - That candidate's version is NOT the current live champion.

    Online improvers don't reach this code; they auto-promote on win.
    """
    import asyncio

    if status is None or status.last_round_result is None:
        return

    last = status.last_round_result
    if not last.promoted:
        return

    candidate_version = last.candidate_version
    target_id = improver.target_artifact_id

    try:
        live = asyncio.run(improver.archive.live_champion(target_id))
    except Exception:
        return

    if live is not None and live.version == candidate_version:
        return  # already live; nothing to promote

    candidate_score = last.candidate_score
    score_text = f" (score {candidate_score:.3f})" if candidate_score is not None else ""
    live_text = f"v{live.version}" if live is not None else "genesis"

    st.info(
        f"Candidate **v{candidate_version}**{score_text} is pending. "
        f"Live champion is **{live_text}**."
    )
    if st.button(
        f"👑 Promote v{candidate_version} → live",
        key=f"promote_{improver.improver_id}_{candidate_version}",
        type="primary",
        use_container_width=True,
    ):
        _promote_candidate(improver, candidate_version, score_text)
        st.rerun()


def _promote_candidate(improver: "ImproverLike", version: int, score_text: str) -> None:
    """Call archive.promote with the current user as approver."""
    import asyncio

    from helix.chat_ui.state import get_session_state

    state = get_session_state()
    approver = state.user_id or "anonymous"
    reason = f"manual promotion via chat UI{score_text}"

    try:
        asyncio.run(
            improver.archive.promote(
                artifact_id=improver.target_artifact_id,
                version=version,
                approver=approver,
                reason=reason,
            )
        )
        st.toast(f"Promoted v{version}", icon="👑")
    except Exception as exc:
        st.error(f"Promotion failed: {exc}")


# ---------------------------------------------------------------------------
# Preview of an unattached spec improver
# ---------------------------------------------------------------------------

def _render_spec_preview(improver: "ImproverLike") -> None:
    """Compact preview of an improver that's declared by the spec but not attached."""
    with st.sidebar.container():
        st.markdown(
            f"**{improver.improver_id}** (not running)<br>"
            f"<span style='font-size:0.85em;'>target: `{improver.target_artifact_id}`</span>",
            unsafe_allow_html=True,
        )
        eval_line = ""
        if hasattr(improver, "eval_source"):
            eval_line = f"\n\neval: `{describe_eval_source(improver.eval_source)}`"
        st.caption(
            f"signal: `{describe_signal(improver.signal)}`\n\n"
            f"search: `{describe_search(improver.search)}`{eval_line}"
        )
        st.markdown("---")


# ---------------------------------------------------------------------------
# Batch attach for declared improvers
# ---------------------------------------------------------------------------

def _attach_and_start_many(agent: "Agent", improvers: list) -> None:
    """Attach a list of improvers to the agent and start their runners."""
    for imp in improvers:
        agent.attach_improver(imp)
        runner = attach_runner(imp)
        runner.start()
