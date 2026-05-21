"""Helix chat application — production-style RAG chat with self-improvement.

Launch:
    streamlit run helix/chat_ui/app.py

What it does:
  - Loads any registered agent spec from agents/*.py (default: helpdesk)
  - Provides a Streamlit chat surface with explicit session_id / user_id /
    org_id so memory tiers scope correctly
  - Collects thumbs / regenerate / copy / outcome feedback into the
    FeedbackStore
  - Optionally attaches an Improver running in the background that watches
    real conversation traffic via the event bus and improves the agent's
    system prompt against a configurable signal set (live judge + implicit
    feedback + golden set)
  - Displays the current champion artifact in the sidebar so the user can
    see when improvement has produced a new winner

This is platform code, not a chapter artifact. Chapter scripts that want a
chat demo can launch this directly or wrap it with chapter-specific
configuration.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helix.agents import describe_agent, list_agents, load_agent, load_genesis_prompt
from helix.archive import SQLiteArchive
from helix.chat_ui.chat import render_chat, render_outcome_controls
from helix.chat_ui.improver_panel import render_improver_panel
from helix.chat_ui.state import (
    ChatMessage,
    append_message,
    get_last_known_champion,
    get_session_state,
    reset_conversation,
    set_agent_name,
    set_archive_path,
    set_last_known_champion,
    set_org_id,
    set_session_id,
    set_user_id,
)
from helix.env import load_env

load_env()


DEFAULT_ARCHIVE = REPO_ROOT / "chapters" / "ch02" / "runs" / "helix_archive.sqlite"


def _build_agent_with_optional_archive(agent_name: str, archive_path: str):
    """Construct the agent, using the archive's live champion if available.

    This is what makes the chat app reflect improvements: when a candidate
    has been *promoted* (online improvers auto-promote on win; offline
    improvers wait for a human click in the panel), the chat agent picks
    up the new live version on the next user turn.

    Unlike archive.best(), live_champion() never returns an unreviewed
    candidate. Offline improvers can fill the archive with high-scoring
    candidates that have not been promoted; the chat agent serves the
    genesis prompt (from the spec) until a human promotes one of them.
    DESIGN_NOTES.md section 10.

    Side effect: when the champion changes from the previously-known one
    (tracked in session state), this function posts a system_notice into
    the conversation so the user sees the new prompt is in play.
    """
    from helix.agents import load_genesis_prompt

    system_prompt = None
    new_champion_ref: tuple[str, int] | None = None

    if archive_path and Path(archive_path).exists():
        try:
            archive = SQLiteArchive(archive_path, check_same_thread=False)
            # Look up the live champion for the spec's prompt id. If no
            # promotion has ever been recorded, live_champion returns None
            # and we fall through to the spec's genesis prompt below.
            genesis = load_genesis_prompt(agent_name)
            live = asyncio.run(archive.live_champion(genesis.id))
            if live is not None:
                system_prompt = live
                new_champion_ref = (system_prompt.id, system_prompt.version)
        except Exception:
            system_prompt = None

    # Detect champion change and post a notice once (idempotent: only fires
    # when the ref actually changes from the last observed value).
    prior = get_last_known_champion()
    if new_champion_ref is not None and prior != new_champion_ref:
        if prior is None:
            # First time we observe a champion in this session — quiet attach
            pass
        else:
            old_v = prior[1]
            new_v = new_champion_ref[1]
            append_message(ChatMessage(
                role="system_notice",
                content=(
                    f"👑 **New champion in play:** v{old_v} → **v{new_v}** "
                    f"(`{system_prompt.created_by}`). "
                    f"Your next message will be answered with the updated prompt."
                ),
            ))
        set_last_known_champion(new_champion_ref)

    return load_agent(agent_name, system_prompt=system_prompt)


def _render_sidebar(state) -> tuple[str, str]:
    """Return (selected_agent_name, archive_path) after rendering all sidebar controls."""
    st.sidebar.title("🧬 Helix Chat")
    st.sidebar.caption("Self-improving agent chat")
    st.sidebar.markdown("---")

    # ---------------- agent selector ----------------
    available = list_agents()
    if not available:
        st.sidebar.error(
            "No agent specs found in `agents/`. Define one in agents/<name>.py "
            "exporting build()."
        )
        st.stop()

    current_idx = available.index(state.agent_name) if state.agent_name in available else 0
    selected = st.sidebar.selectbox(
        "Agent",
        available,
        index=current_idx,
        help="Pick which agent spec to operate. Each spec is a python module in agents/.",
    )
    if selected != state.agent_name:
        set_agent_name(selected)
        reset_conversation()
        st.rerun()

    # Show the spec's description
    try:
        meta = describe_agent(selected)
        if meta.get("description"):
            st.sidebar.caption(meta["description"])
    except Exception:
        pass

    st.sidebar.markdown("---")

    # ---------------- identity ----------------
    st.sidebar.subheader("👤 Identity")
    user_id = st.sidebar.text_input(
        "User id",
        value=state.user_id,
        help="Used by memory tiers for per-user scope. Different users get different memories.",
    )
    if user_id != state.user_id:
        set_user_id(user_id)

    org_id = st.sidebar.text_input(
        "Org id",
        value=state.org_id,
        help="Used by memory tiers for per-org scope (policies, shared facts).",
    )
    if org_id != state.org_id:
        set_org_id(org_id)

    st.sidebar.caption(f"Session: `{state.session_id}`")

    if st.sidebar.button("💬 New session", use_container_width=True):
        reset_conversation()
        st.rerun()

    st.sidebar.markdown("---")

    # ---------------- archive / champion source ----------------
    st.sidebar.subheader("👑 Champion source")
    archive_input = st.sidebar.text_input(
        "Archive path",
        value=state.archive_path or str(DEFAULT_ARCHIVE),
        help=(
            "Path to a helix_archive.sqlite. If set, the chat agent uses "
            "the archive's live champion as its system prompt. Offline "
            "improvers' candidates appear only after you promote them."
        ),
    )
    if archive_input != state.archive_path:
        set_archive_path(archive_input)
        st.rerun()

    # Show current champion summary
    if archive_input and Path(archive_input).exists():
        try:
            arc = SQLiteArchive(archive_input, check_same_thread=False)
            best = asyncio.run(arc.best(k=1))
            if best:
                champ = best[0]
                score = champ.measurement.score if champ.measurement else None
                st.sidebar.success(
                    f"**Champion:** v{champ.artifact.version}"
                    + (f" — score {score:.3f}" if score is not None else "")
                    + f"\n\nby `{champ.artifact.created_by}`"
                )
            else:
                st.sidebar.info("Archive exists but has no measured artifacts yet.")
        except Exception as exc:
            st.sidebar.warning(f"Could not read archive: {exc}")
    else:
        st.sidebar.caption("(no archive — agent uses its genesis prompt)")

    st.sidebar.markdown("---")

    # ---------------- session outcome ----------------
    st.sidebar.subheader("📑 Session outcome")
    render_outcome_controls(state)

    st.sidebar.markdown("---")

    # ---------------- pointer to other panels ----------------
    st.sidebar.caption(
        "Want to see what the improver is doing, the lineage tree, the "
        "verdicts? Run the dashboard separately:\n\n"
        "`streamlit run helix/dashboard/app.py`"
    )

    return selected, archive_input


def main() -> None:
    st.set_page_config(
        page_title="Helix Chat",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    state = get_session_state()
    selected_agent, archive_path = _render_sidebar(state)
    state = get_session_state()  # refresh after sidebar mutations

    # ---------------- header ----------------
    st.title(f"🧬 Helix Chat — `{selected_agent}`")
    st.caption(
        f"session=`{state.session_id}` • user=`{state.user_id}` • org=`{state.org_id}`"
    )

    # ---------------- agent construction ----------------
    try:
        agent = _build_agent_with_optional_archive(selected_agent, archive_path)
    except FileNotFoundError as exc:
        # Most common cause: the helpdesk corpus hasn't been built yet
        st.error(
            f"⚠ Cannot build agent: {exc}\n\n"
            "If you're using the helpdesk agent, ensure the LanceDB corpus exists. "
            "Build it with:\n\n```\npython ingestion/build_index.py\n```"
        )
        st.stop()
    except Exception as exc:
        st.error(f"✖ Agent construction failed: {type(exc).__name__}: {exc}")
        st.stop()

    # ---------------- improver panel ----------------
    st.sidebar.markdown("---")
    render_improver_panel(agent, selected_agent)

    # ---------------- chat surface ----------------
    render_chat(agent, state)


if __name__ == "__main__":
    main()
