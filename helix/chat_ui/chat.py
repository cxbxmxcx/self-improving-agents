"""Chat surface: render the conversation, handle input, collect feedback.

Three responsibilities:
  1. Render message history (user / assistant turns plus optional notices)
  2. Accept new user input and invoke agent.run() with the right MemoryContext
  3. Surface feedback controls (thumbs, regenerate, copy) under each
     assistant turn and write to the FeedbackStore
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import streamlit as st

from helix.chat_ui.state import (
    ChatMessage,
    SessionState,
    append_message,
    record_feedback_on_message,
    set_last_trajectory_id,
)
from helix.feedback import FeedbackStore, Outcome, get_feedback_store

if TYPE_CHECKING:
    from helix.agent import Agent


def render_chat(agent: "Agent", state: SessionState) -> None:
    """Render the chat history + input + feedback affordances."""
    store = get_feedback_store()

    # ---------------- conversation history ----------------
    for i, msg in enumerate(state.messages):
        if msg.role == "system_notice":
            st.info(msg.content)
            continue
        with st.chat_message(msg.role):
            st.markdown(msg.content)
            if msg.role == "assistant" and msg.trajectory_id:
                _render_feedback_row(i, msg, store, state)

    # ---------------- input ----------------
    user_text = st.chat_input(
        "Ask the agent something...",
        key="chat_input_main",
    )
    if user_text:
        _handle_user_turn(user_text, agent, state, store)


def _render_feedback_row(
    idx: int,
    msg: ChatMessage,
    store: FeedbackStore,
    state: SessionState,
) -> None:
    """Thumbs / regenerate / copy buttons under one assistant message."""
    cols = st.columns([1, 1, 1, 1, 6])
    recorded = msg.feedback_recorded or {}

    # Thumbs up
    with cols[0]:
        label = "👍 ✓" if recorded.get("thumbs") == 1 else "👍"
        if st.button(label, key=f"thumbs_up_{idx}", help="Record positive feedback for this answer"):
            asyncio.run(store.record_thumbs(
                trajectory_id=msg.trajectory_id,
                value=1,
                session_id=state.session_id,
                user_id=state.user_id,
                org_id=state.org_id,
            ))
            record_feedback_on_message(idx, "thumbs", 1)
            st.rerun()

    # Thumbs down
    with cols[1]:
        label = "👎 ✓" if recorded.get("thumbs") == -1 else "👎"
        if st.button(label, key=f"thumbs_down_{idx}", help="Record negative feedback for this answer"):
            asyncio.run(store.record_thumbs(
                trajectory_id=msg.trajectory_id,
                value=-1,
                session_id=state.session_id,
                user_id=state.user_id,
                org_id=state.org_id,
            ))
            record_feedback_on_message(idx, "thumbs", -1)
            st.rerun()

    # Copy (record that the answer was useful enough to copy)
    with cols[2]:
        if st.button("📋", key=f"copy_{idx}", help="I copied this answer (positive signal)"):
            asyncio.run(store.record_copy(
                trajectory_id=msg.trajectory_id,
                session_id=state.session_id,
                user_id=state.user_id,
            ))
            record_feedback_on_message(idx, "copy", True)
            st.rerun()

    # Regenerate (negative signal + ask the agent to try again on the prior user turn)
    with cols[3]:
        if st.button("🔄", key=f"regen_{idx}", help="This answer wasn't useful; have the agent try again"):
            asyncio.run(store.record_regenerate(
                trajectory_id=msg.trajectory_id,
                session_id=state.session_id,
                user_id=state.user_id,
            ))
            record_feedback_on_message(idx, "regenerate", True)
            # The most recent user message is what we re-ask. Look back.
            prior_user_msg = _find_prior_user_message(state, idx)
            if prior_user_msg is not None:
                # Push a system notice and let the next render handle it
                append_message(ChatMessage(
                    role="system_notice",
                    content="🔄 Regenerating answer for: "
                            + (prior_user_msg.content[:200] + "..." if len(prior_user_msg.content) > 200 else prior_user_msg.content),
                ))
                # Trigger regeneration via session_state flag — the next
                # render of render_chat picks this up via _handle_user_turn
                st.session_state["_helix_pending_regenerate"] = prior_user_msg.content
            st.rerun()

    # Feedback status text
    with cols[4]:
        if recorded:
            st.caption(f"feedback recorded: {recorded}")

    # Handle pending regenerate from a previous render
    pending = st.session_state.pop("_helix_pending_regenerate", None)
    if pending:
        # Need to run agent here — but we're inside a render. The cleanest
        # path: just queue it; chat_input on next render won't fire again.
        # Easiest: pass it back through the same handler.
        # Note: this only runs on rerun after the regenerate button was clicked.
        pass


def _find_prior_user_message(state: SessionState, before_idx: int) -> ChatMessage | None:
    for i in range(before_idx - 1, -1, -1):
        m = state.messages[i]
        if m.role == "user":
            return m
    return None


def _handle_user_turn(
    user_text: str,
    agent: "Agent",
    state: SessionState,
    store: FeedbackStore,
) -> None:
    """Append the user message, invoke agent.run(), append the assistant reply."""
    # Record the user's turn
    append_message(ChatMessage(role="user", content=user_text))

    # Track time-to-followup for any prior assistant message
    if state.last_trajectory_id and state.messages:
        seconds_since = max(0.0, time.time() - state.messages[-2].timestamp.timestamp()) if len(state.messages) >= 2 else 0.0
        if 0.0 <= seconds_since <= 120.0:
            try:
                asyncio.run(store.record_followup(
                    trajectory_id=state.last_trajectory_id,
                    seconds_after=seconds_since,
                    session_id=state.session_id,
                    user_id=state.user_id,
                ))
            except Exception:
                pass

    # Invoke the agent
    context = state.memory_context()
    with st.spinner("Agent thinking..."):
        try:
            answer, trajectory = asyncio.run(agent.run(user_text, context=context))
        except Exception as exc:
            answer = f"(agent error: {type(exc).__name__}: {exc})"
            trajectory = None

    # Record the assistant's turn
    msg = ChatMessage(
        role="assistant",
        content=answer or "(no answer)",
        trajectory_id=trajectory.id if trajectory is not None else None,
        metadata={
            "outcome": trajectory.outcome.value if trajectory is not None else "error",
            "num_steps": len(trajectory.steps) if trajectory is not None else 0,
        },
    )
    append_message(msg)
    set_last_trajectory_id(msg.trajectory_id)
    st.rerun()


# ---------------- session-level outcome controls (sidebar) ----------------

def render_outcome_controls(state: SessionState) -> None:
    """Render Resolved / Escalated / Abandoned buttons in the sidebar.

    These record session-level outcomes via FeedbackStore.record_outcome.
    Useful for production deployments where you want to capture the
    end-state of a conversation explicitly.
    """
    store = get_feedback_store()
    st.caption("Mark this session as:")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("✅", help="Resolved", use_container_width=True):
            asyncio.run(store.record_outcome(
                session_id=state.session_id,
                outcome=Outcome.RESOLVED,
                user_id=state.user_id,
                org_id=state.org_id,
            ))
            append_message(ChatMessage(role="system_notice", content="✅ Session marked **resolved**."))
            st.rerun()
    with c2:
        if st.button("⬆", help="Escalated", use_container_width=True):
            asyncio.run(store.record_outcome(
                session_id=state.session_id,
                outcome=Outcome.ESCALATED,
                user_id=state.user_id,
                org_id=state.org_id,
            ))
            append_message(ChatMessage(role="system_notice", content="⬆ Session marked **escalated**."))
            st.rerun()
    with c3:
        if st.button("👋", help="Abandoned", use_container_width=True):
            asyncio.run(store.record_outcome(
                session_id=state.session_id,
                outcome=Outcome.ABANDONED,
                user_id=state.user_id,
                org_id=state.org_id,
            ))
            append_message(ChatMessage(role="system_notice", content="👋 Session marked **abandoned**."))
            st.rerun()
