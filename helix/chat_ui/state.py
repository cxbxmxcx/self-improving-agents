"""Session-state helpers for the chat UI.

Streamlit reruns the script on every interaction; persistent state goes in
st.session_state. This module centralizes the keys so panel code doesn't
sprinkle them everywhere.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import streamlit as st

from helix.memory.base import MemoryContext


# Session-state keys
_KEY_SESSION_ID = "_helix_session_id"
_KEY_USER_ID = "_helix_user_id"
_KEY_ORG_ID = "_helix_org_id"
_KEY_AGENT_NAME = "_helix_agent_name"
_KEY_MESSAGES = "_helix_messages"
_KEY_LAST_TRAJECTORY_ID = "_helix_last_trajectory_id"
_KEY_PENDING_INPUT = "_helix_pending_input"
_KEY_IMPROVER_ATTACHED = "_helix_improver_attached"
_KEY_ARCHIVE_PATH = "_helix_archive_path"
_KEY_LAST_KNOWN_CHAMPION = "_helix_last_known_champion"


@dataclass
class ChatMessage:
    """One turn in the conversation visible to the user.

    Not the same as the LLM messages array — this is the UI-facing log.
    Each ChatMessage may correspond to multiple LLM steps (tool calls,
    intermediate model calls) but only the user-visible content is here.
    """

    role: str  # "user" | "assistant" | "system_notice"
    content: str
    trajectory_id: str | None = None
    feedback_recorded: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionState:
    """Typed view over the relevant st.session_state keys.

    Accessed via get_session_state() — that helper initializes defaults on
    first read and writes back to st.session_state. Treat it as ephemeral;
    it's reconstructed on every Streamlit rerun.
    """

    session_id: str
    user_id: str
    org_id: str
    agent_name: str
    archive_path: str
    messages: list[ChatMessage]
    last_trajectory_id: str | None
    improver_attached: bool

    def memory_context(self) -> MemoryContext:
        """Translate the session's identity into a MemoryContext for agent.run()."""
        return MemoryContext(
            session_id=self.session_id or None,
            user_id=self.user_id or None,
            org_id=self.org_id or None,
        )


def get_session_state() -> SessionState:
    """Return the typed session state, initializing defaults on first call."""
    if _KEY_SESSION_ID not in st.session_state:
        st.session_state[_KEY_SESSION_ID] = f"sess-{uuid.uuid4().hex[:8]}"
    if _KEY_USER_ID not in st.session_state:
        st.session_state[_KEY_USER_ID] = "anonymous"
    if _KEY_ORG_ID not in st.session_state:
        st.session_state[_KEY_ORG_ID] = "default"
    if _KEY_AGENT_NAME not in st.session_state:
        st.session_state[_KEY_AGENT_NAME] = "helpdesk"
    if _KEY_ARCHIVE_PATH not in st.session_state:
        st.session_state[_KEY_ARCHIVE_PATH] = ""
    if _KEY_MESSAGES not in st.session_state:
        st.session_state[_KEY_MESSAGES] = []
    if _KEY_LAST_TRAJECTORY_ID not in st.session_state:
        st.session_state[_KEY_LAST_TRAJECTORY_ID] = None
    if _KEY_IMPROVER_ATTACHED not in st.session_state:
        st.session_state[_KEY_IMPROVER_ATTACHED] = False

    return SessionState(
        session_id=st.session_state[_KEY_SESSION_ID],
        user_id=st.session_state[_KEY_USER_ID],
        org_id=st.session_state[_KEY_ORG_ID],
        agent_name=st.session_state[_KEY_AGENT_NAME],
        archive_path=st.session_state[_KEY_ARCHIVE_PATH],
        messages=st.session_state[_KEY_MESSAGES],
        last_trajectory_id=st.session_state[_KEY_LAST_TRAJECTORY_ID],
        improver_attached=st.session_state[_KEY_IMPROVER_ATTACHED],
    )


# ---------------- mutation helpers ----------------

def set_session_id(value: str) -> None:
    st.session_state[_KEY_SESSION_ID] = value


def set_user_id(value: str) -> None:
    st.session_state[_KEY_USER_ID] = value


def set_org_id(value: str) -> None:
    st.session_state[_KEY_ORG_ID] = value


def set_agent_name(value: str) -> None:
    st.session_state[_KEY_AGENT_NAME] = value


def set_archive_path(value: str) -> None:
    st.session_state[_KEY_ARCHIVE_PATH] = value


def append_message(msg: ChatMessage) -> None:
    st.session_state.setdefault(_KEY_MESSAGES, []).append(msg)


def reset_conversation() -> None:
    """Clear messages and rotate the session_id. User_id/org_id persist."""
    st.session_state[_KEY_MESSAGES] = []
    st.session_state[_KEY_SESSION_ID] = f"sess-{uuid.uuid4().hex[:8]}"
    st.session_state[_KEY_LAST_TRAJECTORY_ID] = None


def set_last_trajectory_id(value: str | None) -> None:
    st.session_state[_KEY_LAST_TRAJECTORY_ID] = value


def set_improver_attached(value: bool) -> None:
    st.session_state[_KEY_IMPROVER_ATTACHED] = value


def record_feedback_on_message(idx: int, kind: str, value: Any) -> None:
    """Mark a message as having received feedback so the UI shows it."""
    msgs = st.session_state.get(_KEY_MESSAGES, [])
    if 0 <= idx < len(msgs):
        msgs[idx].feedback_recorded[kind] = value


# ---------------- champion tracking ----------------

def get_last_known_champion() -> tuple[str, int] | None:
    """Return the (id, version) the chat thinks is the current champion,
    or None if no champion has been observed yet."""
    return st.session_state.get(_KEY_LAST_KNOWN_CHAMPION)


def set_last_known_champion(ref: tuple[str, int] | None) -> None:
    st.session_state[_KEY_LAST_KNOWN_CHAMPION] = ref
