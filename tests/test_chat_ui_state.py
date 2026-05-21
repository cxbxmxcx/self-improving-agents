"""Chat UI state helper tests.

Streamlit's st.session_state behaves like a dict outside a runtime; these
tests exercise the state.py helpers without actually launching streamlit.
"""

from __future__ import annotations

import pytest
import streamlit as st

from helix.chat_ui.state import (
    ChatMessage,
    append_message,
    get_session_state,
    record_feedback_on_message,
    reset_conversation,
    set_agent_name,
    set_archive_path,
    set_org_id,
    set_session_id,
    set_user_id,
)


@pytest.fixture(autouse=True)
def clean_session_state():
    """Clear st.session_state between tests so they don't leak into each other."""
    # st.session_state in tests is a dict-like; clear all keys we manage
    keys = [
        "_helix_session_id", "_helix_user_id", "_helix_org_id",
        "_helix_agent_name", "_helix_messages", "_helix_last_trajectory_id",
        "_helix_pending_input", "_helix_improver_attached", "_helix_archive_path",
    ]
    for k in keys:
        if k in st.session_state:
            del st.session_state[k]
    yield


def test_get_session_state_initializes_defaults():
    state = get_session_state()
    assert state.session_id.startswith("sess-")
    assert state.user_id == "anonymous"
    assert state.org_id == "default"
    assert state.agent_name == "helpdesk"
    assert state.messages == []
    assert state.last_trajectory_id is None
    assert state.improver_attached is False


def test_set_user_id_updates_state():
    get_session_state()
    set_user_id("alice")
    state = get_session_state()
    assert state.user_id == "alice"


def test_set_org_id_updates_state():
    get_session_state()
    set_org_id("acme")
    state = get_session_state()
    assert state.org_id == "acme"


def test_set_agent_name_updates_state():
    get_session_state()
    set_agent_name("custom")
    state = get_session_state()
    assert state.agent_name == "custom"


def test_set_archive_path_updates_state():
    get_session_state()
    set_archive_path("/tmp/archive.sqlite")
    state = get_session_state()
    assert state.archive_path == "/tmp/archive.sqlite"


def test_append_message_grows_log():
    get_session_state()
    append_message(ChatMessage(role="user", content="hello"))
    append_message(ChatMessage(role="assistant", content="hi"))
    state = get_session_state()
    assert len(state.messages) == 2
    assert state.messages[0].role == "user"
    assert state.messages[1].role == "assistant"


def test_reset_conversation_rotates_session_id_and_clears_messages():
    get_session_state()
    original_sid = get_session_state().session_id
    set_user_id("alice")
    append_message(ChatMessage(role="user", content="hi"))

    reset_conversation()

    new_state = get_session_state()
    assert new_state.session_id != original_sid
    assert new_state.session_id.startswith("sess-")
    assert new_state.messages == []
    # user_id should persist across session reset
    assert new_state.user_id == "alice"


def test_memory_context_translates_identity():
    get_session_state()
    set_user_id("alice")
    set_org_id("acme")
    state = get_session_state()
    ctx = state.memory_context()
    assert ctx.session_id == state.session_id
    assert ctx.user_id == "alice"
    assert ctx.org_id == "acme"


def test_memory_context_omits_empty_user_id():
    """Empty strings should not become MemoryContext fields."""
    get_session_state()
    set_user_id("")
    state = get_session_state()
    ctx = state.memory_context()
    assert ctx.user_id is None


def test_record_feedback_on_message_marks_message():
    get_session_state()
    append_message(ChatMessage(role="assistant", content="answer", trajectory_id="t1"))
    record_feedback_on_message(0, "thumbs", 1)
    state = get_session_state()
    assert state.messages[0].feedback_recorded == {"thumbs": 1}


def test_record_feedback_on_invalid_index_is_safe():
    get_session_state()
    # No messages; should not raise
    record_feedback_on_message(0, "thumbs", 1)
    record_feedback_on_message(-1, "thumbs", 1)


def test_set_session_id_overrides_default():
    get_session_state()
    set_session_id("custom-sid")
    state = get_session_state()
    assert state.session_id == "custom-sid"
