"""Helix chat UI.

A platform-level Streamlit chat reference that operates any agent spec
(agents/<name>.py) with the full Helix stack: memory tiers, feedback
collection, and an optionally-attached Improver running in the background.

Launch via:
    streamlit run helix/chat_ui/app.py

The UI is intentionally minimal so chapter scripts can either run it as-is
or wrap it with chapter-specific demonstrations. Multi-agent demos in later
chapters select among multiple agent specs from the same UI.
"""

from helix.chat_ui.state import (
    ChatMessage,
    SessionState,
    get_session_state,
)

__all__ = ["ChatMessage", "SessionState", "get_session_state"]
