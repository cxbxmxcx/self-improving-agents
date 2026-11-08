"""The online metacognition loop's recorder, configured for the calendar assistant.

The recorder, controller, detector, and gate are the framework's
(helix.metacognition). The offline search found a learning mechanism; listing
6.5 runs it while the assistant serves requests. Here the recorder is told how
this assistant embeds text and whose runs it holds.
"""

from __future__ import annotations

from pathlib import Path

from chapters.ch05.personal.encoder import embed
from chapters.ch06.assistant.learning import USER_SCOPE
from helix.metacognition import Recorder


def recorder(path: str | Path = ":memory:") -> Recorder:
    """A recorder for Maya's assistant."""
    return Recorder(path, embedding_fn=embed, scope_key=USER_SCOPE, user_id="maya")
