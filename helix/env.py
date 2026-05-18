"""Load environment variables from .env at process start.

Every chapter script and harness calls `load_env()` once before doing anything
that needs an API key. Reads a .env file from the repo root (if present) into
os.environ. LiteLLM then picks up provider keys by their canonical names.

Expected variable names per LiteLLM:
  - OPENAI_API_KEY
  - ANTHROPIC_API_KEY
  - GOOGLE_API_KEY / GEMINI_API_KEY
  - COHERE_API_KEY
  - GROQ_API_KEY
  - MISTRAL_API_KEY
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

_loaded = False


def load_env(dotenv_path: Path | None = None) -> None:
    """Load .env into os.environ. Idempotent."""
    global _loaded
    if _loaded:
        return

    from dotenv import load_dotenv

    path = dotenv_path or (REPO_ROOT / ".env")
    if path.exists():
        load_dotenv(path)

    _loaded = True
