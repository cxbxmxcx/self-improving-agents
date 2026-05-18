"""Repository-root conftest so `import helix` works from anywhere.

Adds the repo root to sys.path. This is the one packaging concession the repo
makes: chapter scripts and tests can write `from helix.agent import Agent`
without any pip install step. Keeps the book's setup story to "clone and run."
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
