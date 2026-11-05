"""Put the book repository's root on this environment's import path.

helix is not a pip-installable package: the book repository expects its root
on sys.path. This script writes a .pth file into the active environment's
site-packages so `import helix` (and the book's `chapters` package, which the
parity tests replay) resolve to that checkout.

    python scripts/link_helix.py                  # the book repository this project sits in
    python scripts/link_helix.py C:/path/to/self-improving-agents

Without an argument it uses HELIX_ROOT, else the nearest folder above this
project that holds helix (the book repository, when the project lives in it),
else a sibling checkout named self-improving-agents.
"""

from __future__ import annotations

import os
import sys
import sysconfig
from pathlib import Path

PTH_NAME = "helix_book_repo.pth"
PROJECT = Path(__file__).resolve().parents[1]


def _is_book(root: Path) -> bool:
    return (root / "helix" / "agent.py").is_file()


def default_root() -> Path:
    if os.environ.get("HELIX_ROOT"):
        return Path(os.environ["HELIX_ROOT"])
    for parent in PROJECT.parents:
        if _is_book(parent):
            return parent
    return PROJECT.parent / "self-improving-agents"


def main() -> int:
    root = (Path(sys.argv[1]) if len(sys.argv) > 1 else default_root()).resolve()
    if not _is_book(root):
        print(f"not a checkout of the book repository (no helix/agent.py): {root}", file=sys.stderr)
        return 1
    if sys.prefix == sys.base_prefix:
        print("refusing to edit the system Python; activate the project's virtual environment first",
              file=sys.stderr)
        return 1
    site_packages = Path(sysconfig.get_paths()["purelib"])
    pth = site_packages / PTH_NAME
    pth.write_text(f"{root}\n", encoding="utf-8")
    print(f"linked {root}\n  via {pth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
