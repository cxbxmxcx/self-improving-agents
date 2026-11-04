"""Launch the Helix Observatory dashboard.

Run from anywhere: python helix_web/run.py [--port 8600] [--archive PATH].
The repo is not an installed package, so the repo root goes on sys.path
first, the same shim every chapter script and the old dashboard use.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Helix Observatory server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8610)
    parser.add_argument(
        "--archive",
        action="append",
        default=[],
        help="extra archive path to register at startup (repeatable)",
    )
    parser.add_argument("--poll-ms", type=int, default=500)
    args = parser.parse_args()

    import uvicorn

    from helix_web.server.app import create_app
    from helix_web.server.config import Settings

    settings = Settings(
        extra_archives=[Path(p) for p in args.archive], poll_ms=args.poll_ms
    )
    uvicorn.run(create_app(settings), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
