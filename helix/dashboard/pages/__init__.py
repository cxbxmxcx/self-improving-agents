"""Dashboard page renderers. Each module exports a render_<name>(archive_path) function."""

from helix.dashboard.pages.compare import render_compare
from helix.dashboard.pages.lineage import render_lineage
from helix.dashboard.pages.overview import render_overview
from helix.dashboard.pages.replay import render_replay
from helix.dashboard.pages.rounds import render_rounds
from helix.dashboard.pages.verdicts import render_verdicts

__all__ = [
    "render_compare",
    "render_lineage",
    "render_overview",
    "render_replay",
    "render_rounds",
    "render_verdicts",
]
