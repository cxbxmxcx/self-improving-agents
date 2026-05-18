"""Dashboard panel renderers.

Named `panels` rather than `pages` to keep Streamlit's auto-discovery of
multi-page apps from picking these up as top-level pages. The dashboard
uses a single-page app pattern with manual routing via a sidebar radio.
"""

from helix.dashboard.panels.champion import render_champion
from helix.dashboard.panels.compare import render_compare
from helix.dashboard.panels.lineage import render_lineage
from helix.dashboard.panels.overview import render_overview
from helix.dashboard.panels.replay import render_replay
from helix.dashboard.panels.rounds import render_rounds
from helix.dashboard.panels.verdicts import render_verdicts

__all__ = [
    "render_champion",
    "render_compare",
    "render_lineage",
    "render_overview",
    "render_replay",
    "render_rounds",
    "render_verdicts",
]
