"""Helix dashboard entry point.

Run:
    streamlit run helix/dashboard/app.py

Or with a specific archive:
    streamlit run helix/dashboard/app.py -- --archive chapters/ch02/runs/helix_archive.sqlite

The sidebar lets the user switch archive paths at runtime, which is handy
when comparing chapters or pre/post-improvement runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helix.dashboard.data import clear_caches
from helix.dashboard.pages import (
    render_compare,
    render_lineage,
    render_overview,
    render_replay,
    render_rounds,
    render_verdicts,
)


DEFAULT_ARCHIVE = REPO_ROOT / "chapters" / "ch02" / "runs" / "helix_archive.sqlite"


def main() -> None:
    st.set_page_config(
        page_title="Helix Dashboard",
        page_icon=":dna:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ---------------- sidebar ----------------
    with st.sidebar:
        st.title(":dna: Helix")
        st.caption("Explorer for self-improving agent runs")
        st.markdown("---")

        st.subheader("Archive")
        archive_path = st.text_input(
            "Path",
            value=str(DEFAULT_ARCHIVE),
            help="Path to a helix_archive.sqlite file.",
        )

        if not Path(archive_path).exists():
            st.error(f"Archive not found at:\n{archive_path}")
            st.stop()

        col1, col2 = st.columns(2)
        with col1:
            if st.button(":arrows_counterclockwise: Refresh", use_container_width=True):
                clear_caches()
                st.rerun()
        with col2:
            page = st.radio(
                "Page",
                ["Overview", "Lineage", "Compare", "Verdicts", "Replay", "Rounds"],
                label_visibility="collapsed",
                index=0,
            )

        st.markdown("---")
        st.caption(
            "Helix is a platform for self-improving agents. "
            "This dashboard mines the archive + trajectory cache + round logs "
            "to make a run inspectable end-to-end."
        )

    # ---------------- main page ----------------
    st.title(f"Helix Dashboard")
    st.caption(f"Archive: `{archive_path}`")

    if page == "Overview":
        render_overview(archive_path)
    elif page == "Lineage":
        render_lineage(archive_path)
    elif page == "Compare":
        render_compare(archive_path)
    elif page == "Verdicts":
        render_verdicts(archive_path)
    elif page == "Replay":
        render_replay(archive_path)
    elif page == "Rounds":
        render_rounds(archive_path)


if __name__ == "__main__":
    main()
