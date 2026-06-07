"""Helix dashboard entry point.

Run:
    streamlit run helix/dashboard/app.py

Sidebar flow:
  1. Enter a path to a helix_archive.sqlite file.
  2. Click "Load archive". Until then, the main area shows a placeholder.
  3. Once loaded, an Artifact Focus selector appears. Pick "all" or a
     specific (artifact_id, kind) pair. Most panels respect the focus.
  4. Pick a panel: Overview, Lineage, Compare, Verdicts, Replay, Rounds.
  5. Refresh to clear caches when a new run has populated the archive.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helix.dashboard.data import (
    clear_caches,
    count_failed_trajectories,
    list_artifact_focus_options,
    list_variants,
)
from helix.dashboard.panels import (
    render_champion,
    render_compare,
    render_lineage,
    render_overview,
    render_replay,
    render_rounds,
    render_verdicts,
)


DEFAULT_ARCHIVE = REPO_ROOT / "chapters" / "ch02" / "runs" / "helix_archive.sqlite"

FOCUS_ALL = "(all artifacts)"


def _placeholder_screen() -> None:
    st.markdown(
        """
        ## Welcome to Helix

        Helix is the platform powering *Self-Improving Agents* (Lanham, Manning).
        This dashboard mines a Helix archive to show what self-improving agents
        actually did during a run.

        **To get started:**

        1. In the sidebar, point the path at an existing `helix_archive.sqlite`
           file. The default location is `chapters/ch02/runs/helix_archive.sqlite`.
        2. Click **Load archive**.
        3. Pick an artifact focus (or "all artifacts") and a panel to explore.

        **No archive yet?** Run a chapter script first:

        ```
        python chapters/ch02/05_spo_offline_loop.py
        ```

        ...or, for a richer Ch 3 run with two Search methods:

        ```
        python chapters/ch03/escalating_improver.py
        ```
        """
    )


def main() -> None:
    st.set_page_config(
        page_title="Helix Dashboard",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ---------------- sidebar: archive picker + load ----------------
    with st.sidebar:
        st.title("🧬 Helix")
        st.caption("Explorer for self-improving agent runs")
        st.markdown("---")

        st.subheader("Archive")
        archive_path = st.text_input(
            "Path",
            value=st.session_state.get("archive_path", str(DEFAULT_ARCHIVE)),
            help="Path to a helix_archive.sqlite file.",
            key="archive_path_input",
        )

        load_col, refresh_col = st.columns(2)
        with load_col:
            if st.button("📂 Load", use_container_width=True, type="primary"):
                if Path(archive_path).exists():
                    clear_caches()
                    st.session_state["loaded_archive"] = archive_path
                    # reset focus on a new load
                    st.session_state.pop("artifact_focus", None)
                    st.rerun()
                else:
                    st.error("File not found at that path.")

        with refresh_col:
            if st.button("🔄 Refresh", use_container_width=True,
                         disabled=not st.session_state.get("loaded_archive")):
                clear_caches()
                st.rerun()

        loaded_archive = st.session_state.get("loaded_archive")

        # ---------------- failed-runs badge (always visible after load) -----
        if loaded_archive:
            n_failed = count_failed_trajectories(loaded_archive)
            if n_failed > 0:
                st.markdown(
                    f'<div style="background:#E64B35; color:white; padding:8px 12px; '
                    f'border-radius:6px; margin-top:8px; font-weight:600; font-size:0.9em;">'
                    f'⚠ {n_failed} failed runs'
                    f'<br><span style="font-weight:normal; font-size:0.85em;">'
                    f'See Replay panel to inspect or purge.'
                    f'</span></div>',
                    unsafe_allow_html=True,
                )

        # ---------------- artifact focus (visible only after load) ----------------
        focus_choice = FOCUS_ALL
        if loaded_archive:
            st.markdown("---")
            st.subheader("Artifact focus")
            options = [FOCUS_ALL] + list_artifact_focus_options(loaded_archive)
            current = st.session_state.get("artifact_focus", FOCUS_ALL)
            if current not in options:
                current = FOCUS_ALL
            focus_choice = st.selectbox(
                "Filter most panels to one artifact",
                options,
                index=options.index(current),
                help=(
                    "Helix supports improving multiple artifact types in parallel "
                    "(prompts, skills, tool descriptions, rubrics, etc). Pick one "
                    "to scope panels like Lineage, Compare, and Replay. The "
                    "Overview panel ignores this and shows the whole archive."
                ),
            )
            st.session_state["artifact_focus"] = focus_choice

            st.markdown("---")
            st.subheader("Panel")
            page = st.radio(
                "Panel",
                ["Overview", "Champion", "Lineage", "Compare", "Verdicts", "Replay", "Rounds"],
                label_visibility="collapsed",
                index=0,
                key="panel_choice",
            )
        else:
            page = None

        st.markdown("---")
        st.caption(
            "Helix mines the archive + trajectory cache + round logs to make a "
            "self-improving run inspectable end-to-end."
        )

    # ---------------- main page ----------------
    if not loaded_archive:
        _placeholder_screen()
        return

    st.title("Helix Dashboard")
    st.caption(
        f"Archive: `{loaded_archive}`  •  Focus: `{focus_choice}`"
    )

    # Translate focus_choice into (artifact_id, kind) or None.
    focus = None if focus_choice == FOCUS_ALL else focus_choice

    if page == "Overview":
        render_overview(loaded_archive, focus=focus)
    elif page == "Champion":
        render_champion(loaded_archive, focus=focus)
    elif page == "Lineage":
        render_lineage(loaded_archive, focus=focus)
    elif page == "Compare":
        render_compare(loaded_archive, focus=focus)
    elif page == "Verdicts":
        render_verdicts(loaded_archive, focus=focus)
    elif page == "Replay":
        render_replay(loaded_archive, focus=focus)
    elif page == "Rounds":
        render_rounds(loaded_archive, focus=focus)


if __name__ == "__main__":
    main()
