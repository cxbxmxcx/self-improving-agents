"""Overview page: archive summary + current champion."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from helix.dashboard.data import (
    get_best,
    list_round_log,
    list_variants,
)


def render_overview(archive_path: str) -> None:
    variants = list_variants(archive_path)
    rounds = list_round_log(archive_path)

    if not variants:
        st.info("Archive is empty. Run a chapter script (e.g. `python chapters/ch02/spo_loop.py`) to populate it.")
        return

    # ---------------- top metrics ----------------
    col1, col2, col3, col4 = st.columns(4)
    scored = [v for v in variants if v["measurement"]]
    scores = [v["measurement"]["score"] for v in scored if v["measurement"]["score"] is not None]
    col1.metric("Artifacts", len(variants))
    col2.metric("With measurements", len(scored))
    col3.metric("Top score", f"{max(scores):.3f}" if scores else "—")
    col4.metric("Rounds run", len(rounds))

    st.markdown("---")

    # ---------------- current champion ----------------
    st.subheader(":trophy: Current champion")
    best_list = get_best(archive_path, k=1)
    if best_list:
        champ = best_list[0]
        m = champ["measurement"] or {}
        c1, c2, c3 = st.columns([1, 1, 2])
        c1.metric("Version", f"v{champ['version']}")
        c2.metric("Score", f"{m.get('score', 0):.3f}" if m.get("score") is not None else "—")
        c3.metric("Created by", champ["created_by"])
        st.caption(f"Artifact id: `{champ['id']}`  •  content_hash: `{champ['content_hash'][:12]}`")

        with st.expander(":scroll: Champion prompt content", expanded=True):
            content = champ["content"]
            if isinstance(content, str):
                st.code(content, language="text")
            else:
                st.json(content)
    else:
        st.info("No measured artifacts in archive yet.")

    st.markdown("---")

    # ---------------- by-method breakdown ----------------
    st.subheader(":hammer_and_wrench: Artifacts by Search method")
    df = pd.DataFrame([
        {
            "version": v["version"],
            "created_by": v["created_by"],
            "search_method": v["search_method"],
            "score": v["measurement"]["score"] if v["measurement"] else None,
            "parent": f"v{v['parent_version']}" if v["parent_version"] else "genesis",
        }
        for v in variants
    ])
    by_method = df.groupby("created_by").agg(
        count=("version", "count"),
        mean_score=("score", "mean"),
        max_score=("score", "max"),
    ).reset_index()
    st.dataframe(by_method, use_container_width=True, hide_index=True)

    # ---------------- recent rounds ----------------
    if rounds:
        st.markdown("---")
        st.subheader(":bar_chart: Recent rounds")
        round_df = pd.DataFrame(rounds)
        cols = [c for c in [
            "round", "candidate_version", "mean_score", "win_rate",
            "n_wins", "n_losses", "n_ties", "promoted",
        ] if c in round_df.columns]
        st.dataframe(round_df[cols].tail(10), use_container_width=True, hide_index=True)
