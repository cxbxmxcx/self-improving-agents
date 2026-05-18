"""Per-question verdicts page: filter and drill into judge decisions."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from helix.dashboard.data import list_question_verdicts


def render_verdicts(archive_path: str) -> None:
    verdicts = list_question_verdicts(archive_path)
    if not verdicts:
        st.info("No per-question verdicts in the archive yet.")
        return

    st.subheader(":judge: Per-question verdicts")
    st.caption(
        "Every pairwise judge decision recorded across every round. "
        "LEFT = candidate won; RIGHT = reference won; TIE = neither."
    )

    df = pd.DataFrame(verdicts)

    # ---------------- filters ----------------
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        version_filter = st.multiselect(
            "Version",
            sorted(df["version"].unique()),
            default=[],
            help="Filter by which candidate artifact was being judged.",
        )
    with col2:
        band_filter = st.multiselect(
            "Band",
            sorted(df["band"].unique()),
            default=[],
        )
    with col3:
        preference_filter = st.multiselect(
            "Preference",
            sorted(df["preference"].unique()),
            default=[],
        )
    with col4:
        role_filter = st.multiselect(
            "Role",
            sorted(df["role"].unique()),
            default=[],
        )

    filtered = df.copy()
    if version_filter:
        filtered = filtered[filtered["version"].isin(version_filter)]
    if band_filter:
        filtered = filtered[filtered["band"].isin(band_filter)]
    if preference_filter:
        filtered = filtered[filtered["preference"].isin(preference_filter)]
    if role_filter:
        filtered = filtered[filtered["role"].isin(role_filter)]

    st.caption(f"Showing {len(filtered)} of {len(df)} verdicts.")

    # ---------------- summary ----------------
    if len(filtered) > 0:
        pref_counts = filtered["preference"].value_counts()
        cols = st.columns(min(len(pref_counts), 4))
        for i, (pref, count) in enumerate(pref_counts.items()):
            cols[i % len(cols)].metric(pref, count)

    # ---------------- table ----------------
    display_cols = [
        "version", "created_by", "question_id", "band",
        "preference", "confidence", "role", "feedback", "recorded_at",
    ]
    available_cols = [c for c in display_cols if c in filtered.columns]
    st.dataframe(
        filtered[available_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "confidence": st.column_config.NumberColumn(format="%.2f"),
            "feedback": st.column_config.TextColumn(width="large"),
        },
    )

    # ---------------- per-question heatmap ----------------
    st.markdown("---")
    st.subheader(":chart_with_upwards_trend: Verdict heatmap by question x version")
    pivot = (
        df.groupby(["question_id", "version"])["preference"]
        .agg(lambda s: s.iloc[-1])  # latest verdict per (question, version)
        .unstack(fill_value="—")
    )
    st.dataframe(pivot, use_container_width=True)
