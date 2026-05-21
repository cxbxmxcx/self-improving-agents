"""Per-question verdicts panel: filter and drill into judge decisions."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from helix.dashboard.data import (
    get_failed_trajectories,
    list_question_verdicts,
    parse_focus,
)


def render_verdicts(archive_path: str, focus: str | None = None) -> None:
    verdicts = list_question_verdicts(archive_path)
    if not verdicts:
        st.info("No per-question verdicts in the archive yet.")
        return

    st.subheader("⚖ Per-question verdicts")
    caption = (
        "Every pairwise judge decision recorded across every round. "
        "LEFT = candidate won; RIGHT = reference won; TIE = neither."
    )
    if focus:
        caption = f"Focused on `{focus}`. " + caption
    st.caption(caption)

    df = pd.DataFrame(verdicts)

    # Mark verdicts where the candidate's underlying trajectory failed.
    failed = get_failed_trajectories(archive_path)
    failed_keys = {(f["artifact_id"], f["version"], f["question_id"]) for f in failed}
    df["candidate_failed"] = df.apply(
        lambda r: (r["artifact_id"], int(r["version"]), r["question_id"]) in failed_keys,
        axis=1,
    )
    n_failed_source = int(df["candidate_failed"].sum())
    if n_failed_source > 0:
        st.warning(
            f"⚠ **{n_failed_source} of {len(df)} verdicts** were judged from "
            "trajectories that failed (empty agent answer). These verdicts effectively "
            "mean 'the candidate produced nothing,' not 'the candidate produced something "
            "worse than the reference.' Purge failed trajectories from the Replay panel "
            "and re-run rounds to get clean verdicts."
        )

    # Apply focus filter
    parsed = parse_focus(focus)
    if parsed is not None:
        fid, _kind = parsed
        df = df[df["artifact_id"] == fid]
        if df.empty:
            st.info("No verdicts for this focus.")
            return

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
        "preference", "confidence", "candidate_failed", "role",
        "feedback", "recorded_at",
    ]
    available_cols = [c for c in display_cols if c in filtered.columns]
    st.dataframe(
        filtered[available_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "confidence": st.column_config.NumberColumn(format="%.2f"),
            "candidate_failed": st.column_config.CheckboxColumn(
                "⚠ failed",
                help="If True, the underlying trajectory was a failed run "
                "(empty answer) — the verdict reflects 'no answer' not "
                "'worse answer'.",
            ),
            "feedback": st.column_config.TextColumn(width="large"),
        },
    )

    # ---------------- per-question heatmap ----------------
    st.markdown("---")
    st.subheader("📈 Verdict heatmap by question x version")
    pivot = (
        df.groupby(["question_id", "version"])["preference"]
        .agg(lambda s: s.iloc[-1])  # latest verdict per (question, version)
        .unstack(fill_value="—")
    )
    st.dataframe(pivot, use_container_width=True)
