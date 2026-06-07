"""Round history page: timeline of rounds with strategy switches."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from helix.dashboard.data import filter_variants_by_focus, list_round_log, list_variants


def render_rounds(archive_path: str, focus: str | None = None) -> None:
    rounds = list_round_log(archive_path)
    variants = list_variants(archive_path)
    if focus:
        variants = filter_variants_by_focus(variants, focus)

    if not rounds and not variants:
        st.info("No round log and no archive contents.")
        return

    st.subheader("🕐 Round history")
    caption = "Per-round score progression and promotion outcomes over time."
    if focus:
        caption += f" Archive timeline filtered to `{focus}`."
    st.caption(caption)

    # If we have a JSONL round log (from 05_spo_offline_loop.py), use it; otherwise
    # synthesize a timeline from the archive's measurement history.
    if rounds:
        _render_from_round_log(rounds)
    else:
        _render_from_archive(variants)


def _render_from_round_log(rounds: list[dict]) -> None:
    df = pd.DataFrame(rounds)

    # ---------------- summary metrics ----------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total rounds", len(df))
    promoted = df["promoted"].sum() if "promoted" in df.columns else 0
    c2.metric("Promotions", int(promoted))
    if "mean_score" in df.columns:
        c3.metric("Best score", f"{df['mean_score'].max():.3f}")
        c4.metric("Mean score", f"{df['mean_score'].mean():.3f}")

    st.markdown("---")

    # ---------------- score timeline ----------------
    if "mean_score" in df.columns and "round" in df.columns:
        st.subheader("📈 Score progression")
        fig = px.line(
            df,
            x="round",
            y="mean_score",
            markers=True,
            title="Candidate mean score by round",
            labels={"round": "Round", "mean_score": "Candidate score"},
        )
        if "promoted" in df.columns:
            promoted_rounds = df[df["promoted"]]
            if len(promoted_rounds) > 0:
                fig.add_trace(go.Scatter(
                    x=promoted_rounds["round"],
                    y=promoted_rounds["mean_score"],
                    mode="markers",
                    marker={"size": 14, "color": "#00A087", "symbol": "star"},
                    name="Promoted",
                ))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # ---------------- per-round table ----------------
    st.subheader("📜 Round details")
    cols = [c for c in [
        "round", "candidate_version", "mean_score", "win_rate",
        "n_wins", "n_losses", "n_ties", "promoted",
        "elapsed_reference_sec", "elapsed_candidate_sec", "elapsed_judge_sec",
    ] if c in df.columns]
    st.dataframe(df[cols], use_container_width=True, hide_index=True)


def _render_from_archive(variants: list[dict]) -> None:
    """Fallback when there's no round log: synthesize from archive measurements."""
    scored = [v for v in variants if v["measurement"] and v["measurement"]["score"] is not None]
    if not scored:
        st.info("No measured artifacts in archive.")
        return

    df = pd.DataFrame([
        {
            "version": v["version"],
            "created_by": v["created_by"],
            "search_method": v["search_method"],
            "score": v["measurement"]["score"],
        }
        for v in scored
    ])

    st.subheader("📈 Latest score by version")
    fig = px.bar(
        df,
        x="version",
        y="score",
        color="created_by",
        title="Latest measured score per artifact",
        labels={"version": "Version", "score": "Score"},
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📜 All artifacts")
    st.dataframe(df.sort_values("version"), use_container_width=True, hide_index=True)
