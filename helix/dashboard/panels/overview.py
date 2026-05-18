"""Overview panel: archive summary + per-kind breakdown + focus-scoped champion.

Always shows the whole archive even when an artifact focus is selected,
because the overview's job is to surface the breadth of activity. The
champion section honors the focus when one is set.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from helix.dashboard.data import (
    filter_variants_by_focus,
    get_failed_trajectories,
    list_cached_trajectories,
    list_round_log,
    list_variants,
)


def render_overview(archive_path: str, focus: str | None = None) -> None:
    all_variants = list_variants(archive_path)
    rounds = list_round_log(archive_path)

    if not all_variants:
        st.info("Archive is empty. Run a chapter script (e.g. `python chapters/ch02/spo_loop.py`) to populate it.")
        return

    # ---------------- top metrics across the whole archive ----------------
    st.subheader(":bar_chart: Archive summary")
    col1, col2, col3, col4 = st.columns(4)
    scored = [v for v in all_variants if v["measurement"]]
    scores = [v["measurement"]["score"] for v in scored if v["measurement"]["score"] is not None]
    col1.metric("Total artifacts", len(all_variants))
    col2.metric("With measurements", len(scored))
    col3.metric("Top score", f"{max(scores):.3f}" if scores else "—")
    col4.metric("Rounds run", len(rounds))

    # ---------------- per-kind breakdown ----------------
    st.markdown("---")
    st.subheader(":file_folder: Artifacts by kind")
    by_kind = pd.DataFrame([
        {"kind": v["kind"], "id": v["id"], "version": v["version"]}
        for v in all_variants
    ])
    if not by_kind.empty:
        agg = by_kind.groupby(["kind", "id"]).agg(versions=("version", "count")).reset_index()
        agg = agg.sort_values(["kind", "id"])
        st.dataframe(agg, use_container_width=True, hide_index=True)
        st.caption(
            "Helix can improve multiple artifact types in parallel (prompts, skills, "
            "tool descriptions, rubrics, etc). Each row is one (kind, id) under search."
        )

    # ---------------- focus-scoped champion ----------------
    st.markdown("---")
    if focus:
        st.subheader(f":trophy: Champion for `{focus}`")
        variants = filter_variants_by_focus(all_variants, focus)
    else:
        st.subheader(":trophy: Overall champion")
        variants = all_variants

    measured = [v for v in variants if v["measurement"] and v["measurement"].get("score") is not None]
    if measured:
        champ = max(measured, key=lambda v: v["measurement"]["score"])
        m = champ["measurement"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Version", f"v{champ['version']}")
        c2.metric("Score", f"{m['score']:.3f}")
        c3.metric("Created by", champ["created_by"])
        c4.metric("Kind", champ["kind"])
        st.caption(
            f"Artifact id: `{champ['id']}`  •  content_hash: `{champ['content_hash'][:12]}`"
        )

        with st.expander(":scroll: Champion content", expanded=True):
            content = champ["content"]
            if isinstance(content, str):
                st.code(content, language="text")
            else:
                st.json(content)
    else:
        st.info("No measured artifacts in this focus.")

    # ---------------- by-method breakdown ----------------
    st.markdown("---")
    st.subheader(":hammer_and_wrench: Artifacts by Search method")
    df_source = filter_variants_by_focus(all_variants, focus) if focus else all_variants
    df = pd.DataFrame([
        {
            "id": v["id"],
            "kind": v["kind"],
            "version": v["version"],
            "created_by": v["created_by"],
            "search_method": v["search_method"],
            "score": v["measurement"]["score"] if v["measurement"] else None,
        }
        for v in df_source
    ])
    if not df.empty:
        by_method = df.groupby("created_by").agg(
            count=("version", "count"),
            mean_score=("score", "mean"),
            max_score=("score", "max"),
        ).reset_index()
        st.dataframe(by_method, use_container_width=True, hide_index=True)

    # ---------------- trajectory cache summary (informational) ----------------
    st.markdown("---")
    st.subheader(":broom: Trajectory cache")
    trajectories = list_cached_trajectories(archive_path)
    failed = get_failed_trajectories(archive_path)
    c1, c2, c3 = st.columns([1, 1, 2])
    c1.metric("Cached trajectories", len(trajectories))
    c2.metric("Failed", len(failed))
    with c3:
        if failed:
            st.caption(
                f"**{len(failed)}** failed cached trajectories. "
                "Go to the **Replay** panel to inspect and purge them."
            )
        else:
            st.caption("All cached trajectories appear to have answers. No cleanup needed.")

    # ---------------- failed-runs breakdown by error type ----------------
    if failed:
        st.markdown(":warning: **Failed runs by error type**")
        fdf = pd.DataFrame(failed)
        by_type = fdf["error_type"].value_counts().reset_index()
        by_type.columns = ["error_type", "count"]
        # Add a column suggesting causes
        causes = {
            "RateLimitError": "Hit provider rate limit. The new token-aware rate budget should prevent re-occurrence; for older entries, purge them.",
            "Timeout": "LLM call took longer than 120s. May indicate provider slowness or an unusually long context.",
            "JSONDecodeError": "Judge or proposer returned malformed JSON. Often transient; retry usually fixes.",
            "zero_step": "Agent run exited before any step was recorded. Most common cause: rate limit on the first call.",
            "failed_outcome": "Trajectory was marked failed without a specific error message.",
            "litellm.RateLimitError": "Hit provider rate limit. The new token-aware rate budget should prevent re-occurrence; for older entries, purge them.",
        }
        by_type["likely_cause"] = by_type["error_type"].map(lambda t: causes.get(t, "Unclassified. Click into the Replay panel to see specifics."))
        st.dataframe(by_type, use_container_width=True, hide_index=True)
        st.caption(
            f"Total failed: **{len(failed)}**. The Replay panel has a Purge button "
            "that drops these from the cache so the next round re-runs them."
        )

    # ---------------- recent rounds ----------------
    if rounds:
        st.markdown("---")
        st.subheader(":clock1: Recent rounds")
        round_df = pd.DataFrame(rounds)
        cols = [c for c in [
            "round", "candidate_version", "mean_score", "win_rate",
            "n_wins", "n_losses", "n_ties", "promoted",
        ] if c in round_df.columns]
        st.dataframe(round_df[cols].tail(10), use_container_width=True, hide_index=True)
