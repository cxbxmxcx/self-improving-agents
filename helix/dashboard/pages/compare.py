"""Compare page: pick two artifacts, see prompt diff and score comparison."""

from __future__ import annotations

import difflib
import html

import pandas as pd
import streamlit as st

from helix.dashboard.data import (
    get_measurement_history,
    list_variants,
)


def _render_diff_side_by_side(a_text: str, b_text: str) -> None:
    """Inline HTML diff highlighting word-level changes."""
    a_lines = a_text.splitlines() or [""]
    b_lines = b_text.splitlines() or [""]
    differ = difflib.HtmlDiff(wrapcolumn=80)
    table = differ.make_table(a_lines, b_lines, fromdesc="A", todesc="B", context=False)
    # difflib's HTML uses some quirky styling; wrap it for streamlit.
    styled = f"""
    <style>
      table.diff {{ font-family: monospace; font-size: 0.85em; border-collapse: collapse; width: 100%; }}
      .diff_header {{ background:#3C5488; color:white; padding:6px; }}
      td.diff_header {{ text-align: right; }}
      .diff_next {{ background:#f0f0f0; }}
      .diff_add {{ background:#a8e6a8; }}
      .diff_chg {{ background:#fff3a8; }}
      .diff_sub {{ background:#f5a8a8; }}
    </style>
    {table}
    """
    st.markdown(styled, unsafe_allow_html=True)


def render_compare(archive_path: str) -> None:
    variants = list_variants(archive_path)
    if len(variants) < 2:
        st.info("Need at least 2 artifacts in the archive to compare.")
        return

    st.subheader(":balance_scale: Compare two artifacts")
    st.caption(
        "Side-by-side diff of artifact content plus measurement comparison. "
        "Pick a baseline (A) and a candidate (B)."
    )

    options = {f"v{v['version']} ({v['created_by']})": v for v in variants}
    col1, col2 = st.columns(2)
    with col1:
        a_label = st.selectbox("Artifact A (baseline)", list(options.keys()), index=0)
    with col2:
        # Default to the highest-scoring one for B
        scored = [v for v in variants if v["measurement"]]
        if scored:
            best = max(scored, key=lambda v: v["measurement"]["score"] or 0)
            best_label = f"v{best['version']} ({best['created_by']})"
            try:
                default_idx = list(options.keys()).index(best_label)
            except ValueError:
                default_idx = min(1, len(options) - 1)
        else:
            default_idx = min(1, len(options) - 1)
        b_label = st.selectbox("Artifact B (candidate)", list(options.keys()), index=default_idx)

    a = options[a_label]
    b = options[b_label]

    if a["version"] == b["version"]:
        st.warning("Pick two different artifacts.")
        return

    # ---------------- score comparison ----------------
    st.markdown("### :bar_chart: Latest measurements")
    c1, c2, c3 = st.columns(3)
    a_m = a["measurement"]
    b_m = b["measurement"]
    a_score = a_m["score"] if a_m and a_m["score"] is not None else None
    b_score = b_m["score"] if b_m and b_m["score"] is not None else None
    c1.metric(f"v{a['version']} score", f"{a_score:.3f}" if a_score is not None else "—")
    c2.metric(f"v{b['version']} score", f"{b_score:.3f}" if b_score is not None else "—",
              delta=f"{b_score - a_score:.3f}" if (a_score is not None and b_score is not None) else None)
    winner = (
        "v" + str(b["version"]) if (b_score or 0) > (a_score or 0)
        else "v" + str(a["version"]) if (a_score or 0) > (b_score or 0)
        else "tie"
    )
    c3.metric("Winner by score", winner)

    # ---------------- content diff ----------------
    st.markdown("### :memo: Content diff")
    a_content = a["content"] if isinstance(a["content"], str) else str(a["content"])
    b_content = b["content"] if isinstance(b["content"], str) else str(b["content"])
    _render_diff_side_by_side(a_content, b_content)

    # ---------------- side-by-side text ----------------
    st.markdown("### :scroll: Side-by-side content")
    cl, cr = st.columns(2)
    with cl:
        st.caption(f"v{a['version']} ({a['created_by']})")
        st.code(a_content, language="text")
    with cr:
        st.caption(f"v{b['version']} ({b['created_by']})")
        st.code(b_content, language="text")

    # ---------------- measurement histories ----------------
    st.markdown("### :chart_with_upwards_trend: Measurement history")
    a_hist = get_measurement_history(archive_path, a["id"], a["version"])
    b_hist = get_measurement_history(archive_path, b["id"], b["version"])
    rows = []
    for i, m in enumerate(a_hist):
        rows.append({"artifact": f"v{a['version']}", "round_idx": i, "score": m["score"], "preference": m["preference"]})
    for i, m in enumerate(b_hist):
        rows.append({"artifact": f"v{b['version']}", "round_idx": i, "score": m["score"], "preference": m["preference"]})
    if rows:
        df = pd.DataFrame(rows)
        st.line_chart(df, x="round_idx", y="score", color="artifact", height=300)
    else:
        st.caption("(no measurement history for these artifacts)")
