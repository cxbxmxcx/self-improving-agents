"""Compare page: pick two artifacts, see prompt diff and score comparison."""

from __future__ import annotations

import difflib
import html
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from helix.dashboard.data import (
    filter_variants_by_focus,
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


def render_compare(archive_path: str, focus: str | None = None) -> None:
    variants = list_variants(archive_path)
    if focus:
        variants = filter_variants_by_focus(variants, focus)
    if len(variants) < 2:
        st.info(
            "Need at least 2 artifacts in this focus to compare." if focus
            else "Need at least 2 artifacts in the archive to compare."
        )
        return

    st.subheader("⚖ Compare two artifacts")
    st.caption(
        f"Side-by-side diff of artifact content plus measurement comparison. "
        f"{'Focused on `' + focus + '`. ' if focus else ''}"
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
    st.markdown("### 📊 Latest measurements")
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
    st.markdown("### 📝 Content diff")
    a_content = a["content"] if isinstance(a["content"], str) else str(a["content"])
    b_content = b["content"] if isinstance(b["content"], str) else str(b["content"])
    _render_diff_side_by_side(a_content, b_content)

    # ---------------- side-by-side text ----------------
    st.markdown("### 📜 Side-by-side content")
    cl, cr = st.columns(2)
    with cl:
        st.caption(f"v{a['version']} ({a['created_by']})")
        st.code(a_content, language="text")
    with cr:
        st.caption(f"v{b['version']} ({b['created_by']})")
        st.code(b_content, language="text")

    # ---------------- measurement histories ----------------
    st.markdown("### 📈 Measurement history")
    a_hist = get_measurement_history(archive_path, a["id"], a["version"])
    b_hist = get_measurement_history(archive_path, b["id"], b["version"])

    if not a_hist and not b_hist:
        st.caption(
            "(neither artifact has a measurement history — they may be raw "
            "genesis artifacts that have not been judged yet)"
        )
        return

    # Build a plotly figure with two traces. Each measurement is plotted
    # against its own measurement index (1, 2, 3, ...) — sharing a timeline
    # across artifacts is misleading because the two artifacts were measured
    # in different rounds at different times. The plot shows each artifact's
    # OWN trajectory of scores over its OWN measurement sequence.
    fig = go.Figure()

    def _add_trace(history, label, color):
        if not history:
            return
        xs = list(range(1, len(history) + 1))
        ys = [m["score"] for m in history]
        tooltips = []
        for i, m in enumerate(history):
            pref = m.get("preference", "?")
            conf = m.get("confidence", 0)
            role = (m.get("metadata") or {}).get("role", "")
            tooltips.append(
                f"measurement #{i+1}<br>"
                f"score: {ys[i]:.3f}<br>"
                f"preference: {pref}<br>"
                f"confidence: {conf:.2f}<br>"
                f"role: {role}"
            )
        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode="lines+markers",
            name=label,
            line={"color": color, "width": 2},
            marker={"size": 10, "color": color},
            hovertext=tooltips,
            hoverinfo="text",
        ))

    _add_trace(a_hist, f"v{a['version']} ({a['created_by']})", "#3C5488")
    _add_trace(b_hist, f"v{b['version']} ({b['created_by']})", "#E64B35")

    fig.update_layout(
        height=350,
        xaxis_title="Measurement #",
        yaxis_title="Score",
        yaxis={"range": [-0.05, 1.05]},
        hovermode="closest",
        showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor": "right", "x": 1.0},
        margin={"l": 50, "r": 30, "t": 30, "b": 50},
    )
    fig.add_hline(y=0.5, line_dash="dot", line_color="#888", annotation_text="tie line")
    st.plotly_chart(fig, use_container_width=True)

    # ---------------- side-by-side history summary ----------------
    cl, cr = st.columns(2)
    with cl:
        st.caption(f"**v{a['version']}** ({a['created_by']}) — {len(a_hist)} measurements")
        if a_hist:
            adf = pd.DataFrame([
                {
                    "#": i + 1,
                    "score": m["score"],
                    "pref": m["preference"],
                    "confidence": m["confidence"],
                }
                for i, m in enumerate(a_hist)
            ])
            st.dataframe(adf, hide_index=True, use_container_width=True)
        else:
            st.caption("(no measurements)")
    with cr:
        st.caption(f"**v{b['version']}** ({b['created_by']}) — {len(b_hist)} measurements")
        if b_hist:
            bdf = pd.DataFrame([
                {
                    "#": i + 1,
                    "score": m["score"],
                    "pref": m["preference"],
                    "confidence": m["confidence"],
                }
                for i, m in enumerate(b_hist)
            ])
            st.dataframe(bdf, hide_index=True, use_container_width=True)
        else:
            st.caption("(no measurements)")
