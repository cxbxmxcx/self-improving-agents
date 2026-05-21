"""Trajectory replay page (heavy detail).

Lets the reader see exactly what the agent did on a given (artifact, question)
pair: every model call, tool call, tool result, hook firing, with expandable
detail. Optional side-by-side diff against another trajectory shows how the
same question was answered by a different prompt.
"""

from __future__ import annotations

import difflib
import json
from typing import Any

import streamlit as st

from helix.dashboard.data import get_trajectory, list_cached_trajectories, list_variants


def _render_step(step: dict, idx: int, expanded: bool = False) -> None:
    """Render one trajectory step as an expandable block."""
    kind = step.get("kind", "unknown")
    payload = step.get("payload", {})
    timestamp = step.get("timestamp", "")

    icon = {
        "model_call": "🤖",
        "tool_call": "🔧",
        "tool_result": "📥",
        "hook_fire": "⚡",
        "artifact_load": "📜",
        "human_input": "👤",
    }.get(kind, "❓")

    # Build a one-line summary
    summary = _summarize_step(kind, payload)

    with st.expander(f"{icon} **Step {idx}** — `{kind}` — {summary}", expanded=expanded):
        if kind == "model_call":
            response = payload.get("response", {})
            content = response.get("content", "")
            tool_calls = response.get("tool_calls") or []
            if content:
                st.markdown("**Assistant content:**")
                st.code(content[:2000], language="markdown")
                if len(content) > 2000:
                    st.caption(f"(truncated, full length {len(content)} chars)")
            if tool_calls:
                st.markdown("**Tool calls requested:**")
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    st.code(f"{fn.get('name', '?')}({fn.get('arguments', '?')})", language="text")
        elif kind == "tool_call":
            st.markdown("**Tool:**")
            st.code(payload.get("name", "?"), language="text")
            st.markdown("**Arguments:**")
            st.json(payload.get("arguments", {}))
        elif kind == "tool_result":
            result = payload.get("result")
            st.markdown("**Result:**")
            if isinstance(result, list) and result and isinstance(result[0], dict) and "source" in result[0]:
                # retrieval-shaped: render each hit
                for hit in result[:10]:
                    st.markdown(
                        f"• `{hit.get('source', '?')}:p{hit.get('page', '?')}` "
                        f"(score {hit.get('score', 0):.3f})"
                    )
                    st.caption(hit.get("text", "")[:300])
            elif isinstance(result, (dict, list)):
                st.json(result)
            else:
                st.code(str(result)[:2000], language="text")
        else:
            st.json(payload)
        st.caption(f"@ {timestamp}")


def _summarize_step(kind: str, payload: dict) -> str:
    if kind == "model_call":
        response = payload.get("response", {})
        tool_calls = response.get("tool_calls") or []
        if tool_calls:
            names = [tc.get("function", {}).get("name", "?") for tc in tool_calls]
            return f"calls: {', '.join(names)}"
        content = response.get("content", "")
        return f"answer ({len(content)} chars)" if content else "(no content)"
    elif kind == "tool_call":
        name = payload.get("name", "?")
        args = payload.get("arguments", {})
        arg_str = ", ".join(f"{k}={str(v)[:30]}" for k, v in (args or {}).items())
        return f"{name}({arg_str})"
    elif kind == "tool_result":
        result = payload.get("result")
        if isinstance(result, list) and result and isinstance(result[0], dict) and "source" in result[0]:
            sources = {h.get("source", "?") for h in result}
            return f"{len(result)} hits from {len(sources)} sources"
        return f"{type(result).__name__} returned"
    elif kind == "hook_fire":
        return payload.get("point", "")
    return ""


def _render_trajectory(traj_dict: dict) -> None:
    """Render a full trajectory dict."""
    st.markdown(f"**Task:** {traj_dict.get('task', '(no task)')}")
    outcome = traj_dict.get("outcome", "unknown")
    final = traj_dict.get("final_output", "")
    metadata = traj_dict.get("metadata", {}) or {}
    error_msg = metadata.get("error")
    steps = traj_dict.get("steps", [])

    c1, c2 = st.columns(2)
    c1.metric("Outcome", outcome)
    c2.metric("Steps", len(steps))

    # ---------------- failed trajectory: show the error prominently ----------------
    if error_msg or (outcome == "failed") or (outcome == "completed" and not steps and not final):
        st.error(
            f"⚠ This trajectory did not complete normally.\n\n"
            f"**Reason:** `{error_msg or 'unknown — completed with 0 steps and no output'}`\n\n"
            f"Common causes: provider rate limit, network timeout, malformed tool args, "
            f"or an unhandled exception in the agent loop. Errored trajectories from new "
            f"runs are no longer cached, but pre-fix archives may still contain them — "
            f"use the Purge button (Overview tab) to clear them."
        )
        if final and isinstance(final, str):
            with st.expander("Raw error payload"):
                st.code(final, language="text")
        return

    # ---------------- successful trajectory ----------------
    st.markdown("**Final output:**")
    if isinstance(final, str):
        if outcome == "completed":
            st.success(final[:3000] or "(empty output)")
        else:
            st.warning(final[:3000] or "(no output)")
    elif final is None:
        st.warning("(no output)")
    else:
        # Defensive: never let st.json() crash if the value isn't valid
        try:
            st.json(final)
        except Exception:
            st.code(str(final)[:3000], language="text")

    st.markdown("---")
    st.markdown("**Step trace:**")
    for i, step in enumerate(steps):
        _render_step(step, i, expanded=False)


def render_replay(archive_path: str, focus: str | None = None) -> None:
    from helix.dashboard.data import (
        count_failed_trajectories,
        get_failed_trajectories,
        parse_focus,
        purge_failed_trajectories,
    )
    trajectories = list_cached_trajectories(archive_path)
    variants = list_variants(archive_path)

    # ---------------- failed-runs banner + purge button at the top ----------------
    n_failed = count_failed_trajectories(archive_path)
    if n_failed > 0:
        st.error(
            f"⚠ **{n_failed} cached trajectories failed.** "
            "Most common cause: provider rate limits during a previous run. "
            "Purging removes them so the next round re-runs those questions "
            "with the rate-budget-aware LLM wrapper now in place."
        )
        c1, c2 = st.columns([1, 4])
        with c1:
            if st.button(
                f"🗑 Purge {n_failed} failed",
                type="primary",
                use_container_width=True,
                key="replay_purge_btn",
            ):
                n = purge_failed_trajectories(archive_path)
                st.success(f"Purged {n} failed trajectory rows. Refreshing...")
                st.rerun()
        with c2:
            with st.expander("🔍 Inspect failed trajectories"):
                failed = get_failed_trajectories(archive_path)
                if failed:
                    import pandas as pd
                    fdf = pd.DataFrame(failed)
                    # Group by error type
                    by_type = fdf["error_type"].value_counts().reset_index()
                    by_type.columns = ["error_type", "count"]
                    st.dataframe(by_type, use_container_width=True, hide_index=True)
                    st.caption(f"Top {min(20, len(fdf))} failed entries:")
                    st.dataframe(
                        fdf[["artifact_id", "version", "question_id", "error_type", "error_message"]].head(20),
                        use_container_width=True,
                        hide_index=True,
                    )
        st.markdown("---")

    if not trajectories:
        st.info("No cached trajectories yet. Run a chapter script that uses trajectory caching.")
        return

    # Filter trajectories by focus
    parsed = parse_focus(focus)
    if parsed is not None:
        fid, _kind = parsed
        trajectories = [t for t in trajectories if t["artifact_id"] == fid]
        if not trajectories:
            st.info("No cached trajectories for this focus.")
            return

    st.subheader("⏪ Trajectory replay")
    caption = (
        "See exactly what the agent did on a given question, with full "
        "step-by-step detail. Optional cross-trajectory diff shows how a "
        "different prompt changed agent behavior."
    )
    if focus:
        caption = f"Focused on `{focus}`. " + caption
    st.caption(caption)

    # If no focus, also let the user pick artifact id from the trajectories
    if parsed is None:
        artifact_ids = sorted({t["artifact_id"] for t in trajectories})
        chosen_id = st.selectbox(
            "Artifact id",
            artifact_ids,
            help="The trajectories are keyed by (artifact_id, version, question). Pick the artifact id first.",
            key="replay_artifact_id",
        )
        trajectories = [t for t in trajectories if t["artifact_id"] == chosen_id]
    else:
        chosen_id = parsed[0]

    # ---------------- picker for primary trajectory ----------------
    st.markdown("### Pick a trajectory")
    versions = sorted({t["version"] for t in trajectories})
    questions = sorted({t["question_id"] for t in trajectories})
    if not versions or not questions:
        st.info("No trajectories to display for the current selection.")
        return

    c1, c2 = st.columns(2)
    with c1:
        primary_version = st.selectbox("Version (which prompt produced this trajectory)", versions, key="primary_v")
    with c2:
        primary_question = st.selectbox("Question", questions, key="primary_q")

    primary = get_trajectory(archive_path, chosen_id, primary_version, primary_question)
    if primary is None:
        st.warning(f"No cached trajectory for {chosen_id} v{primary_version} on {primary_question}.")
        return

    # ---------------- toggle: compare with another ----------------
    compare_mode = st.checkbox("Compare with another trajectory", value=False)

    if compare_mode:
        st.markdown("### Pick the second trajectory")
        c1, c2 = st.columns(2)
        with c1:
            other_versions = [v for v in versions if v != primary_version]
            if not other_versions:
                st.info("Only one version with trajectories; nothing to compare against.")
                compare_mode = False
                other_version = None
            else:
                other_version = st.selectbox("Version", other_versions, key="other_v")
        with c2:
            other_question = st.selectbox("Question", [primary_question], key="other_q",
                                          help="Same question — comparing what different prompts produce.")

        if compare_mode and other_version is not None:
            other = get_trajectory(archive_path, chosen_id, other_version, other_question)

            st.markdown("---")
            cl, cr = st.columns(2)
            with cl:
                st.markdown(f"## v{primary_version} on {primary_question}")
                _render_trajectory(primary["trajectory"])
            with cr:
                st.markdown(f"## v{other_version} on {other_question}")
                if other:
                    _render_trajectory(other["trajectory"])
                else:
                    st.warning("No cached trajectory for that pairing.")

            # ---------------- answer diff ----------------
            if other:
                st.markdown("---")
                st.subheader("🔍 Answer-level diff")
                a_ans = primary["trajectory"].get("final_output") or ""
                b_ans = other["trajectory"].get("final_output") or ""
                differ = difflib.HtmlDiff(wrapcolumn=80)
                table = differ.make_table(
                    a_ans.splitlines() or [""],
                    b_ans.splitlines() or [""],
                    fromdesc=f"v{primary_version}",
                    todesc=f"v{other_version}",
                    context=False,
                )
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
        return

    # ---------------- single trajectory view ----------------
    st.markdown("---")
    _render_trajectory(primary["trajectory"])
