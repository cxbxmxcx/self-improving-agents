"""Champion panel: what the platform actually produced that's worth using.

For each artifact family (id + kind), shows the current champion with:
  - Full content
  - Lineage path from genesis to champion
  - Toggle between champion-vs-parent and champion-vs-genesis diff
  - Top wins (questions where the champion beat its predecessor with
    highest judge confidence)
  - Provenance summary (Search method, generation, total artifacts produced
    to reach this state)

When focus is "all artifacts", renders one section per family. When focus is
set, renders only that family. The other panels (Lineage, Compare, etc.) are
exploration tools; this panel is the headline result.
"""

from __future__ import annotations

import difflib

import streamlit as st

from helix.dashboard.data import (
    get_champions_per_family,
    get_live_champion,
    get_measurement_history,
    get_promotion_history,
    list_variants,
    parse_focus,
    promote_artifact,
)


def _do_promote(archive_path: str, artifact_id: str, version: int, score) -> None:
    """Call promote_artifact with a dashboard-appropriate approver tag.

    The dashboard doesn't carry a user id the way the chat UI does, so we
    record the approver as 'dashboard' and put the score in the reason so
    the audit trail still says why this version was chosen.
    """
    score_text = f" (score {score:.3f})" if score is not None else ""
    promote_artifact(
        archive_path=archive_path,
        artifact_id=artifact_id,
        version=version,
        approver="dashboard",
        reason=f"manual promotion via dashboard{score_text}",
    )


def _lineage_path(variants: list[dict], champion: dict) -> list[dict]:
    """Walk from champion back to genesis. Returns [genesis, ..., champion]."""
    by_ref = {(v["id"], v["version"]): v for v in variants}
    chain: list[dict] = [champion]
    current = champion
    while current.get("parent_version") is not None:
        parent_key = (current["parent_id"] or current["id"], current["parent_version"])
        parent = by_ref.get(parent_key)
        if parent is None or parent == current:
            break
        chain.append(parent)
        current = parent
    chain.reverse()
    return chain


def _render_diff(a_text: str, b_text: str, a_label: str, b_label: str) -> None:
    """Side-by-side HTML diff with color-coded changes."""
    a_lines = a_text.splitlines() or [""]
    b_lines = b_text.splitlines() or [""]
    differ = difflib.HtmlDiff(wrapcolumn=80)
    table = differ.make_table(a_lines, b_lines, fromdesc=a_label, todesc=b_label, context=False)
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


def _render_one_champion(
    archive_path: str,
    family_key: str,
    champion: dict,
    variants: list[dict],
) -> None:
    """Render the full champion section for one artifact family."""
    measurement = champion["measurement"] or {}
    score = measurement.get("score")

    # ---------------- header ----------------
    st.markdown(f"### 👑 Champion: `{family_key}`")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best candidate", f"v{champion['version']}")
    c2.metric("Score", f"{score:.3f}" if score is not None else "—")
    c3.metric("Created by", champion["created_by"])
    c4.metric("Search method", champion["search_method"])
    st.caption(
        f"Artifact id: `{champion['id']}`  •  kind: `{champion['kind']}`  "
        f"•  content_hash: `{champion['content_hash'][:12]}`"
    )

    # ---------------- live champion vs best candidate ----------------
    # The chapter draws the line between candidate ranking and deployment.
    # 'Best candidate' (above) is what the archive ranks highest by score.
    # 'Live champion' (here) is what the running agent actually serves;
    # it changes only when something calls archive.promote(). When the two
    # differ, an offline improver has produced a candidate that beats the
    # current live version but no one has promoted it yet.
    live = get_live_champion(archive_path, champion["id"])
    live_version = live["version"] if live else None
    best_version = champion["version"]

    if live_version == best_version:
        st.success(
            f"✅ Live champion **v{best_version}** is also the best candidate. "
            f"Nothing pending."
        )
    elif live is None:
        st.warning(
            f"⚠️ No promotion recorded for `{champion['id']}` yet. "
            f"The running agent serves the genesis prompt; best candidate "
            f"**v{best_version}** ({score:.3f}) is waiting for a promotion."
            if score is not None else
            f"⚠️ No promotion recorded for `{champion['id']}` yet."
        )
        if st.button(
            f"👑 Promote v{best_version} → live",
            key=f"promote_genesis_{family_key}_{best_version}",
            type="primary",
        ):
            _do_promote(archive_path, champion["id"], best_version, score)
            st.rerun()
    else:
        st.warning(
            f"⚠️ Live champion is **v{live_version}** but best candidate is "
            f"**v{best_version}** ({score:.3f}). An offline improver has "
            f"produced a winner pending review."
            if score is not None else
            f"⚠️ Live champion is **v{live_version}**, best candidate is "
            f"**v{best_version}** (no score)."
        )
        if st.button(
            f"👑 Promote v{best_version} → live",
            key=f"promote_{family_key}_{best_version}",
            type="primary",
        ):
            _do_promote(archive_path, champion["id"], best_version, score)
            st.rerun()

    # ---------------- lineage path ----------------
    path = _lineage_path(variants, champion)
    st.markdown("**Lineage from genesis to champion:**")
    path_pieces = []
    for v in path:
        score_str = (
            f" ({v['measurement']['score']:.3f})"
            if v["measurement"] and v["measurement"].get("score") is not None
            else ""
        )
        path_pieces.append(f"`v{v['version']}` *{v['created_by']}*{score_str}")
    st.markdown(" → ".join(path_pieces))

    # ---------------- content ----------------
    with st.expander("📜 Full champion content", expanded=True):
        content = champion["content"]
        if isinstance(content, str):
            st.code(content, language="text")
        else:
            st.json(content)

    # ---------------- diff toggle ----------------
    st.markdown("---")
    st.markdown("**Compare champion against:**")
    diff_mode = st.radio(
        f"Diff mode for `{family_key}`",
        ["Genesis (v1 → champion)", "Parent (immediate predecessor)"],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
        key=f"champion_diff_{family_key}",
    )

    genesis = path[0] if path else None
    parent = next(
        (v for v in variants
         if v["id"] == (champion.get("parent_id") or champion["id"])
         and v["version"] == champion.get("parent_version")),
        None,
    )

    if diff_mode.startswith("Genesis") and genesis and genesis is not champion:
        gen_content = genesis["content"] if isinstance(genesis["content"], str) else str(genesis["content"])
        champ_content = champion["content"] if isinstance(champion["content"], str) else str(champion["content"])
        _render_diff(gen_content, champ_content,
                     f"v{genesis['version']} (genesis)",
                     f"v{champion['version']} (champion)")
    elif diff_mode.startswith("Parent") and parent is not None:
        p_content = parent["content"] if isinstance(parent["content"], str) else str(parent["content"])
        champ_content = champion["content"] if isinstance(champion["content"], str) else str(champion["content"])
        _render_diff(p_content, champ_content,
                     f"v{parent['version']} (parent)",
                     f"v{champion['version']} (champion)")
    elif champion["parent_version"] is None:
        st.info("This champion is itself the genesis artifact — nothing earlier to compare against.")
    else:
        st.info("Diff target not available in archive.")

    # ---------------- top wins ----------------
    st.markdown("---")
    st.markdown("**Top wins** (questions where the champion's verdict was strongest):")
    per_q = measurement.get("metadata", {}).get("per_question", [])
    if per_q:
        # Sort by confidence among candidate wins
        wins = sorted(
            [q for q in per_q if q.get("preference") == "left"],
            key=lambda q: q.get("confidence", 0),
            reverse=True,
        )[:5]
        if wins:
            for q in wins:
                with st.expander(
                    f"🏆 {q.get('question_id', '?')} (band {q.get('band', '?')})"
                    f" — confidence {q.get('confidence', 0):.2f}"
                ):
                    fb = q.get("feedback", "")
                    if fb:
                        st.text(fb)
                    else:
                        st.caption("(no judge feedback recorded for this verdict)")
        else:
            st.caption(
                "Champion's latest measurement has no candidate-wins recorded — it may have "
                "won by aggregate score with mostly ties, or the metadata format from this "
                "round didn't include per-question detail."
            )
    else:
        st.caption("No per-question verdicts attached to the champion's latest measurement.")

    # ---------------- provenance + cost ----------------
    st.markdown("---")
    st.markdown("**Provenance:**")
    history = get_measurement_history(archive_path, champion["id"], champion["version"])
    n_measurements = len(history)

    # Count siblings under the same id+kind
    family_versions = [v for v in variants if v["id"] == champion["id"] and v["kind"] == champion["kind"]]

    # Estimate cumulative cost: sum cost_tokens from every measurement of every
    # artifact in this family.
    total_tokens = 0
    for v in family_versions:
        if v["measurement"]:
            total_tokens += v["measurement"].get("cost_tokens", 0) or 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Family size", f"{len(family_versions)} versions")
    c2.metric("Champion measurements", n_measurements)
    c3.metric("Family tokens", f"{total_tokens:,}")
    c4.metric("Champion confidence", f"{measurement.get('confidence', 0):.2f}")

    # ---------------- promotion history ----------------
    promo_history = get_promotion_history(archive_path, champion["id"])
    if promo_history:
        st.markdown("---")
        with st.expander(f"📜 Promotion history ({len(promo_history)} promotion(s))"):
            for p in promo_history:
                st.markdown(
                    f"- **v{p['version']}** by `{p['approver']}` "
                    f"at *{p['promoted_at']}* — {p['reason']}"
                )


def render_champion(archive_path: str, focus: str | None = None) -> None:
    variants = list_variants(archive_path)
    if not variants:
        st.info("Archive is empty. Run a chapter script to populate it.")
        return

    champions = get_champions_per_family(archive_path)
    if not champions:
        st.info("No measured artifacts yet — no champion to declare. Run an improvement round.")
        return

    st.subheader("👑 Champion Summary")
    st.caption(
        "What the platform actually produced for you. Each artifact family "
        "(unique id + kind) has one current champion: the highest-scoring "
        "version recorded. This is the headline result."
    )

    parsed_focus = parse_focus(focus)

    if parsed_focus is not None:
        # Single-family view
        key = focus
        champ = champions.get(key)
        if champ is None:
            st.warning(f"No champion exists yet for `{key}`. The artifact may have no measurements.")
            return
        _render_one_champion(archive_path, key, champ, variants)
    else:
        # Multi-family view: one section per family
        st.markdown(
            f"**{len(champions)} artifact families** have champions in this archive."
        )
        for i, (key, champ) in enumerate(sorted(champions.items())):
            if i > 0:
                st.markdown("---")
                st.markdown("---")  # double divider between families
            _render_one_champion(archive_path, key, champ, variants)
