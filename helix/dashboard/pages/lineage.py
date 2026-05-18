"""Lineage page: graphviz tree visualization + click-to-inspect."""

from __future__ import annotations

import streamlit as st

from helix.dashboard.data import get_artifact, get_lineage_tree, list_variants


# Color palette per SPEC's "search-by-signal grid" — colors map to who
# produced the artifact.
COLOR_BY_CREATOR = {
    "human": "#3C5488",          # slate blue
    "spo_round_0": "#00A087",    # ocean teal (SPO family)
    "gepa_mutation": "#E64B35",  # vermillion (GEPA family)
    "gepa_gen0_init": "#E64B35",
    "gepa_crossover": "#F39B7F", # muted salmon (GEPA crossover)
    "hillclimb": "#4DBBD5",      # soft cyan
}


def _creator_color(created_by: str) -> str:
    if created_by in COLOR_BY_CREATOR:
        return COLOR_BY_CREATOR[created_by]
    if created_by.startswith("spo"):
        return COLOR_BY_CREATOR["spo_round_0"]
    if created_by.startswith("gepa"):
        return COLOR_BY_CREATOR["gepa_mutation"]
    if created_by.startswith("hillclimb"):
        return COLOR_BY_CREATOR["hillclimb"]
    return "#888888"


def _build_graphviz(tree: list[dict]) -> str:
    """Return a graphviz DOT string for the lineage forest."""
    lines = [
        "digraph lineage {",
        '  rankdir="TB";',
        '  bgcolor="transparent";',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10];',
        '  edge [color="#888888", arrowsize=0.7];',
    ]

    def walk(node: dict) -> None:
        score = node["measurement"]["score"] if node["measurement"] else None
        score_label = f" score={score:.3f}" if score is not None else ""
        color = _creator_color(node["created_by"])
        node_id = f'{node["id"]}__v{node["version"]}'
        label = f'v{node["version"]}\\n{node["created_by"]}{score_label}'
        lines.append(
            f'  "{node_id}" [label="{label}", fillcolor="{color}", fontcolor="white"];'
        )
        for child in node["children"]:
            child_id = f'{child["id"]}__v{child["version"]}'
            lines.append(f'  "{node_id}" -> "{child_id}";')
            walk(child)

    for root in tree:
        walk(root)

    lines.append("}")
    return "\n".join(lines)


def render_lineage(archive_path: str) -> None:
    tree = get_lineage_tree(archive_path)
    variants = list_variants(archive_path)

    if not tree:
        st.info("No artifacts in archive yet.")
        return

    st.subheader(":evergreen_tree: Lineage tree")
    st.caption(
        "Each node is an artifact in the archive. Colors indicate which Search "
        "produced it. Edges point from parent to child."
    )

    # ---------------- legend ----------------
    cols = st.columns(len(COLOR_BY_CREATOR))
    for i, (creator, color) in enumerate(COLOR_BY_CREATOR.items()):
        cols[i].markdown(
            f'<div style="background:{color}; color:white; padding:8px; '
            f'border-radius:6px; text-align:center; font-size:0.85em;">{creator}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ---------------- the tree ----------------
    dot = _build_graphviz(tree)
    st.graphviz_chart(dot, use_container_width=True)

    # ---------------- inspect a node ----------------
    st.markdown("---")
    st.subheader(":mag: Inspect an artifact")
    versions = sorted({v["version"] for v in variants})
    if not versions:
        return

    selected_v = st.selectbox(
        "Pick a version to inspect",
        versions,
        format_func=lambda v: f"v{v}",
    )
    target = next((v for v in variants if v["version"] == selected_v), None)
    if target is None:
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Version", f"v{target['version']}")
    c2.metric("Created by", target["created_by"])
    c3.metric(
        "Latest score",
        f"{target['measurement']['score']:.3f}" if target["measurement"] and target["measurement"]["score"] is not None else "—",
    )

    if target["parent_version"]:
        st.caption(f"Parent: `v{target['parent_version']}`  •  search method: `{target['search_method']}`")
    else:
        st.caption(f"Genesis artifact  •  search method: `{target['search_method']}`")

    with st.expander(":scroll: Content", expanded=True):
        content = target["content"]
        if isinstance(content, str):
            st.code(content, language="text")
        else:
            st.json(content)

    if target["measurement"] and target["measurement"].get("feedback"):
        with st.expander(":speech_balloon: Latest judge feedback"):
            st.text(target["measurement"]["feedback"])
