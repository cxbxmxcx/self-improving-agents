"""Lineage panel: interactive agraph tree with click-to-inspect.

Replaces the earlier Graphviz rendering with streamlit-agraph for native
click handling. Nodes are colored by which Search produced them, and the
current champion of each artifact family gets a thicker gold outline.

When a focus is set, the tree filters to only that artifact id + kind plus
its lineage; other artifact families in the same archive are hidden.

Clicking a node sets it as the selected version; the detail panel below
the graph updates without a full sidebar refresh.
"""

from __future__ import annotations

import streamlit as st
from streamlit_agraph import Config, Edge, Node, agraph

from helix.dashboard.data import (
    filter_variants_by_focus,
    get_champion_refs,
    get_lineage_tree,
    list_variants,
    parse_focus,
)


# Color palette per SPEC's "search-by-signal grid" — colors map to who
# produced the artifact. NPG-style palette from DESIGN_NOTES §7.
COLOR_BY_CREATOR = {
    "human": "#3C5488",            # slate blue
    "spo_round_0": "#00A087",      # ocean teal (SPO family)
    "gepa_mutation": "#E64B35",    # vermillion (GEPA family)
    "gepa_gen0_init": "#E64B35",
    "gepa_crossover": "#F39B7F",   # muted salmon (GEPA crossover)
    "hillclimb": "#4DBBD5",        # soft cyan
}

CHAMPION_BORDER_COLOR = "#FFD700"  # gold
CHAMPION_BORDER_WIDTH = 5


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


def _node_id(artifact_id: str, version: int) -> str:
    """Stable node id used by both nodes and edges."""
    return f"{artifact_id}__v{version}"


def _build_graph(
    tree: list[dict],
    champion_refs: set[tuple[str, int]],
) -> tuple[list[Node], list[Edge]]:
    """Walk the forest and produce agraph Node / Edge lists."""
    nodes: list[Node] = []
    edges: list[Edge] = []

    def walk(node: dict) -> None:
        nid = _node_id(node["id"], node["version"])
        score = node["measurement"]["score"] if node["measurement"] else None
        score_label = f"\nscore {score:.3f}" if score is not None else ""
        is_champion = (node["id"], node["version"]) in champion_refs

        crown = "★ " if is_champion else ""
        label = f"{crown}v{node['version']}\n{node['created_by']}{score_label}"

        # Tooltip on hover (agraph renders as the node's title attribute).
        tooltip_lines = [
            f"id: {node['id']}",
            f"version: {node['version']}",
            f"kind: {node['kind']}",
            f"created_by: {node['created_by']}",
            f"search_method: {node['search_method']}",
        ]
        if score is not None:
            tooltip_lines.append(f"score: {score:.3f}")
        if node["measurement"] and node["measurement"].get("preference"):
            tooltip_lines.append(f"preference: {node['measurement']['preference']}")
        if is_champion:
            tooltip_lines.append("CHAMPION of this artifact family")
        tooltip = "\n".join(tooltip_lines)

        fill = _creator_color(node["created_by"])

        # agraph's underlying vis.js needs `color` as a dict to color the
        # border separately from the fill. Passing borderColor/borderWidth
        # as top-level kwargs silently doesn't work.
        node_color = {
            "background": fill,
            "border": CHAMPION_BORDER_COLOR if is_champion else "#222222",
            "highlight": {
                "background": fill,
                "border": CHAMPION_BORDER_COLOR if is_champion else "#F8D210",
            },
        }

        nodes.append(
            Node(
                id=nid,
                label=label,
                title=tooltip,
                color=node_color,
                size=32 if is_champion else 24,
                shape="box",
                borderWidth=CHAMPION_BORDER_WIDTH if is_champion else 1,
                # font and shadow keywords map straight through to vis.js
                font={"color": "white", "size": 13, "face": "Helvetica", "bold": is_champion},
                shadow={"enabled": is_champion, "color": CHAMPION_BORDER_COLOR, "size": 18, "x": 0, "y": 0},
            )
        )

        for child in node["children"]:
            child_nid = _node_id(child["id"], child["version"])
            edges.append(Edge(source=nid, target=child_nid))
            walk(child)

    for root in tree:
        walk(root)

    return nodes, edges


def _filter_tree(tree: list[dict], focus_id: str, focus_kind: str) -> list[dict]:
    """Keep only subtrees whose root id+kind matches the focus."""
    out: list[dict] = []
    for node in tree:
        if node["id"] == focus_id and node["kind"] == focus_kind:
            out.append(node)
        else:
            for child in node["children"]:
                out.extend(_filter_tree([child], focus_id, focus_kind))
    return out


def render_lineage(archive_path: str, focus: str | None = None) -> None:
    tree = get_lineage_tree(archive_path)
    variants = list_variants(archive_path)
    champion_refs = get_champion_refs(archive_path)

    parsed_focus = parse_focus(focus)
    if parsed_focus is not None:
        fid, fkind = parsed_focus
        tree = _filter_tree(tree, fid, fkind)
        variants = filter_variants_by_focus(variants, focus)

    if not tree:
        st.info(
            "No artifacts in this focus." if focus else "No artifacts in archive yet."
        )
        return

    st.subheader("🌲 Lineage tree")
    if focus:
        st.caption(
            f"Focused on `{focus}`. Each node is one version of this artifact. "
            "Click a node to inspect it. Champion has a gold outline."
        )
    else:
        st.caption(
            "Each node is an artifact in the archive. Colors indicate which Search "
            "produced it. Champions per artifact family have a gold outline. "
            "Click a node to inspect it. Use the sidebar to focus on one artifact family."
        )

    # ---------------- legend ----------------
    legend_items = list(COLOR_BY_CREATOR.items()) + [("champion", CHAMPION_BORDER_COLOR)]
    cols = st.columns(len(legend_items))
    for i, (label, color) in enumerate(legend_items):
        is_champ = label == "champion"
        style = (
            f"background:transparent; color:{color}; padding:8px; "
            f"border-radius:6px; border:3px solid {color}; "
            f"text-align:center; font-size:0.85em; font-weight:600;"
            if is_champ
            else f"background:{color}; color:white; padding:8px; "
                 f"border-radius:6px; text-align:center; font-size:0.85em;"
        )
        cols[i].markdown(
            f'<div style="{style}">{label}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ---------------- the interactive graph ----------------
    nodes, edges = _build_graph(tree, champion_refs)

    # agraph's underlying vis.js renders to a fixed pixel box. We let the
    # user pick a size so wide monitors get the benefit; on smaller screens
    # the page scrolls horizontally rather than cramping the graph.
    sc1, sc2, sc3 = st.columns([1, 1, 4])
    with sc1:
        graph_width = st.slider(
            "Width (px)",
            min_value=800,
            max_value=2400,
            value=1600,
            step=100,
            help="Set wider on large monitors for clearer node spacing.",
        )
    with sc2:
        graph_height = st.slider(
            "Height (px)",
            min_value=400,
            max_value=1400,
            value=700,
            step=50,
        )
    with sc3:
        direction = st.radio(
            "Layout direction",
            ["Top → Down", "Left → Right"],
            horizontal=True,
            index=0,
            help="Top-Down is hierarchical-classic; Left-Right is better when lineages get deep.",
        )

    direction_code = "UD" if direction.startswith("Top") else "LR"

    # agraph's Config reads hierarchical-layout knobs as FLAT kwargs
    # (see streamlit_agraph/config.py); passing a nested
    # hierarchical_layout={...} dict is silently ignored. direction,
    # levelSeparation, nodeSpacing, treeSpacing, sortMethod must all be
    # top-level kwargs to Config().
    config = Config(
        width=graph_width,
        height=graph_height,
        directed=True,
        physics=False,
        hierarchical=True,
        direction=direction_code,
        sortMethod="directed",
        levelSeparation=150 if direction_code == "UD" else 220,
        nodeSpacing=160,
        treeSpacing=220,
        nodeHighlightBehavior=True,
        highlightColor="#F8D210",
        collapsible=False,
        node={"labelProperty": "label"},
        link={"labelProperty": "label", "renderLabel": False, "color": "#888"},
        interaction={"hover": True, "tooltipDelay": 200, "zoomView": True, "dragView": True},
    )

    clicked_id = agraph(nodes=nodes, edges=edges, config=config)

    # ---------------- selected-node detail panel ----------------
    st.markdown("---")
    st.subheader("🔍 Selected artifact")

    # Translate the clicked node id back to a version dict.
    selected = None
    if clicked_id:
        # Node id format: "<artifact_id>__v<version>"
        try:
            aid, vpart = clicked_id.rsplit("__v", 1)
            v_int = int(vpart)
            for v in variants:
                if v["id"] == aid and v["version"] == v_int:
                    selected = v
                    break
        except (ValueError, AttributeError):
            pass

    # Fallback: dropdown picker (used when nothing's clicked yet or when the
    # click was on an edge/empty area).
    if selected is None:
        if not variants:
            return

        # Default to a champion if one exists in the visible set.
        default_idx = 0
        visible_versions = sorted(variants, key=lambda v: v["version"])
        for i, v in enumerate(visible_versions):
            if (v["id"], v["version"]) in champion_refs:
                default_idx = i
                break

        labels = [
            f"v{v['version']} ({v['created_by']})"
            + (f" — {v['measurement']['score']:.3f}" if v["measurement"] and v["measurement"]["score"] is not None else "")
            for v in visible_versions
        ]
        selected_label = st.selectbox(
            "Click a node above, or pick from the list",
            labels,
            index=default_idx,
        )
        selected = visible_versions[labels.index(selected_label)]

    # ---------------- render the selected artifact's detail ----------------
    is_champion = (selected["id"], selected["version"]) in champion_refs
    if is_champion:
        st.success("👑 This is the current champion of its artifact family.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Version", f"v{selected['version']}")
    c2.metric("Created by", selected["created_by"])
    c3.metric(
        "Latest score",
        f"{selected['measurement']['score']:.3f}"
        if selected["measurement"] and selected["measurement"]["score"] is not None
        else "—",
    )
    c4.metric("Kind", selected["kind"])

    if selected["parent_version"]:
        st.caption(
            f"Parent: `v{selected['parent_version']}`  •  search method: `{selected['search_method']}`"
        )
    else:
        st.caption(f"Genesis artifact  •  search method: `{selected['search_method']}`")

    with st.expander("📜 Content", expanded=True):
        content = selected["content"]
        if isinstance(content, str):
            st.code(content, language="text")
        else:
            st.json(content)

    if selected["measurement"] and selected["measurement"].get("feedback"):
        with st.expander("💬 Latest judge feedback"):
            st.text(selected["measurement"]["feedback"])
