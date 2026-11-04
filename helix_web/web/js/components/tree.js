// The lineage tree: a tidy d3 forest with crossover edges and a gold crown.
// Renders in raw hex (not CSS variables) so exported SVG stands alone.

import { creatorHex, escapeHtml, fmtScore } from "../format.js";

const d3 = window.d3;

const GOLD = "#FFD700";
const SALMON = "#F39B7F";
const EDGE = "#BFAE8E";
const INK = "#2E2820";
const NODE_STROKE = "#3A2F1F";
const SELECT_STROKE = "#0F766E";
const SERIF = "'EB Garamond', 'Palatino Linotype', Georgia, serif";

// Failure-fingerprint mode: color nodes by what the checker said instead of
// who created them, so a stall shared across many nodes reads as one wall.
const FP_SOLVED = "solved: every check passes";
const FP_UNMEASURED = "(unmeasured)";
const FP_PALETTE = ["#E64B35", "#3C5488", "#7E6148", "#B8860B", "#F39B7F", "#4DBBD5", "#27324A"];

function fingerprintOf(v) {
  const m = v.measurement;
  if (!m || m.score === null || m.score === undefined) return FP_UNMEASURED;
  if (m.score >= 0.9999) return FP_SOLVED;
  const fb = (m.feedback || "").replace(/\s+/g, " ").trim();
  return fb || `fails at ${fmtScore(m.score)}`;
}

export function createTree(svgEl, opts = {}) {
  const nodeW = opts.nodeW ?? 150;
  const nodeH = opts.nodeH ?? 56;
  const rankSep = opts.rankSep ?? 110;
  const sibSep = opts.sibSep ?? 26;

  let orient = opts.orient ?? "vertical";
  let viewMode = opts.viewMode ?? "tree"; // "tree" | "generations"
  let laneMeta = [];
  let colorMode = opts.colorMode ?? "creator";
  let fpColors = new Map();
  let forest = [];
  let extraEdges = [];
  let collapsed = new Set();
  let selectedRef = null;
  let handlers = { select: () => {} };
  let laidOutNodes = [];

  const svg = d3.select(svgEl);
  svg.selectAll("*").remove();
  // A faint turbulence displacement gives every box the line quality of an
  // engraved plate; it travels with SVG export since it lives in the defs.
  const defs = svg.append("defs");
  defs
    .append("filter")
    .attr("id", "inkWobble")
    .attr("x", "-5%").attr("y", "-5%").attr("width", "110%").attr("height", "110%")
    .call((f) => {
      f.append("feTurbulence")
        .attr("type", "fractalNoise")
        .attr("baseFrequency", "0.028")
        .attr("numOctaves", "3")
        .attr("seed", "8")
        .attr("result", "n");
      f.append("feDisplacementMap")
        .attr("in", "SourceGraphic")
        .attr("in2", "n")
        .attr("scale", "2.4")
        .attr("xChannelSelector", "R")
        .attr("yChannelSelector", "G");
    });
  const zoomLayer = svg.append("g").attr("class", "zoom-layer");
  const gLanes = zoomLayer.append("g").attr("class", "lanes");
  const gCross = zoomLayer.append("g").attr("class", "crossover-links");
  const gLinks = zoomLayer.append("g").attr("class", "links");
  const gNodes = zoomLayer.append("g").attr("class", "nodes");

  const tooltip = document.createElement("div");
  Object.assign(tooltip.style, {
    position: "fixed",
    display: "none",
    background: "rgba(46,40,32,0.96)",
    color: "#F1E9D4",
    borderLeft: "2px solid #B08D2E",
    padding: "8px 10px",
    borderRadius: "2px",
    font: "11.5px/1.5 var(--font-mono, monospace)",
    pointerEvents: "none",
    zIndex: 40,
    whiteSpace: "pre",
  });
  document.body.appendChild(tooltip);

  const zoom = d3
    .zoom()
    .scaleExtent([0.15, 3])
    .on("zoom", (event) => zoomLayer.attr("transform", event.transform));
  svg.call(zoom).on("dblclick.zoom", null);

  function visibleChildren(node) {
    if (collapsed.has(node.ref)) return null;
    return node.children && node.children.length ? node.children : null;
  }

  function hiddenCount(node) {
    let n = 0;
    (function walk(x) {
      for (const c of x.children ?? []) {
        n += 1;
        walk(c);
      }
    })(node);
    return n;
  }

  function layout() {
    // Lay each root out independently, then pack the roots side by side so
    // separate genesis artifacts read as separate trees, not one mega-tree.
    const treeLayout = d3.tree().nodeSize([nodeW + sibSep, rankSep]);
    let offset = 0;
    laidOutNodes = [];
    laneMeta = [];
    const links = [];
    for (const root of forest) {
      const h = d3.hierarchy(root, visibleChildren);
      treeLayout(h);
      let minX = Infinity;
      let maxX = -Infinity;
      h.each((n) => {
        minX = Math.min(minX, n.x);
        maxX = Math.max(maxX, n.x);
      });
      h.each((n) => {
        n.x = n.x - minX + offset;
        n.y = n.y + nodeH; // headroom for the root row
      });
      offset += maxX - minX + nodeW + sibSep * 3;
      h.each((n) => laidOutNodes.push(n));
      links.push(...h.links());
    }
    return links;
  }

  // Lane index for the generations view: genesis in lane 0, a child bred in
  // round or iteration k in lane k+1.
  function laneOf(v) {
    const vm = v.variant_metadata || {};
    if (vm.round !== undefined) return vm.round + 1;
    if (vm.iter !== undefined) return vm.iter + 1;
    return 0;
  }

  function layoutGenerations() {
    // Time-true swimlanes: one lane per generation/iteration, the breeding
    // pool in the gutter, so selection and discards read straight off rows.
    // Vertical orientation stacks lanes as rows; rotating turns them into
    // columns with the gutter on top, so time flows left to right instead.
    const vert = orient === "vertical";
    const nodes = [];
    for (const root of forest) {
      (function walk(d, parent) {
        const n = { data: d, parent, x: 0, y: 0 };
        nodes.push(n);
        (d.children || []).forEach((c) => walk(c, n));
      })(root, null);
    }
    const lanes = new Map();
    for (const n of nodes) {
      const li = laneOf(n.data);
      if (!lanes.has(li)) lanes.set(li, []);
      lanes.get(li).push(n);
    }
    const laneIdxs = [...lanes.keys()].sort((a, b) => a - b);
    laneMeta = [];
    // Each branch owns a column: the first child continues the parent's
    // column, every later child of that parent branches to a fresh one.
    // A champion bred from an old candidate then reads as a long diagonal
    // back to its column, instead of a false straight chain.
    const laneH = 150;
    const laneW = nodeW + 70;
    const slotW = nodeW + 26;
    const slotH = nodeH + 30;
    const chipsPerRow = vert ? 5 : Math.floor((laneW - 24) / 40);
    // Rotated lanes need headroom above the first node for the tallest
    // chip gutter; the last lane's gutter can hold every earlier version.
    let gutterTop = 100;
    if (!vert) {
      const rows = Math.ceil(Math.max(1, nodes.length - 1) / chipsPerRow);
      gutterTop = 78 + rows * 23;
    }
    const colOf = new Map();
    const claimed = new Set();
    let nextFree = 0;
    laneIdxs.forEach((li, rowIdx) => {
      const row = lanes.get(li).sort((a, b) => a.data.version - b.data.version);
      const laneC = vert ? 70 + rowIdx * laneH : 40 + rowIdx * laneW + laneW / 2;
      row.forEach((n) => {
        let col;
        if (n.parent && !claimed.has(n.parent)) {
          col = colOf.get(n.parent);
          claimed.add(n.parent);
        } else {
          col = nextFree;
        }
        nextFree = Math.max(nextFree, col + 1);
        colOf.set(n, col);
        if (vert) {
          n.y = laneC;
          n.x = 330 + col * slotW;
        } else {
          n.x = laneC;
          n.y = gutterTop + 50 + col * slotH;
        }
      });
      const isIter = row[0]?.data.variant_metadata?.iter !== undefined;
      const vmAny = row
        .map((n) => n.data.variant_metadata || {})
        .find((m) => m.bred_from_pool);
      const used = new Set();
      row.forEach((n) => {
        if (n.parent) used.add(n.parent.data.version);
        if (n.data.crossover_parent) used.add(n.data.crossover_parent.version);
      });
      const pool = vmAny
        ? [...new Set(vmAny.bred_from_pool)].sort((a, b) => a - b)
        : null;
      let discarded = [];
      if (pool) {
        const bornBefore = nodes
          .filter((m) => laneOf(m.data) < li)
          .map((m) => m.data.version);
        discarded = [...new Set(bornBefore.filter((v) => !pool.includes(v)))].sort(
          (a, b) => a - b
        );
      }
      const modes = [
        ...new Set(row.map((n) => n.data.variant_metadata?.mode).filter(Boolean)),
      ];
      laneMeta.push({
        li,
        c: laneC,
        laneH,
        laneW,
        chipsPerRow,
        label: li === 0 ? "start" : isIter ? `iter ${li - 1}` : `gen ${li - 1}`,
        pool,
        used,
        discarded,
        modes,
      });
    });
    laidOutNodes = nodes;
    return nodes
      .filter((n) => n.parent)
      .map((n) => ({ source: n.parent, target: n }));
  }

  function pos(n) {
    if (viewMode === "generations") return [n.x, n.y];
    return orient === "vertical" ? [n.x, n.y] : [n.y * 1.6, n.x * 0.62];
  }

  function scoreOf(d) {
    const m = d?.data?.measurement;
    const s = m ? m.score : null;
    return s === null || s === undefined ? null : Number(s);
  }

  function deltaOf(d) {
    const own = scoreOf(d);
    const parent = scoreOf(d.parent);
    if (own === null || parent === null) return null;
    return own - parent;
  }

  function isLive(v) {
    return v.live_version !== null && v.live_version !== undefined
      && v.live_version === v.version;
  }

  function nodeTooltip(d) {
    const v = d.data;
    const m = v.measurement;
    const lines = [
      `id: ${v.id}`,
      `version: v${v.version}`,
      `kind: ${v.kind}`,
      `created_by: ${v.created_by}`,
      `search_method: ${v.search_method}`,
    ];
    if (m && m.score !== null && m.score !== undefined)
      lines.push(`score: ${fmtScore(m.score)}`);
    const delta = deltaOf(d);
    if (delta !== null)
      lines.push(`vs parent: ${delta >= 0 ? "+" : ""}${fmtScore(delta)}`);
    if (m && m.feedback)
      lines.push(`checker: ${m.feedback.split(";")[0].trim().slice(0, 60)}`);
    if ((v.n_measurements ?? 0) > 1)
      lines.push(`measured ${v.n_measurements} times (content re-proposed)`);
    if (m && m.preference && m.preference !== "none")
      lines.push(`preference: ${m.preference}`);
    if (m && m.recorded_at) lines.push(`recorded: ${m.recorded_at}`);
    const vm = v.variant_metadata || {};
    if (vm.round !== undefined) lines.push(`generation: ${vm.round}`);
    if (vm.iter !== undefined) lines.push(`iteration: ${vm.iter}`);
    if (vm.mode)
      lines.push(`sampled by ${vm.mode} from v${vm.sampled_parent_version}`);
    if (vm.bred_from_pool)
      lines.push(`archive at birth: ${vm.bred_from_pool.length} candidates`);
    if (v.is_champion) lines.push("CHAMPION of this artifact family");
    if (isLive(v)) lines.push("LIVE: this version serves production");
    return lines.join("\n");
  }

  function update({ animate = true } = {}) {
    const links = viewMode === "generations" ? layoutGenerations() : layout();
    const vert = orient === "vertical";
    const t = animate
      ? svg.transition().duration(360).ease(d3.easeCubicOut)
      : svg.transition().duration(0);

    const adjustSource = (l) => {
      const [x, y] = pos(l.source);
      return vert ? [x, y + nodeH / 2] : [x + nodeW / 2, y];
    };

    // A parent edge from a laid-out source node into a child's top edge; the
    // optional offset lets a crossover edge land beside the primary one so
    // two parents visibly converge on the same child.
    const parentEdgePath = (sourceNode, targetNode, offset = 0) => {
      const src = adjustSource({ source: sourceNode });
      const [tx, ty] = pos(targetNode);
      const tgt = vert
        ? [tx + offset, ty - nodeH / 2]
        : [tx - nodeW / 2, ty + offset];
      if (vert) {
        const my = (src[1] + tgt[1]) / 2;
        return `M${src[0]},${src[1]} C${src[0]},${my} ${tgt[0]},${my} ${tgt[0]},${tgt[1]}`;
      }
      const mx = (src[0] + tgt[0]) / 2;
      return `M${src[0]},${src[1]} C${mx},${src[1]} ${mx},${tgt[1]} ${tgt[0]},${tgt[1]}`;
    };

    // Selection emphasis: the ancestry path lights up, contemporaries stay as
    // context, and versions born after the selection fade to ghosts, so one
    // click shows both the lineage and the archive as it stood at breeding.
    const byRefData = new Map(laidOutNodes.map((n) => [n.data.ref, n.data]));
    const selData = selectedRef ? byRefData.get(selectedRef) : null;
    const ancestry = new Set();
    if (selData) {
      const stack = [selectedRef];
      while (stack.length) {
        const r = stack.pop();
        if (!r || ancestry.has(r)) continue;
        const v = byRefData.get(r);
        if (!v) continue;
        ancestry.add(r);
        if (v.parent) stack.push(`${v.parent.id}__v${v.parent.version}`);
        if (v.crossover_parent)
          stack.push(`${v.crossover_parent.id}__v${v.crossover_parent.version}`);
      }
    }
    const pool = new Set(
      (selData?.variant_metadata?.bred_from_pool ?? []).map(
        (ver) => `${selData.id}__v${ver}`
      )
    );
    const onPath = (a, b) => ancestry.has(a) && ancestry.has(b);
    const nodeState = (v) => {
      if (!selData) return "normal";
      if (ancestry.has(v.ref)) return "path";
      if (v.id === selData.id && v.version > selData.version) return "unborn";
      return "context";
    };
    const NODE_OPACITY = { normal: 1, path: 1, context: 0.45, unborn: 0.12 };

    // Swimlane scaffolding, generations mode only: alternating bands, a lane
    // label naming the sampler's move, and the breeding pool as chips with
    // picked (teal), available (plain), and truncation-discarded (struck).
    // Vertical: lanes are rows with the gutter on the left. Rotated: lanes
    // are columns with the gutter on top, so time reads left to right.
    const laneData = viewMode === "generations" ? laneMeta : [];
    const laneMaxX = laidOutNodes.length
      ? Math.max(...laidOutNodes.map((n) => n.x)) + nodeW
      : 0;
    const laneMaxY = laidOutNodes.length
      ? Math.max(...laidOutNodes.map((n) => n.y)) + nodeH
      : 0;
    const labelOrigin = (d) =>
      vert ? [-14, d.c - d.laneH / 2 + 30] : [d.c - d.laneW / 2 + 22, 26];
    const laneSel = gLanes.selectAll("g.lane").data(laneData, (d) => d.li);
    laneSel.exit().remove();
    const laneEnter = laneSel.enter().append("g").attr("class", "lane");
    laneEnter.append("rect").attr("class", "band");
    laneEnter.append("text").attr("class", "lab");
    laneEnter.append("text").attr("class", "sub");
    laneEnter.append("g").attr("class", "chips");
    const laneAll = laneEnter.merge(laneSel);
    laneAll
      .select("rect.band")
      .attr("x", (d) => (vert ? -30 : d.c - d.laneW / 2 + 8))
      .attr("y", (d) => (vert ? d.c - d.laneH / 2 + 8 : -10))
      .attr("width", (d) => (vert ? laneMaxX + 140 : d.laneW - 16))
      .attr("height", (d) => (vert ? d.laneH - 16 : laneMaxY + 60))
      .attr("fill", (d, i) => (i % 2 ? "#F2EBD3" : "#F8F3E3"))
      .attr("rx", 6);
    laneAll
      .select("text.lab")
      .attr("x", (d) => labelOrigin(d)[0])
      .attr("y", (d) => labelOrigin(d)[1])
      .attr("font-family", SERIF)
      .attr("font-size", 13)
      .attr("font-weight", 700)
      .attr("fill", INK)
      .text((d) =>
        d.label +
        (d.modes.length
          ? `  (${d.modes.join(", ")} pick${d.modes.length > 1 ? "s" : ""})`
          : "")
      );
    laneAll
      .select("text.sub")
      .attr("x", (d) => labelOrigin(d)[0])
      .attr("y", (d) => labelOrigin(d)[1] + 16)
      .attr("font-family", SERIF)
      .attr("font-size", 10)
      .attr("fill", "#97876A")
      .text((d) => (d.pool ? "pool for selection:" : d.li === 0 ? "genesis" : ""));
    laneAll.select("g.chips").each(function (d) {
      const g = d3.select(this);
      g.selectAll("*").remove();
      if (!d.pool) return;
      const chW = 34;
      const chH = 18;
      const perRow = d.chipsPerRow;
      const [ox, oy] = labelOrigin(d);
      const chips = d.pool
        .map((v) => ({ v, kind: d.used.has(v) ? "used" : "avail" }))
        .concat(d.discarded.map((v) => ({ v, kind: "out" })));
      chips.forEach((c, i) => {
        const cx = ox + (i % perRow) * (chW + 6);
        const cy = oy + 24 + Math.floor(i / perRow) * (chH + 5);
        g.append("rect")
          .attr("x", cx).attr("y", cy)
          .attr("width", chW).attr("height", chH).attr("rx", 9)
          .attr("fill", c.kind === "used" ? "#00A087" : c.kind === "avail" ? "#EDE5CC" : "#F8F3E3")
          .attr("stroke", c.kind === "used" ? "#00A087" : "#B7A67F")
          .attr("stroke-dasharray", c.kind === "out" ? "3 2" : null)
          .attr("opacity", c.kind === "out" ? 0.7 : 1);
        g.append("text")
          .attr("x", cx + chW / 2).attr("y", cy + 13)
          .attr("text-anchor", "middle")
          .attr("font-family", SERIF)
          .attr("font-size", 10.5).attr("font-weight", 600)
          .attr("fill", c.kind === "used" ? "#FFFFFF" : "#6B5F49")
          .text("v" + c.v);
        if (c.kind === "out")
          g.append("line")
            .attr("x1", cx + 3).attr("y1", cy + chH / 2)
            .attr("x2", cx + chW - 3).attr("y2", cy + chH / 2)
            .attr("stroke", "#E64B35").attr("stroke-width", 1.4);
      });
    });

    gLinks
      .selectAll("path.link")
      .data(links, (l) => l.target.data.ref)
      .join(
        (enter) =>
          enter
            .append("path")
            .attr("class", "link")
            .attr("fill", "none")
            .attr("stroke", EDGE)
            .attr("stroke-width", 1.6)
            .attr("opacity", 0)
            .call((sel) => sel.transition(t).attr("opacity", 1)),
        (update_) => update_,
        (exit) => exit.remove()
      )
      .transition(t)
      .attr("stroke", (l) => {
        if (selData && onPath(l.source.data.ref, l.target.data.ref)) return "#00A087";
        // An explore pick is the archive paying off: the sampler reached past
        // the current best. Make those breeding edges identifiable at rest.
        return l.target.data.variant_metadata?.mode === "explore" ? "#B8860B" : EDGE;
      })
      .attr("stroke-dasharray", (l) =>
        !(selData && onPath(l.source.data.ref, l.target.data.ref)) &&
        l.target.data.variant_metadata?.mode === "explore"
          ? "2 4"
          : null
      )
      .attr("stroke-width", (l) =>
        selData && onPath(l.source.data.ref, l.target.data.ref) ? 4 : 1.6
      )
      .attr("opacity", (l) =>
        !selData ? 1 : onPath(l.source.data.ref, l.target.data.ref) ? 1 : 0.2
      )
      .attr("d", (l) => parentEdgePath(l.source, l.target));

    // Crossover edges: the second parent converges on the child exactly like
    // the primary edge does, dashed and salmon, landing just beside it so the
    // merge reads as two parents mating into one offspring.
    const byRef = new Map(laidOutNodes.map((n) => [n.data.ref, n]));
    const crossPairs = extraEdges
      .map((e) => ({ from: byRef.get(e.from), to: byRef.get(e.to) }))
      .filter((e) => e.from && e.to);
    gCross
      .selectAll("path.crossover")
      .data(crossPairs, (e) => `${e.from.data.ref}->${e.to.data.ref}`)
      .join("path")
      .attr("class", "crossover")
      .attr("fill", "none")
      .attr("stroke-dasharray", "6 4")
      .transition(t)
      .attr("stroke", SALMON)
      .attr("stroke-width", (e) =>
        selData && onPath(e.from.data.ref, e.to.data.ref) ? 4 : 2.2
      )
      .attr("opacity", (e) =>
        !selData ? 1 : onPath(e.from.data.ref, e.to.data.ref) ? 1 : 0.2
      )
      .attr("d", (e) => parentEdgePath(e.from, e.to, 14));

    const nodeSel = gNodes
      .selectAll("g.node")
      .data(laidOutNodes, (n) => n.data.ref)
      .join(
        (enter) => {
          const g = enter.append("g").attr("class", "node").attr("opacity", 0);
          g.append("rect").attr("class", "box");
          g.append("rect").attr("class", "scorebar");
          g.append("text").attr("class", "line1");
          g.append("text").attr("class", "line2");
          g.append("text").attr("class", "line3");
          g.append("g").attr("class", "livetag");
          g.append("g").attr("class", "remeasure");
          g.append("g").attr("class", "badge");
          g.call((sel) => sel.transition(t).attr("opacity", 1));
          // New nodes pulse once so a live run reads at a glance.
          g.select("rect.box")
            .attr("stroke-width", 6)
            .transition()
            .duration(650)
            .attr("stroke-width", (n) => (n.data.is_champion ? 3 : 1.2));
          return g;
        },
        (update_) => update_,
        (exit) => exit.transition(t).attr("opacity", 0).remove()
      );

    nodeSel
      .style("cursor", "pointer")
      .on("click", (event, n) => {
        selectedRef = n.data.ref;
        update({ animate: false });
        handlers.select(n.data.ref, n.data);
      })
      .on("mousemove", (event, n) => {
        tooltip.textContent = nodeTooltip(n);
        tooltip.style.display = "block";
        tooltip.style.left = `${event.clientX + 14}px`;
        tooltip.style.top = `${event.clientY + 10}px`;
      })
      .on("mouseleave", () => (tooltip.style.display = "none"));

    nodeSel.transition(t)
      .attr("opacity", (n) => NODE_OPACITY[nodeState(n.data)])
      .attr("transform", (n) => {
        const [x, y] = pos(n);
        return `translate(${x},${y})`;
      });

    if (colorMode === "failure") {
      const fps = [...new Set(laidOutNodes.map((n) => fingerprintOf(n.data)))]
        .filter((f) => f !== FP_SOLVED && f !== FP_UNMEASURED)
        .sort();
      fpColors = new Map(fps.map((f, i) => [f, FP_PALETTE[i % FP_PALETTE.length]]));
      fpColors.set(FP_SOLVED, "#00A087");
      fpColors.set(FP_UNMEASURED, "#888888");
    }
    const nodeFill = (n) =>
      colorMode === "failure"
        ? fpColors.get(fingerprintOf(n.data))
        : creatorHex(n.data.created_by);

    nodeSel
      .select("rect.box")
      .attr("filter", "url(#inkWobble)")
      .attr("x", (n) => (-nodeW / 2) * (n.data.is_champion ? 1.12 : 1))
      .attr("y", (n) => (-nodeH / 2) * (n.data.is_champion ? 1.12 : 1))
      .attr("width", (n) => nodeW * (n.data.is_champion ? 1.12 : 1))
      .attr("height", (n) => nodeH * (n.data.is_champion ? 1.12 : 1))
      .attr("rx", 3)
      .attr("fill", nodeFill)
      .attr("stroke", (n) =>
        n.data.ref === selectedRef
          ? SELECT_STROKE
          : n.data.is_champion
            ? GOLD
            : pool.has(n.data.ref)
              ? "#3C5488"
              : NODE_STROKE
      )
      .attr("stroke-dasharray", (n) =>
        pool.has(n.data.ref) && n.data.ref !== selectedRef && !n.data.is_champion
          ? "6 3"
          : null
      )
      .attr("stroke-width", (n) =>
        n.data.ref === selectedRef
          ? 4
          : n.data.is_champion
            ? 3
            : pool.has(n.data.ref)
              ? 2.5
              : 1.2
      );

    const line = (cls, dy, weight, size) =>
      nodeSel
        .select(`text.${cls}`)
        .attr("text-anchor", "middle")
        .attr("y", dy)
        .attr("fill", "#FFFFFF")
        .attr("font-family", SERIF)
        .attr("font-size", size)
        .attr("font-weight", weight);

    // Text rows scale with the node so compact trees keep their clearances.
    const y1 = -nodeH * 0.143;
    const y2 = nodeH * 0.125;
    const y3 = nodeH * 0.34;
    line("line1", y1, (n) => (n.data.is_champion ? 700 : 600), 13).text((n) =>
      `${n.data.is_champion ? "★ " : ""}v${n.data.version}`
    );
    line("line2", y2, 400, 11).attr("font-style", "italic").text((n) => n.data.created_by);
    line("line3", y3, 500, 10.5).text((n) => {
      const s = scoreOf(n);
      if (s === null) return "";
      const delta = deltaOf(n);
      const deltaText =
        delta === null ? "" : ` ${delta >= 0 ? "+" : "-"}${fmtScore(Math.abs(delta))}`;
      return `score ${fmtScore(s)}${deltaText}`;
    });

    // Score gauge along the bottom edge: the climb as geometry, not numerals.
    nodeSel
      .select("rect.scorebar")
      .attr("x", -nodeW / 2 + 4)
      .attr("y", nodeH / 2 - 4.5)
      .attr("height", 2.5)
      .attr("rx", 1.25)
      .attr("fill", "#FFFFFF")
      .attr("opacity", 0.85)
      .attr("width", (n) => {
        const s = scoreOf(n);
        return s === null ? 0 : Math.max(0, Math.min(1, s)) * (nodeW - 8);
      });

    // Re-proposal badge: content-hash dedup folds identical proposals onto
    // one version, so a repeat count is the only trace the search circled back.
    nodeSel.select("g.remeasure").each(function (n) {
      const g = d3.select(this);
      g.selectAll("*").remove();
      const count = n.data.n_measurements ?? 0;
      if (count <= 1) return;
      const x = -nodeW / 2 + 2;
      const y = -nodeH / 2 - 8;
      g.append("rect")
        .attr("x", x).attr("y", y)
        .attr("width", 26).attr("height", 15).attr("rx", 7.5)
        .attr("fill", "#6B5F49").attr("stroke", "#FFFFFF").attr("stroke-width", 1.2);
      g.append("text")
        .attr("x", x + 13).attr("y", y + 11)
        .attr("text-anchor", "middle")
        .attr("font-family", SERIF)
        .attr("font-size", 8.5).attr("font-weight", 700).attr("fill", "#FFFFFF")
        .text(`x${count}`);
    });

    // The promote gate made visible: a LIVE tag on the version in production.
    nodeSel.select("g.livetag").each(function (n) {
      const tag = d3.select(this);
      tag.selectAll("*").remove();
      if (!isLive(n.data)) return;
      const w = 36;
      const x = nodeW / 2 - w + 6;
      const y = -nodeH / 2 - 8;
      tag
        .append("rect")
        .attr("x", x)
        .attr("y", y)
        .attr("width", w)
        .attr("height", 15)
        .attr("rx", 7.5)
        .attr("fill", "#00A087")
        .attr("stroke", "#FFFFFF")
        .attr("stroke-width", 1.4);
      tag
        .append("text")
        .attr("x", x + w / 2)
        .attr("y", y + 11)
        .attr("text-anchor", "middle")
        .attr("font-family", SERIF)
        .attr("font-size", 8.5)
        .attr("font-weight", 700)
        .attr("fill", "#FFFFFF")
        .text("LIVE");
    });

    // Collapse badge: bottom edge, only when the node has hidden children.
    nodeSel.select("g.badge").each(function (n) {
      const badge = d3.select(this);
      badge.selectAll("*").remove();
      if (viewMode === "generations") return; // no collapsing in lane mode
      const hasKids = (n.data.children ?? []).length > 0;
      if (!hasKids) return;
      const isCollapsed = collapsed.has(n.data.ref);
      const by = orient === "vertical" ? nodeH / 2 : 0;
      const bx = orient === "vertical" ? 0 : nodeW / 2;
      badge
        .append("circle")
        .attr("cx", bx)
        .attr("cy", by)
        .attr("r", 9)
        .attr("fill", isCollapsed ? "#6B5F49" : "#FFFFFF")
        .attr("stroke", "#6B5F49")
        .attr("stroke-width", 1.2)
        .style("cursor", "pointer")
        .on("click", (event) => {
          event.stopPropagation();
          toggle(n.data.ref);
        });
      badge
        .append("text")
        .attr("x", bx)
        .attr("y", by + 3.5)
        .attr("text-anchor", "middle")
        .attr("font-size", 9)
        .attr("font-weight", 700)
        .attr("fill", isCollapsed ? "#FFFFFF" : "#6B5F49")
        .style("pointer-events", "none")
        .text(isCollapsed ? `+${hiddenCount(n.data)}` : "−");
    });
  }

  function toggle(ref) {
    if (collapsed.has(ref)) collapsed.delete(ref);
    else collapsed.add(ref);
    update();
  }

  function countNodes(roots) {
    let n = 0;
    const walk = (x) => {
      n += 1;
      (x.children ?? []).forEach(walk);
    };
    roots.forEach(walk);
    return n;
  }

  function autoCollapse(roots) {
    // Big archives open collapsed below depth 3 so the shape reads first.
    const walk = (x, depth) => {
      if (depth >= 3 && (x.children ?? []).length) collapsed.add(x.ref);
      (x.children ?? []).forEach((c) => walk(c, depth + 1));
    };
    roots.forEach((r) => walk(r, 1));
  }

  function zoomToFit(pad = 40) {
    // A re-render can orphan this tree while its settle timers are still
    // pending; a detached svg cannot resolve its percentage size, so bail.
    if (!svgEl.isConnected) return;
    const bbox = zoomLayer.node().getBBox();
    const w = svgEl.clientWidth || svgEl.parentNode.clientWidth || 900;
    const h = svgEl.clientHeight || 600;
    if (!bbox.width || !bbox.height) return;
    const scale = Math.min(
      3,
      Math.max(0.15, Math.min((w - pad) / bbox.width, (h - pad) / bbox.height))
    );
    const tx = w / 2 - scale * (bbox.x + bbox.width / 2);
    const ty = h / 2 - scale * (bbox.y + bbox.height / 2);
    svg
      .transition()
      .duration(300)
      .call(zoom.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));
  }

  function exportSVG({ title = "", subtitle = "" } = {}) {
    const bbox = zoomLayer.node().getBBox();
    const pad = 36;
    const headroom = title ? 64 : 0;
    const clone = zoomLayer.node().cloneNode(true);
    clone.removeAttribute("transform");
    clone.setAttribute(
      "transform",
      `translate(${pad - bbox.x},${pad + headroom - bbox.y})`
    );
    const out = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    const width = Math.ceil(bbox.width + pad * 2);
    const height = Math.ceil(bbox.height + pad * 2 + headroom);
    out.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    out.setAttribute("width", width);
    out.setAttribute("height", height);
    out.setAttribute("viewBox", `0 0 ${width} ${height}`);
    const bg = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    bg.setAttribute("width", "100%");
    bg.setAttribute("height", "100%");
    bg.setAttribute("fill", "#FFFFFF");
    out.appendChild(bg);
    if (title) {
      const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
      t.setAttribute("x", pad);
      t.setAttribute("y", pad + 4);
      t.setAttribute("font-family", SERIF);
      t.setAttribute("font-size", "18");
      t.setAttribute("font-weight", "650");
      t.setAttribute("fill", INK);
      t.textContent = title;
      out.appendChild(t);
      if (subtitle) {
        const s = document.createElementNS("http://www.w3.org/2000/svg", "text");
        s.setAttribute("x", pad);
        s.setAttribute("y", pad + 26);
        s.setAttribute("font-family", SERIF);
        s.setAttribute("font-size", "12");
        s.setAttribute("fill", "#6B5F49");
        s.textContent = subtitle;
        out.appendChild(s);
      }
    }
    out.appendChild(clone);
    const blob = new Blob([new XMLSerializer().serializeToString(out)], {
      type: "image/svg+xml",
    });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${(title || "lineage").toLowerCase().replace(/[^a-z0-9]+/g, "_")}.svg`;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  const apiObj = {
    setForest(roots, { extraEdges: edges = [], preserveViewport = false, champions } = {}) {
      forest = roots;
      extraEdges = edges;
      if (!preserveViewport) {
        collapsed = new Set();
        if (countNodes(roots) > 200) autoCollapse(roots);
      }
      void champions;
      update({ animate: preserveViewport });
      // Fit right away for a sane first paint, then again once the entry
      // transition settles so the final bbox wins in a real browser.
      if (!preserveViewport) {
        requestAnimationFrame(() => zoomToFit());
        setTimeout(() => zoomToFit(), 420);
      }
      return apiObj;
    },
    update,
    on(event, fn) {
      handlers[event] = fn;
      return apiObj;
    },
    select(ref) {
      selectedRef = ref;
      update({ animate: false });
      return apiObj;
    },
    setOrientation(o) {
      orient = o;
      update();
      requestAnimationFrame(() => zoomToFit());
      return apiObj;
    },
    setColorMode(mode) {
      colorMode = mode;
      update({ animate: false });
      return apiObj;
    },
    setViewMode(mode) {
      viewMode = mode;
      collapsed.clear();
      update({ animate: false });
      requestAnimationFrame(() => zoomToFit());
      setTimeout(() => zoomToFit(), 420);
      return apiObj;
    },
    fingerprintLegend() {
      const present = new Set(laidOutNodes.map((n) => fingerprintOf(n.data)));
      return [...fpColors.entries()]
        .filter(([label]) => present.has(label))
        .map(([label, color]) => ({ label, color }));
    },
    expandAll() {
      collapsed.clear();
      update();
      return apiObj;
    },
    zoomToFit,
    exportSVG,
    destroy() {
      tooltip.remove();
    },
  };
  return apiObj;
}

export function legendHtml() {
  const items = [
    ["human (genesis)", "#3C5488"],
    ["spo*", "#00A087"],
    ["hillclimb*", "#4DBBD5"],
    ["gepa*", "#E64B35"],
    ["gepa_crossover", "#F39B7F"],
    ["dgm*", "#7E6148"],
  ];
  const chips = items
    .map(
      ([label, hex]) =>
        `<span class="chip" style="background:${hex}">${escapeHtml(label)}</span>`
    )
    .join(" ");
  return (
    chips +
    ` <span class="chip outline" style="border-color:#FFD700;color:#8a7400">★ champion</span>` +
    ` <span class="chip outline" style="border-color:#00A087;color:#00A087">LIVE = promoted</span>` +
    ` <span class="chip outline" style="border-style:dashed;border-color:#F39B7F;color:#a85f42">crossover edge</span>` +
    ` <span class="chip outline" style="border-style:dotted;border-color:#B8860B;color:#8a6508">explore pick</span>`
  );
}
