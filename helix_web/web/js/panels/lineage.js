// Lineage panel: the tidy forest, a toolbar, and a click-to-inspect drawer.

import * as api from "../api.js";
import { createTree, legendHtml } from "../components/tree.js";
import { openDrawer, closeDrawer } from "../components/drawer.js";
import { escapeHtml, fmtScore, fmtWhen } from "../format.js";
import { onStreamEvent } from "../stream.js";
import { navigate, parseHash } from "../state.js";

let tree = null;
let unsubscribers = [];

// Expanded mode pairs the URL param with best-effort browser fullscreen.
// When Esc leaves fullscreen, drop the param too so the two stay in step.
let fsSyncWired = false;
export function wireFullscreenSync() {
  if (fsSyncWired) return;
  fsSyncWired = true;
  document.addEventListener("fullscreenchange", () => {
    if (!document.fullscreenElement && parseHash().params.get("expand") === "1") {
      navigate(null, { expand: null });
    }
  });
}

export function wireExpandButton(container, ctx, buttonId) {
  wireFullscreenSync();
  const expand = ctx.params.get("expand") === "1";
  document.body.classList.toggle("expanded", expand);
  const btn = container.querySelector(buttonId);
  btn.textContent = expand ? "Exit full screen" : "Full screen";
  btn.addEventListener("click", () => {
    // Request fullscreen before navigating: the click's user activation is
    // still live here, and the re-render then fits trees to the new size.
    if (!expand) document.documentElement.requestFullscreen?.().catch(() => {});
    else if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
    ctx.navigate(null, { expand: expand ? null : "1" });
  });
  return expand;
}

function focusParams(ctx) {
  const params = {};
  if (ctx.focus) {
    params.focus_id = ctx.focus.id;
    if (ctx.focus.kind) params.focus_kind = ctx.focus.kind;
  }
  return params;
}

async function showDetail(ctx, ref) {
  const [artifactId, version] = ref.split("__v");
  const v = await api.get(
    `/api/archives/${ctx.archiveKey}/variants/${encodeURIComponent(artifactId)}/${version}`
  );
  const el = document.createElement("div");
  const m = v.measurement;
  el.innerHTML = `
    <h2 class="panel-title" style="font-size:var(--fs-lg)">
      ${v.is_champion ? "👑 " : ""}${escapeHtml(v.id)} · v${v.version}</h2>
    ${
      v.is_champion
        ? `<div class="banner success">This is the current champion of its artifact family.</div>`
        : ""
    }
    <div class="tiles" style="margin-bottom:var(--sp-4)">
      <div class="tile"><div class="label">Version</div><div class="value">v${v.version}</div></div>
      <div class="tile"><div class="label">Created by</div><div class="value" style="font-size:var(--fs-md)">${escapeHtml(v.created_by)}</div></div>
      <div class="tile"><div class="label">Latest score</div><div class="value">${fmtScore(m?.score)}</div></div>
      <div class="tile"><div class="label">Search</div><div class="value" style="font-size:var(--fs-md)">${escapeHtml(v.search_method)}</div></div>
    </div>
    <p class="panel-caption">
      ${
        v.parent
          ? `Parent: v${v.parent.version}`
          : "Genesis artifact"
      }${
        v.crossover_parent
          ? ` · second parent (crossover): v${v.crossover_parent.version}`
          : ""
      } · created ${escapeHtml(fmtWhen(v.created_at))}
      ${v.live_version ? ` · live version: v${v.live_version}` : ""}
    </p>
    ${(() => {
      const vm = v.variant_metadata ?? {};
      const bits = [];
      if (vm.round !== undefined) bits.push(`generation ${vm.round}`);
      if (vm.iter !== undefined) bits.push(`iteration ${vm.iter}`);
      if (vm.mode) bits.push(`sampled by <b>${escapeHtml(vm.mode)}</b> from v${vm.sampled_parent_version}`);
      if (vm.bred_from_pool)
        bits.push(`the archive then held ${vm.bred_from_pool.length} candidate${vm.bred_from_pool.length === 1 ? "" : "s"}: ${vm.bred_from_pool.map((x) => "v" + x).join(", ")}`);
      return bits.length
        ? `<div class="banner info" style="margin-bottom:var(--sp-3)">Bred in ${bits.join(" · ")}</div>`
        : "";
    })()}
    <section class="block">
      <h3>Content${v.content_truncated ? " (truncated)" : ""}</h3>
      <pre class="content">${escapeHtml(
        typeof v.content === "string" ? v.content : JSON.stringify(v.content, null, 2)
      )}</pre>
    </section>
    ${
      m?.feedback
        ? `<details class="expander"><summary>Latest judge feedback</summary>
           <div class="body"><pre class="content">${escapeHtml(m.feedback)}</pre></div></details>`
        : ""
    }
    ${
      (v.measurement_history ?? []).length > 1
        ? `<section class="block" style="margin-top:var(--sp-4)">
             <h3>Measurement history (${v.measurement_history.length})</h3>
             <table class="data"><thead><tr><th>#</th><th class="num">score</th><th>pref</th><th class="num">confidence</th></tr></thead>
             <tbody>${v.measurement_history
               .map(
                 (h, i) =>
                   `<tr><td>${i + 1}</td><td class="num">${fmtScore(h.score)}</td>` +
                   `<td>${escapeHtml(h.preference)}</td><td class="num">${fmtScore(h.confidence, 2)}</td></tr>`
               )
               .join("")}</tbody></table>
           </section>`
        : ""
    }
  `;
  openDrawer(el, {
    onClose: () => ctx.navigate(null, { sel: null }),
  });
}

export async function render(container, ctx) {
  if (!ctx.archiveKey) {
    container.innerHTML = `<div class="placeholder">Pick an archive in the header to begin.</div>`;
    return;
  }
  const body = await api.get(
    `/api/archives/${ctx.archiveKey}/lineage`,
    focusParams(ctx),
    { signal: ctx.signal }
  );

  container.innerHTML = `
    <h2 class="panel-title">Lineage</h2>
    <p class="panel-caption">Every version stores one parent pointer; following
      those pointers draws this tree. ${body.stats.n_nodes} versions in
      ${body.stats.n_roots} famil${body.stats.n_roots === 1 ? "y" : "ies"}${
        body.stats.truncated ? " (truncated view)" : ""
      }. Click a node to light its lineage: ancestors stay bright, the breeding
      pool gets a dashed ring, and versions born later fade to ghosts.</p>
    <div style="display:flex; align-items:center; gap:var(--sp-2); margin-bottom:var(--sp-3); flex-wrap:wrap">
      <button class="btn" id="lg-viewmode">Generations view</button>
      <button class="btn" id="lg-fit">Fit</button>
      <button class="btn" id="lg-fullscreen">Full screen</button>
      <button class="btn" id="lg-orient">Rotate layout</button>
      <button class="btn" id="lg-expand">Expand all</button>
      <button class="btn" id="lg-colormode">Color by failure</button>
      <button class="btn primary" id="lg-export">Export SVG</button>
      <span style="margin-left:auto" id="lg-legend"></span>
    </div>
    <div class="card plate" style="padding:0; overflow:hidden">
      <svg id="lg-svg"></svg>
    </div>
  `;
  container.querySelector("#lg-legend").innerHTML = legendHtml();

  if (tree) tree.destroy();
  tree = createTree(container.querySelector("#lg-svg"));
  tree.setForest(body.roots, {
    extraEdges: body.extra_edges,
    champions: body.champions,
  });
  tree.on("select", (ref) => {
    ctx.navigate(null, { sel: ref });
    showDetail(ctx, ref).catch(() => {});
  });

  // The three lenses (view, orientation, color) live in the URL. Applying
  // them here touches only the tree; toolbar clicks navigate, and the
  // re-render lands back here with the new params. Mutating the tree and
  // then navigating would race the re-render against in-flight transitions.
  const viewM = ctx.params.get("view") === "generations" ? "generations" : "tree";
  const orient = ctx.params.get("orient") === "horizontal" ? "horizontal" : "vertical";
  const colorMode = ctx.params.get("colormode") === "failure" ? "failure" : "creator";

  // The generations view: one swimlane per round, with the breeding pool,
  // the picked parents, and truncation's discards in each lane's gutter.
  if (viewM === "generations") {
    tree.setViewMode("generations");
    container.querySelector("#lg-viewmode").textContent = "Tree view";
  }
  if (orient === "horizontal") tree.setOrientation("horizontal");
  // The second lens: same tree, colored by what the checker said. A stall
  // shared by many nodes turns monochrome, which is the point.
  if (colorMode === "failure") {
    tree.setColorMode("failure");
    container.querySelector("#lg-colormode").textContent = "Color by search";
    container.querySelector("#lg-legend").innerHTML = tree
      .fingerprintLegend()
      .map(
        (f) =>
          `<span class="chip" style="background:${f.color}">${escapeHtml(
            f.label.length > 42 ? f.label.slice(0, 41) + "…" : f.label
          )}</span>`
      )
      .join(" ");
  }

  container.querySelector("#lg-viewmode").addEventListener("click", () => {
    ctx.navigate(null, { view: viewM === "tree" ? "generations" : null });
  });
  container.querySelector("#lg-fit").addEventListener("click", () => tree.zoomToFit());
  wireExpandButton(container, ctx, "#lg-fullscreen");
  container.querySelector("#lg-orient").addEventListener("click", () => {
    ctx.navigate(null, { orient: orient === "vertical" ? "horizontal" : null });
  });
  container.querySelector("#lg-expand").addEventListener("click", () => tree.expandAll());
  container.querySelector("#lg-colormode").addEventListener("click", () => {
    ctx.navigate(null, { colormode: colorMode === "creator" ? "failure" : null });
  });
  container.querySelector("#lg-export").addEventListener("click", () => {
    const archive = ctx.archiveKey.split(".").pop();
    tree.exportSVG({
      title: `Lineage: ${archive}`,
      subtitle: "One parent pointer per version draws the tree; the crown marks the top score, not production.",
    });
  });

  const sel = ctx.params.get("sel");
  if (sel) {
    tree.select(sel);
    // drawer=0 keeps the inspector closed, for screenshots of the bare tree.
    if (ctx.params.get("drawer") !== "0") {
      showDetail(ctx, sel).catch(() => closeDrawer());
    }
  }

  // Live mode: when the stream reports growth, refetch and redraw in place.
  // The d3 join animates only the entering nodes, so a running loop reads
  // as the tree growing, with the viewport and collapse state untouched.
  for (const unsub of unsubscribers) unsub();
  let refreshTimer = null;
  const refresh = () => {
    clearTimeout(refreshTimer);
    refreshTimer = setTimeout(async () => {
      try {
        const fresh = await api.get(
          `/api/archives/${ctx.archiveKey}/lineage`,
          focusParams(ctx)
        );
        tree.setForest(fresh.roots, {
          extraEdges: fresh.extra_edges,
          champions: fresh.champions,
          preserveViewport: true,
        });
      } catch {
        /* transient; the next event retries */
      }
    }, 300);
  };
  unsubscribers = ["artifact", "measurement", "champions", "resync"].map((t) =>
    onStreamEvent(t, refresh)
  );
}
