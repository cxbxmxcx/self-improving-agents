// The showdown: lineage trees from several archives, side by side.
// This is the screen §3.4.3's two-trees figure gets shot from.

import * as api from "../api.js";
import { createTree, legendHtml } from "../components/tree.js";
import { cache } from "../state.js";
import { escapeHtml, fmtScore } from "../format.js";
import { wireExpandButton } from "./lineage.js";

let trees = [];

// Walk a forest for the headline facts the tiny nodes cannot carry:
// the best score (and who holds it), the live version, and measured count.
function forestFacts(roots) {
  let best = null;
  let live = null;
  let measured = 0;
  const stack = [...roots];
  while (stack.length) {
    const node = stack.pop();
    const score = node.measurement?.score;
    if (score !== null && score !== undefined) {
      measured += 1;
      if (!best || score > best.score) best = { score, version: node.version };
    }
    if (node.live_version === node.version) live = node.version;
    stack.push(...(node.children ?? []));
  }
  return { best, live, measured };
}

export async function render(container, ctx) {
  const chosen = (ctx.params.get("keys") ?? "")
    .split(",")
    .map((k) => k.trim())
    .filter(Boolean);

  const available = cache.archives.filter((a) => a.valid);
  const checkboxes = available
    .map((a) => {
      const checked = chosen.includes(a.key) ? "checked" : "";
      return `<label style="display:inline-flex;align-items:center;gap:4px;margin-right:var(--sp-3);font-size:var(--fs-sm)">
        <input type="checkbox" value="${a.key}" ${checked}> ${escapeHtml(a.rel_path ?? a.label)}
      </label>`;
    })
    .join("");

  container.innerHTML = `
    <h2 class="panel-title">Lineage compare</h2>
    <p class="panel-caption">The same task, different searches: each archive's
      family tree drawn to the same scale. Pick two or three runs and the
      shapes argue for themselves.</p>
    <div class="card" style="margin-bottom:var(--sp-4)">
      <div id="lc-picks">${checkboxes}</div>
    </div>
    <div style="display:flex;align-items:center;gap:var(--sp-2);margin-bottom:var(--sp-3);flex-wrap:wrap">
      <button class="btn" id="lc-viewmode" ${chosen.length ? "" : "disabled"}>Generations view</button>
      <button class="btn" id="lc-fit" ${chosen.length ? "" : "disabled"}>Fit</button>
      <button class="btn" id="lc-fullscreen" ${chosen.length ? "" : "disabled"}>Full screen</button>
      <button class="btn" id="lc-orient" ${chosen.length ? "" : "disabled"}>Rotate layout</button>
      <button class="btn" id="lc-colormode" ${chosen.length ? "" : "disabled"}>Color by failure</button>
      <button class="btn primary" id="lc-export" ${chosen.length ? "" : "disabled"}>Export all SVG</button>
      <span style="margin-left:auto" id="lc-legend">${legendHtml()}</span>
    </div>
    <div id="lc-grid" style="display:grid;grid-template-columns:repeat(${Math.max(chosen.length, 1)},1fr);gap:var(--sp-4)"></div>
  `;

  container.querySelectorAll("#lc-picks input").forEach((box) => {
    box.addEventListener("change", () => {
      const keys = [...container.querySelectorAll("#lc-picks input:checked")].map(
        (b) => b.value
      );
      ctx.navigate(null, { keys: keys.join(",") || null });
    });
  });

  const grid = container.querySelector("#lc-grid");
  if (!chosen.length) {
    grid.innerHTML = `<div class="placeholder">Select archives above to compare their lineage trees.</div>`;
    return;
  }

  const body = await api.get(
    "/api/lineage-compare",
    { keys: chosen.join(",") },
    { signal: ctx.signal }
  );

  for (const t of trees) t.destroy();
  trees = [];
  for (const arch of body.archives) {
    const cell = document.createElement("div");
    const stats = arch.forest.stats;
    const facts = forestFacts(arch.forest.roots);
    const bestHtml = facts.best
      ? `<span class="chip" style="background:var(--npg-teal)">best ${fmtScore(facts.best.score)}</span>
         <span style="color:var(--fg-muted);font-size:var(--fs-xs)">★ v${facts.best.version}${
           facts.live ? ` · LIVE v${facts.live}` : ""
         } · ${facts.measured}/${stats.n_nodes} measured</span>`
      : `<span class="chip outline" style="border-color:var(--warn);color:var(--warn)">no scores recorded</span>`;
    cell.innerHTML = `
      <div class="card plate" style="padding:0;overflow:hidden">
        <div style="display:flex;align-items:center;gap:var(--sp-2);flex-wrap:wrap;padding:var(--sp-2) var(--sp-3);border-bottom:1px solid var(--border)">
          <b>${escapeHtml(arch.label)}</b>
          <span style="color:var(--fg-faint);font-size:var(--fs-xs)">
            ${arch.chapter ? escapeHtml(arch.chapter) + " · " : ""}${stats.n_nodes} versions · depth ${stats.max_depth}</span>
          <span style="margin-left:auto;display:flex;align-items:center;gap:var(--sp-2)">${bestHtml}</span>
        </div>
        <svg class="lc-svg"></svg>
      </div>`;
    grid.appendChild(cell);
    const tree = createTree(cell.querySelector("svg"), { nodeW: 120, nodeH: 50, sibSep: 16, rankSep: 90 });
    tree.setForest(arch.forest.roots, {
      extraEdges: arch.forest.extra_edges,
      champions: arch.forest.champions,
    });
    trees.push(tree);
  }

  // One toolbar drives every tree at once, so the lanes line up and the
  // side-by-side comparison stays honest: same encoding, different search.
  // The lenses live in the URL: applying them here touches only the trees,
  // and toolbar clicks navigate so the re-render lands back here.
  const viewM = ctx.params.get("view") === "generations" ? "generations" : "tree";
  const orientC = ctx.params.get("orient") === "horizontal" ? "horizontal" : "vertical";
  const colorC = ctx.params.get("colormode") === "failure" ? "failure" : "creator";

  if (viewM === "generations") {
    for (const t of trees) t.setViewMode("generations");
    container.querySelector("#lc-viewmode").textContent = "Tree view";
  }
  if (orientC === "horizontal") for (const t of trees) t.setOrientation("horizontal");
  // The failure lens across archives: one merged legend, first color wins
  // when two trees hand the same fingerprint different palette slots.
  if (colorC === "failure") {
    for (const t of trees) t.setColorMode("failure");
    container.querySelector("#lc-colormode").textContent = "Color by search";
    const seen = new Map();
    for (const t of trees)
      for (const f of t.fingerprintLegend())
        if (!seen.has(f.label)) seen.set(f.label, f.color);
    container.querySelector("#lc-legend").innerHTML = [...seen.entries()]
      .map(
        ([label, color]) =>
          `<span class="chip" style="background:${color}">${escapeHtml(
            label.length > 42 ? label.slice(0, 41) + "…" : label
          )}</span>`
      )
      .join(" ");
  }

  container.querySelector("#lc-viewmode").addEventListener("click", () => {
    ctx.navigate(null, { view: viewM === "tree" ? "generations" : null });
  });
  container.querySelector("#lc-fit").addEventListener("click", () => {
    for (const t of trees) t.zoomToFit();
  });
  wireExpandButton(container, ctx, "#lc-fullscreen");
  container.querySelector("#lc-orient").addEventListener("click", () => {
    ctx.navigate(null, { orient: orientC === "vertical" ? "horizontal" : null });
  });
  container.querySelector("#lc-colormode").addEventListener("click", () => {
    ctx.navigate(null, { colormode: colorC === "creator" ? "failure" : null });
  });

  container.querySelector("#lc-export").addEventListener("click", () => {
    body.archives.forEach((arch, i) => {
      trees[i].exportSVG({
        title: `Lineage: ${arch.label}`,
        subtitle: "Same task, different search; the tree is the method's shape.",
      });
    });
  });
}
