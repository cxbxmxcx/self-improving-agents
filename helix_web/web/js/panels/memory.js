// Memory panel (chapter 5): memory entries as artifacts, ranked by utility.
// Tier tabs and the scope filter live in the URL hash, so a filtered view is
// a link; clicking a row opens the entry's content, chain, and utility history.

import * as api from "../api.js";
import { renderOutline } from "../components/outline.js";
import { dataTable } from "../components/table.js";
import { openDrawer } from "../components/drawer.js";
import { tiles } from "../components/tiles.js";
import { escapeHtml, fmtInt, fmtScore, fmtWhen, truncate } from "../format.js";

const TIERS = ["all", "episodic", "semantic", "procedural"];

// The spec outline stays as the empty state until an archive writes memory.
function renderEmpty(container) {
  renderOutline(container, {
    title: "Memory",
    caption:
      "Every memory entry is an artifact with a parent pointer and a utility " +
      "score the agent's own outcomes keep updating. Tiers across the top, " +
      "the ranked store in the middle, the last consolidation pass below.",
    chapter: 5,
    route: "#/memory?archive=<key>&tier=episodic|semantic|procedural&scope=<user|org|global>",
    endpoints: [
      "GET /api/archives/{key}/memory?tier=&scope=",
      "GET /api/archives/{key}/memory/{id}",
    ],
    events: ["MemoryRead", "MemoryWritten", "Consolidated"],
    figures: [
      "5.3: an entry's utility rising and falling over turns",
      "5.5: the memory lane growing under live traffic",
    ],
    regions: [
      { name: "Tier tabs", note: "episodic · semantic · procedural, with scope filter (session, user, org, global)", grow: true },
      { name: "Ranked entries", note: "utility, recency, access count, operator that bred it (ADD, UPVOTE, DOWNVOTE, EDIT); sparkline of utility over turns; re-ranks live" },
      { name: "Entry drawer", note: "content, lineage (parent chain), the trajectories that recalled it, utility history" },
      { name: "Consolidation strip", note: "last 3 AM pass: merged, pruned, promoted; next scheduled run", grow: true },
    ],
  });
}

function tierTabsHtml(active, facets) {
  return TIERS.map((t) => {
    const count = t === "all"
      ? Object.values(facets).reduce((a, b) => a + b, 0)
      : facets[t] ?? 0;
    const on = (active || "all") === t;
    return `<button class="btn${on ? " primary" : ""}" data-tier="${t}"
      ${count === 0 && t !== "all" ? "disabled" : ""}>${escapeHtml(t)} (${fmtInt(count)})</button>`;
  }).join("");
}

function scopeChipsHtml(active, facets) {
  const names = Object.keys(facets).sort();
  if (!names.length) return "";
  return names
    .map((s) => {
      const on = active === s;
      return `<label class="chip${on ? "" : " outline"}" style="cursor:pointer; display:inline-flex;
        align-items:center; gap:5px; padding:3px 9px">
        <input type="checkbox" data-scope="${escapeHtml(s)}" ${on ? "checked" : ""} style="margin:0">
        ${escapeHtml(s)} (${fmtInt(facets[s])})</label>`;
    })
    .join("");
}

async function openEntry(ctx, artifactId) {
  const d = await api.get(
    `/api/archives/${ctx.archiveKey}/memory/${encodeURIComponent(artifactId)}`,
    {},
    { signal: ctx.signal }
  );
  const el = document.createElement("div");
  const chain = d.versions
    .map(
      (v) => `<li><b>v${v.version}</b> · ${escapeHtml(v.operator ?? "?")} · utility
        ${fmtScore(v.utility, 2)} · ${escapeHtml(v.created_by)} · ${escapeHtml(fmtWhen(v.created_at))}
        ${v.parent ? `<span style="color:var(--fg-muted)">(parent v${v.parent.version})</span>` : "<span style=\"color:var(--fg-muted)\">(genesis)</span>"}</li>`
    )
    .join("");
  const history = d.utility_history.length
    ? `<table class="data"><thead><tr><th>#</th><th class="num">utility</th><th>signal</th><th>feedback</th><th>recorded</th></tr></thead>
       <tbody>${d.utility_history
         .map(
           (h, i) =>
             `<tr><td>${i + 1}</td><td class="num">${fmtScore(h.score, 2)}</td>` +
             `<td>${escapeHtml(h.signal_id ?? "")}</td><td>${escapeHtml(truncate(h.feedback, 120))}</td>` +
             `<td>${escapeHtml(fmtWhen(h.recorded_at))}</td></tr>`
         )
         .join("")}</tbody></table>`
    : `<div class="placeholder">No utility measurements recorded yet.</div>`;
  el.innerHTML = `
    <h2 class="panel-title" style="word-break:break-all">${escapeHtml(d.artifact_id)}</h2>
    <div class="tiles" style="margin-bottom:var(--sp-3)">
      <div class="tile"><div class="label">Utility</div><div class="value">${fmtScore(d.utility, 2)}</div></div>
      <div class="tile"><div class="label">Tier</div><div class="value" style="font-size:var(--fs-md)">${escapeHtml(d.tier)}</div></div>
      <div class="tile"><div class="label">Scope</div><div class="value" style="font-size:var(--fs-md)">${escapeHtml(d.scope)}</div></div>
      <div class="tile"><div class="label">Versions</div><div class="value">${fmtInt(d.n_versions)}</div></div>
    </div>
    <p class="panel-caption">Latest v${d.version} by <b>${escapeHtml(d.operator ?? "?")}</b>
      · ${escapeHtml(d.created_by)} · ${escapeHtml(fmtWhen(d.created_at))}</p>
    <section class="block">
      <h3>Content</h3>
      <pre class="content">${escapeHtml(JSON.stringify(d.content.entry ?? d.content, null, 2))}</pre>
    </section>
    <section class="block" style="margin-top:var(--sp-4)">
      <h3>Version chain (${d.versions.length})</h3>
      <ul>${chain}</ul>
    </section>
    <section class="block" style="margin-top:var(--sp-4)">
      <h3>Utility history (${d.utility_history.length})</h3>
      ${history}
    </section>`;
  openDrawer(el, { onClose: () => ctx.navigate(null, { sel: null }) });
}

export async function render(container, ctx) {
  if (!ctx.archiveKey) {
    container.innerHTML = `<div class="placeholder">Pick an archive in the header to begin.</div>`;
    return;
  }
  const tier = ctx.params.get("tier") || "";
  const scope = ctx.params.get("scope") || "";
  const body = await api.get(
    `/api/archives/${ctx.archiveKey}/memory`,
    { tier: tier === "all" ? "" : tier, scope },
    { signal: ctx.signal }
  );
  if (!body.grand_total) {
    renderEmpty(container);
    return;
  }

  container.innerHTML = `
    <h2 class="panel-title">Memory</h2>
    <p class="panel-caption">Every row is one memory entry at its latest version, ranked
      by the utility its own outcomes keep updating. Each vote or edit is a new version
      with a parent pointer, never an overwrite.</p>
    <section class="block" id="mem-tabs" style="display:flex; gap:var(--sp-2); flex-wrap:wrap"></section>
    <section class="block" id="mem-scopes" style="display:flex; gap:var(--sp-2); flex-wrap:wrap; align-items:center"></section>
    <section class="block" id="mem-tiles"></section>
    <p class="panel-caption">Showing ${fmtInt(body.total)} of ${fmtInt(body.grand_total)} entries.</p>
    <section class="block"><div id="mem-table"></div></section>`;

  const tabs = container.querySelector("#mem-tabs");
  tabs.innerHTML = tierTabsHtml(tier, body.facets.tiers);
  tabs.addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-tier]");
    if (!btn) return;
    ctx.navigate(null, { tier: btn.dataset.tier === "all" ? null : btn.dataset.tier, sel: null });
  });

  const scopes = container.querySelector("#mem-scopes");
  scopes.innerHTML = scopeChipsHtml(scope, body.facets.scopes);
  scopes.addEventListener("change", (e) => {
    const box = e.target.closest("input[data-scope]");
    if (!box) return;
    ctx.navigate(null, { scope: box.checked ? box.dataset.scope : null, sel: null });
  });

  const byOperator = {};
  for (const r of body.items) byOperator[r.operator ?? "?"] = (byOperator[r.operator ?? "?"] ?? 0) + 1;
  container.querySelector("#mem-tiles").appendChild(
    tiles(
      Object.entries(byOperator)
        .sort((a, b) => b[1] - a[1])
        .map(([op, n]) => ({ label: `latest op: ${op}`, value: fmtInt(n) }))
    )
  );

  container.querySelector("#mem-table").appendChild(
    dataTable({
      columns: [
        { key: "utility", label: "Utility", align: "right", format: (v) => fmtScore(v, 2) },
        { key: "tier", label: "Tier" },
        { key: "scope", label: "Scope" },
        { key: "n_versions", label: "Versions", align: "right" },
        { key: "operator", label: "Operator" },
        { key: "preview", label: "Preview", format: (v) => truncate(v, 120) },
      ],
      rows: body.items,
      empty: "No entries match this tier and scope.",
      onRowClick: (row) => ctx.navigate(null, { sel: row.artifact_id }),
    })
  );

  const sel = ctx.params.get("sel");
  if (sel && body.items.some((r) => r.artifact_id === sel)) {
    try {
      await openEntry(ctx, sel);
    } catch (err) {
      if (err.name !== "AbortError") container.insertAdjacentHTML(
        "beforeend",
        `<div class="banner error">${escapeHtml(err.message ?? String(err))}</div>`
      );
    }
  }
}
