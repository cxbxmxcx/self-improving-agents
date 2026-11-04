// Verdicts panel: per-question judge calls, facet filters, and the win heatmap.

import * as api from "../api.js";
import { dataTable } from "../components/table.js";
import { tiles } from "../components/tiles.js";
import { heatmap } from "../components/charts.js";
import { escapeHtml, fmtInt, fmtScore, fmtWhen, truncate } from "../format.js";

// URL param name -> server query param name, in the order the filter row shows them.
const FILTERS = [
  { param: "v", server: "version", label: "Version", facet: "versions", prefix: "v" },
  { param: "band", server: "band", label: "Band", facet: "bands", prefix: "band " },
  { param: "pref", server: "preference", label: "Preference", facet: "preferences", prefix: "" },
  { param: "role", server: "role", label: "Role", facet: "roles", prefix: "" },
];

// Selections ride in the hash as one comma-joined param per filter, so a
// filtered view is a link somebody can paste into a chapter draft.
function readSelections(params) {
  const sel = {};
  for (const f of FILTERS) {
    const raw = params.get(f.param) ?? "";
    sel[f.param] = raw ? raw.split(",").filter(Boolean) : [];
  }
  return sel;
}

// api.get only takes scalar params, so repeated list filters need a hand-built query.
function buildQuery(ctx, sel) {
  const qs = new URLSearchParams();
  if (ctx.focus) qs.set("focus_id", ctx.focus.id);
  for (const f of FILTERS) {
    for (const value of sel[f.param]) qs.append(f.server, value);
  }
  return qs.toString();
}

function chipGroupHtml(filter, options, active) {
  if (!options.length) return "";
  const chips = options
    .map((option) => {
      const value = String(option);
      const on = active.includes(value);
      return `
        <label class="chip${on ? "" : " outline"}" style="cursor:pointer; display:inline-flex;
          align-items:center; gap:5px; padding:3px 9px">
          <input type="checkbox" data-filter="${escapeHtml(filter.param)}"
            value="${escapeHtml(value)}" ${on ? "checked" : ""} style="margin:0">
          ${escapeHtml(filter.prefix + value)}
        </label>`;
    })
    .join("");
  return `
    <div style="display:flex; align-items:center; gap:var(--sp-2); flex-wrap:wrap">
      <span style="font-size:var(--fs-xs); color:var(--fg-muted); text-transform:uppercase;
        letter-spacing:0.04em; min-width:78px">${escapeHtml(filter.label)}</span>
      ${chips}
    </div>`;
}

function preferenceAccent(preference) {
  const p = String(preference).toLowerCase();
  if (p === "left") return "teal";
  if (p === "right") return "slate";
  return undefined;
}

export async function render(container, ctx) {
  if (!ctx.archiveKey) {
    container.innerHTML = `<div class="placeholder">Pick an archive in the header to begin.</div>`;
    return;
  }

  const sel = readSelections(ctx.params);
  const query = buildQuery(ctx, sel);
  const body = await api.get(
    `/api/archives/${ctx.archiveKey}/verdicts${query ? "?" + query : ""}`,
    {},
    { signal: ctx.signal }
  );

  container.innerHTML = `
    <h2 class="panel-title">Verdicts</h2>
    <p class="panel-caption">Every row is one judge call on one question.
      LEFT = candidate won; RIGHT = reference won; TIE = neither.</p>
    <div id="vd-body"></div>
  `;
  const host = container.querySelector("#vd-body");

  if (!body.grand_total) {
    host.innerHTML = `<div class="banner info">No per-question verdicts in the archive yet.
      Pairwise signals write them into measurement metadata, so run a loop that judges
      candidate against reference to fill this panel.</div>`;
    return;
  }

  const showingAll = body.items.length >= body.total;
  host.innerHTML = `
    ${
      body.n_from_failed
        ? `<div class="banner warning">${fmtInt(body.n_from_failed)} of
             ${fmtInt(body.grand_total)} verdicts come from runs whose trajectory later
             failed. Treat those preferences as unreliable evidence.</div>`
        : ""
    }
    <section class="block" id="vd-filters" style="display:flex; flex-direction:column;
      gap:var(--sp-2)"></section>
    <p class="panel-caption">Showing ${fmtInt(body.total)} of
      ${fmtInt(body.grand_total)} verdicts.${
        showingAll ? "" : ` The table lists the first ${fmtInt(body.items.length)}.`
      }</p>
    <section class="block" id="vd-tiles"></section>
    <section class="block"><div id="vd-table"></div></section>
    <section class="block">
      <h3>Preference heatmap</h3>
      <p class="panel-caption">Latest verdict per question and version. The heatmap
        answers "which questions did this version stop losing?" so it deliberately
        ignores the filters above.</p>
      <div id="vd-heatmap"><div class="placeholder">Loading heatmap…</div></div>
    </section>
  `;

  const filterHost = host.querySelector("#vd-filters");
  filterHost.innerHTML = FILTERS.map((f) =>
    chipGroupHtml(f, body.facets[f.facet] ?? [], sel[f.param])
  ).join("");
  // One handler for all four groups: collect every box, then navigate once.
  filterHost.addEventListener("change", () => {
    const patch = {};
    for (const f of FILTERS) {
      const checked = [...filterHost.querySelectorAll(`input[data-filter="${f.param}"]:checked`)];
      patch[f.param] = checked.length ? checked.map((el) => el.value).join(",") : null;
    }
    ctx.navigate(null, patch);
  });

  const counts = Object.entries(body.counts_by_preference).sort((a, b) => b[1] - a[1]);
  host.querySelector("#vd-tiles").appendChild(
    tiles(
      counts.length
        ? counts.map(([preference, count]) => ({
            label: preference || "(unlabeled)",
            value: fmtInt(count),
            accent: preferenceAccent(preference),
          }))
        : [{ label: "Verdicts", value: "0", muted: true }]
    )
  );

  host.querySelector("#vd-table").appendChild(
    dataTable({
      columns: [
        { key: "version", label: "Version", align: "right", format: (v) => `v${v}` },
        { key: "created_by", label: "Created by" },
        { key: "question_id", label: "Question" },
        { key: "band", label: "Band", align: "right" },
        { key: "preference", label: "Preference" },
        { key: "confidence", label: "Confidence", align: "right", format: (v) => fmtScore(v, 2) },
        { key: "candidate_failed", label: "Failed", format: (v) => (v ? "⚠" : "") },
        { key: "role", label: "Role" },
        { key: "feedback", label: "Feedback", format: (v) => truncate(v, 160) },
        { key: "recorded_at", label: "Recorded", format: (v) => fmtWhen(v) },
      ],
      rows: body.items,
      empty: "No verdicts match these filters.",
    })
  );

  const heatHost = host.querySelector("#vd-heatmap");
  try {
    const grid = await api.get(
      `/api/archives/${ctx.archiveKey}/verdicts/heatmap`,
      ctx.focus ? { focus_id: ctx.focus.id } : {},
      { signal: ctx.signal }
    );
    if (grid.questions.length && grid.versions.length) heatmap(heatHost, grid);
    else heatHost.innerHTML = `<div class="placeholder">Not enough verdicts to plot.</div>`;
  } catch (err) {
    if (err.name === "AbortError") return;
    heatHost.innerHTML = `<div class="banner error">${escapeHtml(
      err.message ?? String(err)
    )}</div>`;
  }
}
