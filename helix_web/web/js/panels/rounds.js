// Rounds panel: the jsonl round log when a loop wrote one, else a per-version
// score series derived from the archive itself.

import * as api from "../api.js";
import { barChart, lineChart } from "../components/charts.js";
import { dataTable } from "../components/table.js";
import { tiles } from "../components/tiles.js";
import { escapeHtml, fmtInt, fmtScore } from "../format.js";

// One placeholder for every empty cell, borrowed from the score formatter so
// the panel never hard-codes the glyph.
const DASH = fmtScore(null);

// Columns the round log may carry; only the ones present in row 0 are shown.
const ROUND_COLUMNS = [
  { key: "round", label: "round", align: "right", format: (v) => fmtInt(v) },
  { key: "candidate_version", label: "candidate", align: "right", format: (v) => (v === null || v === undefined ? DASH : `v${v}`) },
  { key: "mean_score", label: "mean score", align: "right", format: (v) => fmtScore(v) },
  { key: "win_rate", label: "win rate", align: "right", format: (v) => fmtScore(v) },
  { key: "n_wins", label: "wins", align: "right", format: (v) => fmtInt(v) },
  { key: "n_losses", label: "losses", align: "right", format: (v) => fmtInt(v) },
  { key: "n_ties", label: "ties", align: "right", format: (v) => fmtInt(v) },
  { key: "promoted", label: "promoted", format: (v) => (v ? "yes" : "no") },
  { key: "elapsed_reference_sec", label: "ref sec", align: "right", format: fmtSeconds },
  { key: "elapsed_candidate_sec", label: "cand sec", align: "right", format: fmtSeconds },
  { key: "elapsed_judge_sec", label: "judge sec", align: "right", format: fmtSeconds },
];

function fmtSeconds(v) {
  return v === null || v === undefined ? DASH : `${Number(v).toFixed(1)}s`;
}

function focusParams(ctx) {
  const params = {};
  if (ctx.focus) {
    params.focus_id = ctx.focus.id;
    if (ctx.focus.kind) params.focus_kind = ctx.focus.kind;
  }
  return params;
}

// A selected row is echoed as a plain key/value card so deep links land on it.
function detailCard(title, row) {
  const body = Object.entries(row)
    .map(
      ([k, v]) =>
        `<div><span class="label">${escapeHtml(k)}</span> ` +
        `<strong>${escapeHtml(
          v === null || v === undefined
            ? DASH
            : typeof v === "object"
              ? JSON.stringify(v)
              : String(v)
        )}</strong></div>`
    )
    .join("");
  const el = document.createElement("div");
  el.className = "card";
  el.innerHTML = `<h3 style="margin-top:0">${escapeHtml(title)}</h3>
    <div style="display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
      gap:var(--sp-2); font-size:var(--fs-sm)">${body}</div>`;
  return el;
}

function renderJsonl(container, ctx, body) {
  const rows = body.rounds;
  const s = body.summary;
  const selected = ctx.params.get("round");

  container.innerHTML = `
    <h2 class="panel-title">Rounds</h2>
    <p class="panel-caption">Read from the round log beside the archive:
      ${escapeHtml(body.source_file ?? "")}. Stars mark rounds whose candidate
      was promoted to champion.</p>
    <section class="block" id="rd-tiles"></section>
    <section class="block"><h3>Score progression</h3><div id="rd-line"></div></section>
    <section class="block" id="rd-detail"></section>
    <section class="block"><h3>Round details</h3>
      <p class="panel-caption">Click a row to pin it in the URL.</p>
      <div id="rd-table"></div></section>
  `;

  container.querySelector("#rd-tiles").appendChild(
    tiles([
      { label: "Total rounds", value: fmtInt(s.n_rounds), accent: "slate" },
      {
        label: "Promotions",
        value: fmtInt(s.n_promotions),
        accent: s.n_promotions ? "teal" : undefined,
        sub: "candidate beat the reference",
      },
      { label: "Best score", value: fmtScore(s.best_score), muted: s.best_score === null },
      { label: "Mean score", value: fmtScore(s.mean_score), muted: s.mean_score === null },
    ])
  );

  // The round number is the x axis; rows without a score cannot be plotted.
  const withRound = rows.map((r, i) => ({ ...r, _x: r.round ?? i + 1 }));
  const points = withRound
    .filter((r) => r.mean_score !== null && r.mean_score !== undefined)
    .map((r) => ({
      x: r._x,
      y: r.mean_score,
      tooltip: `round ${r._x}\nscore ${fmtScore(r.mean_score)}`,
    }));
  const stars = withRound
    .filter((r) => r.promoted && r.mean_score !== null && r.mean_score !== undefined)
    .map((r) => ({ x: r._x, y: r.mean_score }));
  lineChart(
    container.querySelector("#rd-line"),
    [{ name: "mean score", color: "#3C5488", points }],
    { xLabel: "round", stars }
  );

  const first = rows[0] ?? {};
  const columns = ROUND_COLUMNS.filter((c) => c.key in first);
  container.querySelector("#rd-table").appendChild(
    dataTable({
      columns,
      rows,
      empty: "No rounds in the log.",
      onRowClick: (row) => ctx.navigate(null, { round: row.round ?? null }),
    })
  );

  if (selected) {
    const hit = rows.find((r) => String(r.round) === selected);
    if (hit) {
      container.querySelector("#rd-detail").appendChild(detailCard(`Round ${selected}`, hit));
    }
  }
}

function renderArchive(container, ctx, body) {
  const series = body.fallback_series;
  const measured = series.filter((r) => r.score !== null && r.score !== undefined);
  const unmeasured = series.length - measured.length;
  const selected = ctx.params.get("ver");

  if (!measured.length) {
    container.innerHTML = `
      <h2 class="panel-title">Rounds</h2>
      <div class="banner info">No round history and no measured artifacts in this archive.</div>`;
    return;
  }

  container.innerHTML = `
    <h2 class="panel-title">Rounds</h2>
    <p class="panel-caption">No round log found next to this archive; showing the
      latest score per version instead.</p>
    <section class="block"><h3>Latest score by version</h3>
      <div id="rd-bar"></div>
      <p class="panel-caption" id="rd-bar-note"></p></section>
    <section class="block" id="rd-detail"></section>
    <section class="block"><h3>Versions</h3>
      <p class="panel-caption">Click a row to pin it in the URL.</p>
      <div id="rd-table"></div></section>
  `;

  barChart(
    container.querySelector("#rd-bar"),
    series.map((r) => ({
      x: r.version,
      y: r.score,
      colorBy: r.created_by,
      tooltip: `v${r.version} ${r.created_by}: ${fmtScore(r.score)}`,
    })),
    { xLabel: "version" }
  );
  container.querySelector("#rd-bar-note").textContent = unmeasured
    ? `${unmeasured} version${unmeasured === 1 ? "" : "s"} carry no measurement and draw no bar.`
    : "Bar color follows the search method that created the version.";

  container.querySelector("#rd-table").appendChild(
    dataTable({
      columns: [
        { key: "version", label: "Version", align: "right", format: (v) => `v${v}` },
        { key: "created_by", label: "Created by" },
        { key: "search_method", label: "Search method" },
        { key: "score", label: "Score", align: "right", format: (v) => fmtScore(v) },
        { key: "round", label: "Round", align: "right", format: (v) => fmtInt(v) },
      ],
      rows: series,
      empty: "No versions match the focus.",
      onRowClick: (row) => ctx.navigate(null, { ver: row.version ?? null }),
    })
  );

  if (selected) {
    const hit = series.find((r) => String(r.version) === selected);
    if (hit) {
      container.querySelector("#rd-detail").appendChild(detailCard(`Version v${selected}`, hit));
    }
  }
}

export async function render(container, ctx) {
  if (!ctx.archiveKey) {
    container.innerHTML = `<div class="placeholder">Pick an archive in the header to begin.</div>`;
    return;
  }
  const body = await api.get(`/api/archives/${ctx.archiveKey}/rounds`, focusParams(ctx), {
    signal: ctx.signal,
  });

  if (body.source === "jsonl" && body.rounds.length) {
    renderJsonl(container, ctx, body);
  } else {
    renderArchive(container, ctx, body);
  }
}
