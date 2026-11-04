// Compare panel: two versions side by side, with score tiles, a content diff,
// and the measurement histories that explain how each score was earned.

import * as api from "../api.js";
import { lineChart } from "../components/charts.js";
import { diffView } from "../components/diff.js";
import { dataTable } from "../components/table.js";
import { tiles } from "../components/tiles.js";
import { escapeHtml, fmtScore } from "../format.js";

const A_COLOR = "#3C5488"; // slate, the baseline
const B_COLOR = "#E64B35"; // vermillion, the candidate

const SELECT_CSS =
  "font:inherit; font-size:var(--fs-sm); padding:var(--sp-1) var(--sp-2);" +
  "border:1px solid var(--border-strong); border-radius:var(--radius-sm);" +
  "background:var(--bg-panel); color:var(--fg); max-width:420px";
const LABEL_CSS = "font-size:var(--fs-xs); color:var(--fg-muted)";

function focusParams(ctx) {
  const params = {};
  if (ctx.focus) {
    params.focus_id = ctx.focus.id;
    if (ctx.focus.kind) params.focus_kind = ctx.focus.kind;
  }
  return params;
}

// Refs are "<artifact id>__v<n>" and ids may themselves contain "__v",
// so split on the last occurrence exactly like the server's parse_ref.
function parseRef(value) {
  const cut = value.lastIndexOf("__v");
  return { id: value.slice(0, cut), version: Number(value.slice(cut + 3)) };
}

function scoreOf(variant) {
  return variant?.measurement?.score ?? null;
}

function optionsHtml(items, selected) {
  return items
    .map(
      (v) =>
        `<option value="${escapeHtml(v.ref)}"${v.ref === selected ? " selected" : ""}>` +
        `v${v.version} · ${escapeHtml(v.created_by)} · ${fmtScore(scoreOf(v))}</option>`
    )
    .join("");
}

// The useful default is the first variant against its strongest rival.
function pickDefaults(items) {
  const rivals = items.slice(1);
  const scored = rivals.filter((v) => scoreOf(v) !== null);
  const best = scored.length
    ? scored.reduce((top, v) => (scoreOf(v) > scoreOf(top) ? v : top))
    : rivals[0];
  return { a: items[0].ref, b: best.ref };
}

function winnerLabel(vA, vB) {
  const sa = scoreOf(vA);
  const sb = scoreOf(vB);
  if (sa === null && sb === null) return null;
  if (sa === null) return `v${vB.version}`;
  if (sb === null) return `v${vA.version}`;
  if (sa === sb) return "tie";
  return sb > sa ? `v${vB.version}` : `v${vA.version}`;
}

function seriesFor(name, color, history) {
  return {
    name,
    color,
    // Index over the full history so a null-scored measurement still costs a slot.
    points: history
      .map((m, i) => ({ m, n: i + 1 }))
      .filter(({ m }) => m.score !== null && m.score !== undefined)
      .map(({ m, n }) => ({
        x: n,
        y: m.score,
        tooltip:
          `measurement #${n}\nscore ${fmtScore(m.score)}\n` +
          `preference ${m.preference}\nconfidence ${fmtScore(m.confidence, 2)}`,
      })),
  };
}

function historyTable(title, history) {
  const box = document.createElement("div");
  box.style.cssText = "flex:1 1 300px; min-width:280px";
  box.innerHTML = `<p class="panel-caption" style="margin-bottom:var(--sp-2)">${escapeHtml(title)}</p>`;
  box.appendChild(
    dataTable({
      columns: [
        { key: "n", label: "#", align: "right" },
        { key: "score", label: "score", align: "right", format: (v) => fmtScore(v) },
        { key: "preference", label: "pref" },
        { key: "confidence", label: "confidence", align: "right", format: (v) => fmtScore(v, 2) },
      ],
      rows: history.map((m, i) => ({
        n: i + 1,
        score: m.score,
        preference: m.preference,
        confidence: m.confidence,
      })),
      empty: "No measurements recorded.",
    })
  );
  return box;
}

async function fetchHistory(ctx, variant) {
  const body = await api.get(
    `/api/archives/${ctx.archiveKey}/variants/${encodeURIComponent(variant.id)}/${variant.version}/measurements`,
    {},
    { signal: ctx.signal }
  );
  return body.items;
}

// Route aborts must keep bubbling; anything else degrades to a local banner.
function sectionError(host, err, what) {
  if (err.name === "AbortError") throw err;
  host.innerHTML = `<div class="banner error">Could not load ${escapeHtml(what)}: ${escapeHtml(err.message ?? String(err))}</div>`;
}

export async function render(container, ctx) {
  if (!ctx.archiveKey) {
    container.innerHTML = `<div class="placeholder">Pick an archive in the header to begin.</div>`;
    return;
  }
  const body = await api.get(
    `/api/archives/${ctx.archiveKey}/variants`,
    focusParams(ctx),
    { signal: ctx.signal }
  );
  const items = body.items;

  const head = `
    <h2 class="panel-title">Compare</h2>
    <p class="panel-caption">Two versions side by side: what the search changed in the
      content, and whether that change moved the score.</p>`;

  if (items.length < 2) {
    container.innerHTML = `${head}
      <div class="banner info">Comparing needs at least two variants; this archive has
        ${items.length}${ctx.focus ? " under the current focus" : ""}. Widen the focus
        picker or run another round.</div>`;
    return;
  }

  const byRef = new Map(items.map((v) => [v.ref, v]));
  const defaults = pickDefaults(items);
  const aRef = byRef.has(ctx.params.get("a")) ? ctx.params.get("a") : defaults.a;
  const bRef = byRef.has(ctx.params.get("b")) ? ctx.params.get("b") : defaults.b;

  container.innerHTML = `${head}
    <section class="block" style="display:flex; gap:var(--sp-4); flex-wrap:wrap">
      <div><div style="${LABEL_CSS}">A (baseline)</div>
        <select id="cmp-a" style="${SELECT_CSS}">${optionsHtml(items, aRef)}</select></div>
      <div><div style="${LABEL_CSS}">B (candidate)</div>
        <select id="cmp-b" style="${SELECT_CSS}">${optionsHtml(items, bRef)}</select></div>
    </section>
    <div id="cmp-body"></div>`;

  // Selection lives in the URL so any comparison can be linked to directly.
  container.querySelector("#cmp-a").addEventListener("change", (e) => {
    ctx.navigate(null, { a: e.target.value });
  });
  container.querySelector("#cmp-b").addEventListener("change", (e) => {
    ctx.navigate(null, { b: e.target.value });
  });

  const host = container.querySelector("#cmp-body");
  if (aRef === bRef) {
    host.innerHTML = `<div class="banner warning">A and B are the same version. Pick a
      different candidate to see a diff.</div>`;
    return;
  }

  const vA = byRef.get(aRef);
  const vB = byRef.get(bRef);
  const sa = scoreOf(vA);
  const sb = scoreOf(vB);
  const delta = sa !== null && sb !== null ? sb - sa : null;
  const winner = winnerLabel(vA, vB);

  host.innerHTML = `
    <section class="block" id="cmp-tiles"></section>
    <section class="block"><h3>Content diff</h3><div id="cmp-diff"></div></section>
    <section class="block"><h3>Measurement history</h3>
      <div id="cmp-chart"></div>
      <div id="cmp-hist" style="display:flex; gap:var(--sp-4); flex-wrap:wrap; margin-top:var(--sp-4)"></div>
    </section>`;

  host.querySelector("#cmp-tiles").appendChild(
    tiles([
      { label: `A · v${vA.version} score`, value: fmtScore(sa), accent: "slate", muted: sa === null },
      {
        label: `B · v${vB.version} score`,
        value: fmtScore(sb),
        accent: delta === null ? undefined : delta > 0 ? "teal" : "vermillion",
        muted: sb === null,
        sub: delta === null ? undefined : `${delta >= 0 ? "+" : ""}${delta.toFixed(3)} vs A`,
      },
      // The missing-value glyph comes from fmtScore so it stays consistent.
      { label: "Winner by score", value: winner ?? fmtScore(null), muted: !winner },
    ])
  );

  const a = parseRef(aRef);
  const b = parseRef(bRef);
  const diffPromise = api.get(
    `/api/archives/${ctx.archiveKey}/diff`,
    { a_id: a.id, a_version: a.version, b_id: b.id, b_version: b.version },
    { signal: ctx.signal }
  );
  const historyPromise = Promise.all([fetchHistory(ctx, a), fetchHistory(ctx, b)]);

  const diffHost = host.querySelector("#cmp-diff");
  try {
    diffHost.appendChild(diffView(await diffPromise));
  } catch (err) {
    sectionError(diffHost, err, "the content diff");
  }

  const chartHost = host.querySelector("#cmp-chart");
  try {
    const [aHist, bHist] = await historyPromise;
    if (!aHist.length && !bHist.length) {
      chartHost.innerHTML = `<p class="panel-caption">(neither artifact has a measurement history)</p>`;
    } else {
      lineChart(
        chartHost,
        [
          seriesFor(`v${vA.version}`, A_COLOR, aHist),
          seriesFor(`v${vB.version}`, B_COLOR, bHist),
        ],
        { refLine: 0.5, xLabel: "measurement #" }
      );
    }
    const histHost = host.querySelector("#cmp-hist");
    histHost.appendChild(historyTable(`A · v${vA.version}`, aHist));
    histHost.appendChild(historyTable(`B · v${vB.version}`, bHist));
  } catch (err) {
    sectionError(chartHost, err, "the measurement history");
  }
}
