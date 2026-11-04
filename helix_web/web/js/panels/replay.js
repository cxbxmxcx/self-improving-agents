// Replay panel: pick a cached run, walk its steps, and diff it against a sibling.

import * as api from "../api.js";
import { dataTable } from "../components/table.js";
import { diffView } from "../components/diff.js";
import { tiles } from "../components/tiles.js";
import { toast } from "../components/toast.js";
import { escapeHtml, fmtInt, fmtWhen, truncate } from "../format.js";

const STEP_ICONS = {
  model_call: "🤖", tool_call: "🔧", tool_result: "📥",
  hook_fire: "⚡", artifact_load: "📜", human_input: "👤",
};
const PAYLOAD_LIMIT = 2000;
const OUTPUT_LIMIT = 3000;

const PANEL_CSS = `
  #rp-controls select { font:inherit; font-size:var(--fs-sm); padding:var(--sp-1) var(--sp-2);
    border:1px solid var(--border-strong); border-radius:var(--radius-sm);
    background:var(--bg-panel); color:var(--fg); max-width:320px; }
  #rp-controls label { font-size:var(--fs-xs); color:var(--fg-muted); }
  .rp-col { flex:1 1 420px; min-width:320px; }`;

function optionsHtml(values, selected) {
  return values
    .map((v) => `<option value="${escapeHtml(v)}"${
      String(v) === String(selected) ? " selected" : ""}>${escapeHtml(v)}</option>`)
    .join("");
}

// Unique, sorted values of one column across the cached index rows.
function uniq(rows, key) {
  const out = [...new Set(rows.map((r) => r[key]))];
  return typeof out[0] === "number" ? out.sort((a, b) => a - b) : out.sort();
}

function stepTrace(t) {
  const steps = t.steps ?? [];
  const items = steps.map((s, i) => {
    const kind = String(s.kind ?? "unknown");
    const raw = JSON.stringify(s.payload ?? {}, null, 2);
    const cut = raw.length > PAYLOAD_LIMIT;
    return `<details class="expander" style="margin-bottom:var(--sp-2)">
      <summary>${STEP_ICONS[kind] ?? "❓"} Step ${(t.step_offset ?? 0) + i + 1} ·
        ${escapeHtml(kind)}</summary>
      <div class="body"><pre class="content">${escapeHtml(truncate(raw, PAYLOAD_LIMIT))}</pre>
      ${cut ? `<p class="panel-caption" style="margin:var(--sp-2) 0 0">Payload cut at
        ${fmtInt(PAYLOAD_LIMIT)} of ${fmtInt(raw.length)} characters.</p>` : ""}
      </div></details>`;
  });
  const wrap = document.createElement("section");
  wrap.className = "block";
  wrap.style.marginTop = "var(--sp-4)";
  wrap.innerHTML =
    `<h3>Step trace (${fmtInt(t.n_steps)})</h3>` +
    (steps.length ? items.join("") : `<div class="placeholder">No steps recorded.</div>`) +
    (t.steps_truncated
      ? `<p class="panel-caption" style="margin-top:var(--sp-2)">Some payloads were trimmed
         server side to stay under the content budget.</p>`
      : "");
  return wrap;
}

function trajectoryView(t) {
  const failed = t.status === "failed";
  const el = document.createElement("div");
  el.innerHTML = `<p class="panel-caption" style="margin-bottom:var(--sp-3)">
    <strong>Task:</strong> ${escapeHtml(truncate(t.task ?? "(no task recorded)", 240))}
    · recorded ${escapeHtml(fmtWhen(t.recorded_at))}</p>`;
  el.appendChild(tiles([
    { label: "Outcome", value: String(t.outcome ?? t.status ?? "unknown"),
      accent: failed ? "vermillion" : "teal" },
    { label: "Steps", value: fmtInt(t.n_steps) },
  ]));

  const note = document.createElement("div");
  note.style.marginTop = "var(--sp-3)";
  if (failed) {
    // A failed run has no answer worth showing, so the error is the whole story.
    note.className = "banner error";
    note.innerHTML = `<strong>${escapeHtml(t.error_type ?? "unknown")}</strong>` +
      (t.error_message ? `<br>${escapeHtml(t.error_message)}` : "");
    el.appendChild(note);
    return el;
  }
  note.className = "banner success";
  note.innerHTML = `<strong>Final output</strong>
    <div style="margin-top:var(--sp-2); white-space:pre-wrap">${escapeHtml(
      truncate(t.final_output ?? t.answer ?? "", OUTPUT_LIMIT))}</div>`;
  el.appendChild(note);
  el.appendChild(stepTrace(t));
  return el;
}

async function fillFailedDetails(body, key) {
  const failed = await api.get(`/api/archives/${key}/trajectories/failed`);
  const byType = [
    { key: "error_type", label: "Error type" },
    { key: "count", label: "Count", align: "right" },
    { key: "likely_cause", label: "Likely cause" },
  ];
  const itemCols = [
    { key: "artifact_id", label: "Artifact" },
    { key: "version", label: "Version", align: "right" },
    { key: "question_id", label: "Question" },
    { key: "error_type", label: "Error" },
    { key: "error_message", label: "Message", format: (v) => truncate(v ?? "", 100) },
    { key: "purgeable", label: "Purgeable", format: (v) => (v ? "yes" : "no") },
    { key: "recorded_at", label: "Recorded", format: (v) => fmtWhen(v) },
  ];
  const list = document.createElement("div");
  list.style.marginTop = "var(--sp-4)";
  list.appendChild(dataTable({ columns: itemCols, rows: failed.items, empty: "No failed runs." }));
  body.innerHTML = "";
  body.appendChild(dataTable({ columns: byType, rows: failed.by_type, empty: "No failures." }));
  body.appendChild(list);
}

function mountFailedBlock(host, container, ctx, index) {
  // The two counts diverge by design, so name the rule that separates them.
  const split = index.n_failed !== index.n_purgeable
    ? `<p class="panel-caption">The counts differ on purpose: the display rule flags any run
       carrying an error or no output, while the purge rule also drops zero-step rows that
       still carry an output.</p>`
    : "";
  host.innerHTML = `
    <div class="banner error">${fmtInt(index.n_failed)} failed runs cached</div>
    ${index.n_purgeable > 0 ? `<button class="btn danger" id="rp-purge"
      style="margin-bottom:var(--sp-3)">🗑 Purge ${fmtInt(index.n_purgeable)} purgeable</button>`
      : ""}${split}
    <details class="expander" id="rp-failed"><summary>Inspect failed trajectories</summary>
      <div class="body"><div class="placeholder">Loading…</div></div></details>`;

  const purge = host.querySelector("#rp-purge");
  if (purge) purge.addEventListener("click", async () => {
    if (!window.confirm(`Delete ${index.n_purgeable} purgeable trajectories?`)) return;
    purge.disabled = true;
    try {
      const path = `/api/archives/${ctx.archiveKey}/trajectories/purge`;
      const res = await api.postJSON(path, { confirm: true });
      toast(`Deleted ${res.deleted}`);
      await render(container, ctx);
    } catch (err) {
      toast(err.message ?? String(err), "error");
      purge.disabled = false;
    }
  });

  // The failed view is a second query, so it waits until someone opens the expander.
  const details = host.querySelector("#rp-failed");
  details.addEventListener("toggle", async () => {
    if (!details.open || details.dataset.loaded) return;
    details.dataset.loaded = "1";
    const body = details.querySelector(".body");
    try {
      await fillFailedDetails(body, ctx.archiveKey);
    } catch (err) {
      body.innerHTML = `<div class="banner error">${escapeHtml(err.message ?? String(err))}</div>`;
    }
  });
}

export async function render(container, ctx) {
  if (!ctx.archiveKey) {
    container.innerHTML = `<div class="placeholder">Pick an archive in the header to begin.</div>`;
    return;
  }
  const key = ctx.archiveKey;
  const index = await api.get(`/api/archives/${key}/trajectories`, {}, { signal: ctx.signal });

  container.innerHTML = `
    <style>${PANEL_CSS}</style>
    <h2 class="panel-title">Replay</h2>
    <p class="panel-caption">Cached runs replayed step by step, so a score always traces back
      to the trace that earned it. ${fmtInt(index.total)} cached.</p>
    <section class="block" id="rp-failed-block"></section>
    <section class="block" id="rp-controls"></section>
    <section class="block" id="rp-body"></section>`;

  if (index.n_failed > 0) {
    mountFailedBlock(container.querySelector("#rp-failed-block"), container, ctx, index);
  }
  const body = container.querySelector("#rp-body");
  if (!index.items.length) {
    body.innerHTML = `<div class="banner info">No cached trajectories yet.</div>`;
    return;
  }

  // Focus wins over the artifact param; every other pick falls back to the first option.
  const rows = index.items;
  const artifactIds = index.facets.artifact_ids;
  const focusId = ctx.focus?.id ?? null;
  if (focusId && !artifactIds.includes(focusId)) {
    body.innerHTML = `<div class="banner info">No cached trajectories for
      ${escapeHtml(focusId)}. Clear the focus picker to browse the rest.</div>`;
    return;
  }
  const artifactId = focusId ??
    (artifactIds.includes(ctx.params.get("ta")) ? ctx.params.get("ta") : artifactIds[0]);
  const versions = uniq(rows.filter((r) => r.artifact_id === artifactId), "version");
  const askedV = ctx.params.has("tv") ? Number(ctx.params.get("tv")) : NaN;
  const version = versions.includes(askedV) ? askedV : versions[0];
  const questions = uniq(
    rows.filter((r) => r.artifact_id === artifactId && r.version === version), "question_id");
  const askedQ = ctx.params.get("tq");
  const question = questions.includes(askedQ) ? askedQ : questions[0];

  // Only versions holding the same question can be compared without a 404.
  const cmpVersions = uniq(rows.filter((r) => r.artifact_id === artifactId &&
    r.question_id === question && r.version !== version), "version");
  const compareOn = ctx.params.has("tcmp");
  const askedC = compareOn ? Number(ctx.params.get("tcmp")) : NaN;
  const cmpVersion = cmpVersions.includes(askedC) ? askedC : cmpVersions[0] ?? null;

  const controls = container.querySelector("#rp-controls");
  controls.innerHTML = `
    <div style="display:flex; gap:var(--sp-4); align-items:flex-end; flex-wrap:wrap">
      ${focusId ? "" : `<label>Artifact<br>
        <select id="rp-ta">${optionsHtml(artifactIds, artifactId)}</select></label>`}
      <label>Version<br><select id="rp-tv">${optionsHtml(versions, version)}</select></label>
      <label>Question<br><select id="rp-tq">${optionsHtml(questions, question)}</select></label>
      <label style="font-size:var(--fs-sm); color:var(--fg)">
        <input type="checkbox" id="rp-cmp"${compareOn ? " checked" : ""}>
        Compare with another trajectory</label>
      ${compareOn && cmpVersion !== null ? `<label>Against<br>
        <select id="rp-tcmp">${optionsHtml(cmpVersions, cmpVersion)}</select></label>` : ""}
    </div>`;

  // Selections live in the URL, so every view here is a link someone can paste.
  const on = (sel, handler) => {
    const el = controls.querySelector(sel);
    if (el) el.addEventListener("change", handler);
  };
  on("#rp-ta", (e) => ctx.navigate(null, { ta: e.target.value, tv: null, tq: null, tcmp: null }));
  on("#rp-tv", (e) => ctx.navigate(null, { tv: e.target.value, tcmp: null }));
  on("#rp-tq", (e) => ctx.navigate(null, { tq: e.target.value, tcmp: null }));
  on("#rp-tcmp", (e) => ctx.navigate(null, { tcmp: e.target.value }));
  on("#rp-cmp", (e) =>
    ctx.navigate(null, { tcmp: e.target.checked ? cmpVersions[0] ?? "none" : null }));

  const base = `/api/archives/${key}/trajectories/${encodeURIComponent(artifactId)}`;
  const qs = encodeURIComponent(question);
  const primary = await api.get(`${base}/${version}/${qs}`, {}, { signal: ctx.signal });

  body.innerHTML = "";
  const columns = document.createElement("div");
  columns.style.cssText = "display:flex; gap:var(--sp-5); align-items:flex-start; flex-wrap:wrap";
  const primaryCol = document.createElement("div");
  primaryCol.className = "rp-col";
  primaryCol.innerHTML = `<h3>v${version} · ${escapeHtml(question)}</h3>`;
  primaryCol.appendChild(trajectoryView(primary));
  columns.appendChild(primaryCol);
  body.appendChild(columns);

  if (!compareOn) return;
  if (cmpVersion === null) {
    body.insertAdjacentHTML("afterbegin",
      `<div class="banner warning">No other version has this question cached.</div>`);
    return;
  }

  const otherCol = document.createElement("div");
  otherCol.className = "rp-col";
  otherCol.innerHTML = `<h3>v${cmpVersion} · ${escapeHtml(question)}</h3>`;
  columns.appendChild(otherCol);
  try {
    const other = await api.get(`${base}/${cmpVersion}/${qs}`, {}, { signal: ctx.signal });
    otherCol.appendChild(trajectoryView(other));
    const payload = await api.get(`/api/archives/${key}/diff/trajectory`,
      { artifact_id: artifactId, a_version: version, b_version: cmpVersion, question_id: question },
      { signal: ctx.signal });
    const diffBlock = document.createElement("section");
    diffBlock.className = "block";
    diffBlock.style.marginTop = "var(--sp-5)";
    diffBlock.innerHTML = `<h3>Answer diff</h3>`;
    diffBlock.appendChild(diffView(payload));
    body.appendChild(diffBlock);
  } catch (err) {
    otherCol.innerHTML += `<div class="banner error">${escapeHtml(err.message ?? String(err))}</div>`;
  }
}
