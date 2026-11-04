// Champion panel: the best candidate per artifact family, and the gate between
// "best score in the archive" and "what the live agent actually serves."

import * as api from "../api.js";
import { diffView } from "../components/diff.js";
import { tiles } from "../components/tiles.js";
import { toast } from "../components/toast.js";
import { escapeHtml, fmtInt, fmtScore, fmtWhen } from "../format.js";

const DIFF_MODES = [
  { value: "genesis", label: "Genesis (v1 → champion)" },
  { value: "parent", label: "Parent (immediate predecessor)" },
];

function focusParams(ctx) {
  const params = {};
  if (ctx.focus) {
    params.focus_id = ctx.focus.id;
    if (ctx.focus.kind) params.focus_kind = ctx.focus.kind;
  }
  return params;
}

// The crown is a score claim; promotion is a production claim. This block is
// where the two are compared, so the difference stays visible.
function promotionHtml(fam, index) {
  const best = fam.best.version;
  const live = fam.live_version;
  const button =
    `<button class="btn primary" data-promote="${index}">` +
    `👑 Promote v${best} → live</button>`;
  if (live === best) {
    return `<div class="banner success">Live champion v${best} is also the
      best candidate. Nothing pending.</div>`;
  }
  if (live === null || live === undefined) {
    return `<div class="banner warning">No promotion recorded; the live agent
      still serves the genesis prompt.</div>${button}`;
  }
  return `<div class="banner warning">Live champion is v${live} but the best
    candidate is v${best}; winner pending review.</div>${button}`;
}

function lineageHtml(path) {
  if (!path.length) return "";
  const walk = path
    .map(
      (p) =>
        `v${p.version} (${escapeHtml(p.created_by)}, ${fmtScore(p.score)})`
    )
    .join(" → ");
  return `<section class="block">
    <h3>Lineage from genesis to champion</h3>
    <p class="panel-caption" style="margin:0">${walk}</p>
  </section>`;
}

function winsHtml(wins) {
  if (!wins.length) return "";
  const items = wins
    .map(
      (w) => `<details class="expander" style="margin-bottom:var(--sp-2)">
        <summary>🏆 ${escapeHtml(w.question_id)} (band ${escapeHtml(w.band)})
          · confidence ${fmtScore(w.confidence, 2)}</summary>
        <div class="body"><pre class="content">${escapeHtml(w.feedback)}</pre></div>
      </details>`
    )
    .join("");
  return `<section class="block"><h3>Top wins</h3>${items}</section>`;
}

function promotionsHtml(promotions) {
  if (!promotions.length) return "";
  const items = promotions
    .map(
      (p) => `<li>v${p.version} by ${escapeHtml(p.approver)} at
        ${escapeHtml(fmtWhen(p.promoted_at))} · ${escapeHtml(p.reason)}</li>`
    )
    .join("");
  return `<details class="expander" style="margin-top:var(--sp-4)">
    <summary>Promotion history (${promotions.length})</summary>
    <div class="body"><ul style="margin:0; padding-left:var(--sp-5);
      font-size:var(--fs-sm)">${items}</ul></div>
  </details>`;
}

function familyHtml(fam, index, mode) {
  const best = fam.best;
  const content =
    typeof best.content === "string"
      ? best.content
      : JSON.stringify(best.content, null, 2);
  const radios = DIFF_MODES.map(
    (m) => `<label style="margin-right:var(--sp-4); font-size:var(--fs-sm)">
      <input type="radio" class="ch-diffmode" name="ch-diffmode-${index}"
        value="${m.value}"${m.value === mode ? " checked" : ""}>
      ${escapeHtml(m.label)}</label>`
  ).join("");

  return `<section class="family" data-index="${index}">
    <h3 style="font-size:var(--fs-lg)">👑 Champion: ${escapeHtml(fam.family.label)}</h3>
    <div id="ch-tiles-${index}"></div>
    <p class="panel-caption" style="margin:var(--sp-3) 0 var(--sp-4)">
      ${escapeHtml(best.id)} · ${escapeHtml(best.kind)}
      · hash ${escapeHtml(String(best.content_hash ?? "").slice(0, 12))}</p>
    ${promotionHtml(fam, index)}
    ${lineageHtml(fam.lineage_path ?? [])}
    <section class="block">
      <h3>Champion content${best.content_truncated ? " (truncated)" : ""}</h3>
      <div class="card gold"><pre class="content">${escapeHtml(content)}</pre></div>
    </section>
    <section class="block">
      <h3>What changed</h3>
      <div style="margin-bottom:var(--sp-3)">${radios}</div>
      <div id="ch-diff-${index}"><div class="placeholder">Loading diff…</div></div>
    </section>
    ${winsHtml(fam.top_wins ?? [])}
    <section class="block">
      <h3>Provenance</h3>
      <div id="ch-prov-${index}"></div>
    </section>
    ${promotionsHtml(fam.promotions ?? [])}
  </section>`;
}

// Genesis mode anchors on the first version of the walk, parent mode on the one
// immediately before the champion; either can be absent, which is not an error.
async function renderDiff(host, ctx, fam, mode) {
  const path = fam.lineage_path ?? [];
  const best = fam.best;
  const aVersion =
    mode === "parent"
      ? path.length >= 2
        ? path[path.length - 2].version
        : null
      : path.length
        ? path[0].version
        : 1;
  if (aVersion === null || aVersion === best.version) {
    host.innerHTML = `<div class="banner info">${
      mode === "parent"
        ? "This champion has no recorded parent, so there is nothing to diff."
        : "This champion is the genesis artifact, so there is nothing to diff."
    }</div>`;
    return;
  }
  try {
    const payload = await api.get(
      `/api/archives/${ctx.archiveKey}/diff`,
      {
        a_id: best.id,
        a_version: aVersion,
        b_id: best.id,
        b_version: best.version,
      },
      { signal: ctx.signal }
    );
    host.innerHTML = "";
    host.appendChild(diffView(payload));
  } catch (err) {
    if (err.name === "AbortError") return;
    host.innerHTML = `<div class="banner error">${escapeHtml(
      err.message ?? String(err)
    )}</div>`;
  }
}

async function promote(container, ctx, fam) {
  const best = fam.best;
  const score = fmtScore(best.measurement?.score);
  const ok = window.confirm(
    `Promote v${best.version} of ${best.id} to live? This writes a promotion ` +
      `record into the archive.`
  );
  if (!ok) return;
  try {
    await api.postJSON(`/api/archives/${ctx.archiveKey}/promote`, {
      artifact_id: best.id,
      version: best.version,
      approver: "dashboard",
      reason: `manual promotion via dashboard (score ${score})`,
    });
    toast(`Promoted ${best.id} v${best.version} to live.`, "success");
    await render(container, ctx);
  } catch (err) {
    toast(err.message ?? String(err), "error");
  }
}

export async function render(container, ctx) {
  if (!ctx.archiveKey) {
    container.innerHTML = `<div class="placeholder">Pick an archive in the header to begin.</div>`;
    return;
  }
  const body = await api.get(
    `/api/archives/${ctx.archiveKey}/champions`,
    focusParams(ctx),
    { signal: ctx.signal }
  );
  const families = body.families ?? [];
  const mode = ctx.params.get("diffmode") === "parent" ? "parent" : "genesis";

  const head = `
    <h2 class="panel-title">Champion</h2>
    <p class="panel-caption">The crown marks the top measured score; the live
      version is whatever was promoted. When those disagree, a winner is
      waiting on review.</p>`;

  if (!families.length) {
    container.innerHTML = `${head}
      <div class="banner info">No measured artifacts yet, so no champion to declare.</div>`;
    return;
  }

  container.innerHTML =
    head +
    (families.length > 1
      ? `<p class="panel-caption">${families.length} artifact families have champions</p>`
      : "") +
    families.map((fam, i) => familyHtml(fam, i, mode)).join("<hr style='margin:var(--sp-6) 0'>");

  families.forEach((fam, i) => {
    const best = fam.best;
    container.querySelector(`#ch-tiles-${i}`).appendChild(
      tiles([
        { label: "Best candidate", value: `v${best.version}`, accent: "teal" },
        { label: "Score", value: fmtScore(best.measurement?.score), accent: "teal" },
        { label: "Created by", value: best.created_by },
        { label: "Search method", value: best.search_method },
      ])
    );
    container.querySelector(`#ch-prov-${i}`).appendChild(
      tiles([
        { label: "Family size", value: `${fam.family_size} versions` },
        { label: "Champion measurements", value: fmtInt(fam.n_measurements) },
        { label: "Family tokens", value: fmtInt(fam.family_tokens) },
        { label: "Champion confidence", value: fmtScore(fam.confidence, 2) },
      ])
    );
    renderDiff(container.querySelector(`#ch-diff-${i}`), ctx, fam, mode).catch(() => {});
  });

  // Selections live in the URL so a pending-promotion view is a shareable link.
  container.querySelectorAll("input.ch-diffmode").forEach((input) => {
    input.addEventListener("change", () =>
      ctx.navigate(null, { diffmode: input.value })
    );
  });
  container.querySelectorAll("button[data-promote]").forEach((btn) => {
    btn.addEventListener("click", () =>
      promote(container, ctx, families[Number(btn.dataset.promote)])
    );
  });
}
