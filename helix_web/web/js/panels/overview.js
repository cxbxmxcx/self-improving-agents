// Overview panel: archive tiles, breakdowns, champion card, failure histogram.

import * as api from "../api.js";
import { tiles } from "../components/tiles.js";
import { dataTable } from "../components/table.js";
import { escapeHtml, fmtInt, fmtScore, truncate } from "../format.js";

export async function render(container, ctx) {
  if (!ctx.archiveKey) {
    container.innerHTML = `<div class="placeholder">Pick an archive in the header to begin.</div>`;
    return;
  }
  const params = {};
  if (ctx.focus) {
    params.focus_id = ctx.focus.id;
    if (ctx.focus.kind) params.focus_kind = ctx.focus.kind;
  }
  const body = await api.get(`/api/archives/${ctx.archiveKey}/summary`, params, {
    signal: ctx.signal,
  });
  const t = body.totals;

  container.innerHTML = `
    <h2 class="panel-title">Overview</h2>
    <p class="panel-caption">The whole archive at a glance; the champion and
      by-method table honor the focus picker, the totals do not.</p>
    <section class="block" id="ov-tiles"></section>
    <section class="block"><h3>Artifacts by kind</h3><div id="ov-kinds"></div></section>
    <section class="block"><h3 id="ov-champ-title">Champion</h3><div id="ov-champ"></div></section>
    <section class="block"><h3>Artifacts by search creator</h3><div id="ov-creators"></div></section>
    <section class="block"><h3>Trajectory cache</h3><div id="ov-traj"></div></section>
    <section class="block" id="ov-failed-block" hidden>
      <h3>Failed runs by error type</h3><div id="ov-failed"></div></section>
    <section class="block" id="ov-rounds-block" hidden>
      <h3>Recent rounds</h3><div id="ov-rounds"></div></section>
  `;

  container.querySelector("#ov-tiles").appendChild(
    tiles([
      { label: "Total artifacts", value: fmtInt(t.artifacts), accent: "slate" },
      { label: "With measurements", value: fmtInt(t.measured) },
      {
        label: "Top score",
        value: fmtScore(t.top_score),
        accent: t.top_score !== null ? "teal" : undefined,
        muted: t.top_score === null,
      },
      { label: "Rounds run", value: fmtInt(t.rounds) },
      {
        label: "Trajectories",
        value: fmtInt(t.trajectories),
        sub: t.failed ? `${t.failed} failed / ${t.purgeable} purgeable` : undefined,
      },
    ])
  );

  container.querySelector("#ov-kinds").appendChild(
    dataTable({
      columns: [
        { key: "kind", label: "Kind" },
        { key: "id", label: "Artifact id" },
        { key: "versions", label: "Versions", align: "right" },
      ],
      rows: body.by_kind,
      empty: "No artifacts in this archive.",
    })
  );

  const championHost = container.querySelector("#ov-champ");
  if (body.champion) {
    const c = body.champion;
    container.querySelector("#ov-champ-title").textContent = ctx.focus
      ? `Champion for ${ctx.focus.id}`
      : "Overall champion";
    championHost.innerHTML = `
      <div class="card gold">
        <div class="tiles">
          <div class="tile"><div class="label">Version</div><div class="value">v${c.version}</div></div>
          <div class="tile"><div class="label">Score</div><div class="value accent">${fmtScore(c.measurement?.score)}</div></div>
          <div class="tile"><div class="label">Created by</div><div class="value" style="font-size:var(--fs-md)">${escapeHtml(c.created_by)}</div></div>
          <div class="tile"><div class="label">Kind</div><div class="value" style="font-size:var(--fs-md)">${escapeHtml(c.kind)}</div></div>
        </div>
        <p class="panel-caption" style="margin:var(--sp-3) 0 0">
          ${escapeHtml(c.id)} · hash ${escapeHtml(truncate(c.content_hash ?? "", 12))}
          · <a href="#/lineage?archive=${ctx.archiveKey}&sel=${c.ref}">view in lineage</a>
        </p>
      </div>`;
  } else {
    championHost.innerHTML = `<div class="banner info">No measured artifacts${
      ctx.focus ? " in this focus" : ""
    }, so no champion yet. The chapter-4 archives predate score recording; re-run the loops to fill them.</div>`;
  }

  container.querySelector("#ov-creators").appendChild(
    dataTable({
      columns: [
        { key: "created_by", label: "Created by" },
        { key: "count", label: "Count", align: "right" },
        {
          key: "mean_score",
          label: "Mean score",
          align: "right",
          format: (v) => fmtScore(v),
        },
        {
          key: "max_score",
          label: "Max score",
          align: "right",
          format: (v) => fmtScore(v),
        },
      ],
      rows: body.by_creator,
      empty: "No artifacts match the focus.",
    })
  );

  container.querySelector("#ov-traj").appendChild(
    tiles([
      { label: "Cached trajectories", value: fmtInt(t.trajectories) },
      {
        label: "Failed",
        value: fmtInt(t.failed),
        accent: t.failed ? "vermillion" : undefined,
        sub: "see Replay to inspect or purge",
      },
    ])
  );

  if (body.failed_by_type.length) {
    const block = container.querySelector("#ov-failed-block");
    block.hidden = false;
    block.querySelector("#ov-failed").appendChild(
      dataTable({
        columns: [
          { key: "error_type", label: "Error type" },
          { key: "count", label: "Count", align: "right" },
          { key: "likely_cause", label: "Likely cause" },
        ],
        rows: body.failed_by_type,
      })
    );
  }

  if (body.recent_rounds.length) {
    const block = container.querySelector("#ov-rounds-block");
    block.hidden = false;
    const cols = Object.keys(body.recent_rounds[0]).slice(0, 8);
    block.querySelector("#ov-rounds").appendChild(
      dataTable({
        columns: cols.map((k) => ({ key: k, label: k, align: "right" })),
        rows: body.recent_rounds,
      })
    );
  }
}
