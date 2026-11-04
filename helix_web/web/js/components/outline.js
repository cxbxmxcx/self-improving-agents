// Outline panels: the planned layout of a panel whose data does not exist
// yet. Each block is drawn as a labeled frame so the shape can be reviewed
// (and the figure composition planned) before the chapter that fills it.

import { escapeHtml } from "../format.js";

// spec = { title, caption, chapter, route, endpoints: [..], events: [..],
//          regions: [{ name, note, grow? }], figures: [..] }
export function renderOutline(container, spec) {
  const regions = spec.regions
    .map(
      (r) => `
      <div class="outline-region" style="${r.grow ? "grid-column: 1 / -1;" : ""}">
        <div class="outline-region-name">${escapeHtml(r.name)}</div>
        <div class="outline-region-note">${escapeHtml(r.note)}</div>
      </div>`
    )
    .join("");
  const list = (items) =>
    items.map((x) => `<li><code>${escapeHtml(x)}</code></li>`).join("");
  container.innerHTML = `
    <h2 class="panel-title">${escapeHtml(spec.title)}</h2>
    <p class="panel-caption">${escapeHtml(spec.caption)}</p>
    <div class="banner info" style="margin-bottom:var(--sp-3)">
      Outline only. This panel fills in with chapter ${escapeHtml(String(spec.chapter))};
      the frames below are its planned layout.
    </div>
    <div class="card plate outline-grid">${regions}</div>
    <details class="expander" style="margin-top:var(--sp-3)">
      <summary>Planned data and events</summary>
      <div class="body">
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:var(--sp-4)">
          <div><b>Route</b><ul><li><code>${escapeHtml(spec.route)}</code></li></ul></div>
          <div><b>Endpoints</b><ul>${list(spec.endpoints)}</ul></div>
          <div><b>Bus events</b><ul>${list(spec.events)}</ul></div>
          <div><b>Figures shot here</b><ul>${spec.figures
            .map((f) => `<li>${escapeHtml(f)}</li>`)
            .join("")}</ul></div>
        </div>
      </div>
    </details>`;
}
