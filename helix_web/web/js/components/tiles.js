// Metric tile row: [{label, value, sub, accent, muted}] -> element.

import { escapeHtml } from "../format.js";

export function tiles(items) {
  const wrap = document.createElement("div");
  wrap.className = "tiles";
  for (const item of items) {
    const tile = document.createElement("div");
    tile.className = "tile" + (item.accent ? ` accent-${item.accent}` : "");
    tile.innerHTML = `
      <div class="label">${escapeHtml(item.label)}</div>
      <div class="value${item.muted ? " muted" : ""}">${escapeHtml(item.value)}</div>
      ${item.sub ? `<div class="sub">${escapeHtml(item.sub)}</div>` : ""}
    `;
    wrap.appendChild(tile);
  }
  return wrap;
}
