// Renders the server's opcode diff payload as a side-by-side table.

import { escapeHtml } from "../format.js";

export function diffView(payload) {
  const wrap = document.createElement("div");
  const { a, b, opcodes, stats, truncated } = payload;

  const rows = [];
  for (const op of opcodes) {
    const aLines = a.lines.slice(op.a1, op.a2);
    const bLines = b.lines.slice(op.b1, op.b2);
    const n = Math.max(aLines.length, bLines.length);
    for (let i = 0; i < n; i++) {
      const left = aLines[i];
      const right = bLines[i];
      let leftCls = "", rightCls = "";
      if (op.tag === "replace") { leftCls = "del"; rightCls = "add"; }
      else if (op.tag === "delete") leftCls = "del";
      else if (op.tag === "insert") rightCls = "add";
      rows.push(
        `<tr>
          <td class="ln">${left !== undefined ? op.a1 + i + 1 : ""}</td>
          <td class="code ${leftCls}">${left !== undefined ? escapeHtml(left) : ""}</td>
          <td class="ln">${right !== undefined ? op.b1 + i + 1 : ""}</td>
          <td class="code ${rightCls}">${right !== undefined ? escapeHtml(right) : ""}</td>
        </tr>`
      );
    }
  }

  wrap.innerHTML = `
    <style>
      .diff-table { width:100%; border-collapse:collapse; background:var(--bg-panel);
        border:1px solid var(--border); font-family:var(--font-mono); font-size:var(--fs-sm); }
      .diff-table th { background:var(--npg-slate); color:#fff; padding:4px 8px;
        font-family:var(--font-sans); font-size:var(--fs-xs); text-align:left; }
      .diff-table td { vertical-align:top; padding:1px 8px; white-space:pre-wrap; word-break:break-word; }
      .diff-table td.ln { color:var(--fg-faint); text-align:right; user-select:none; width:1%; }
      .diff-table td.code { width:49%; }
      .diff-table td.add { background:var(--diff-add-bg); }
      .diff-table td.del { background:var(--diff-del-bg); }
    </style>
    <p class="panel-caption" style="margin-bottom:var(--sp-2)">
      +${stats.added} added · −${stats.removed} removed · ${stats.unchanged} unchanged
      ${truncated ? " · diff truncated" : ""}</p>
    <div style="max-height:480px; overflow:auto">
      <table class="diff-table">
        <thead><tr><th></th><th>${escapeHtml(a.label)}</th><th></th><th>${escapeHtml(b.label)}</th></tr></thead>
        <tbody>${rows.join("")}</tbody>
      </table>
    </div>`;
  return wrap;
}
