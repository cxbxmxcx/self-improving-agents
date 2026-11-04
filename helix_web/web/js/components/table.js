// Declarative data table with client-side sorting.
// columns: [{key, label, format?, align?, sortable?}] ; rows: array of objects.

import { escapeHtml } from "../format.js";

export function dataTable({ columns, rows, empty = "No rows.", onRowClick }) {
  const wrap = document.createElement("div");
  if (!rows.length) {
    wrap.innerHTML = `<div class="placeholder">${escapeHtml(empty)}</div>`;
    return wrap;
  }

  let sortKey = null;
  let sortDir = 1;

  const table = document.createElement("table");
  table.className = "data";

  function render() {
    const sorted = [...rows];
    if (sortKey) {
      sorted.sort((a, b) => {
        const av = a[sortKey], bv = b[sortKey];
        if (av === bv) return 0;
        if (av === null || av === undefined) return 1;
        if (bv === null || bv === undefined) return -1;
        return (av > bv ? 1 : -1) * sortDir;
      });
    }
    const head = columns
      .map(
        (c) =>
          `<th class="${c.align === "right" ? "num" : ""}" data-key="${c.key}">` +
          `${escapeHtml(c.label)}${sortKey === c.key ? (sortDir > 0 ? " ▲" : " ▼") : ""}</th>`
      )
      .join("");
    const body = sorted
      .map((row, i) => {
        const cells = columns
          .map((c) => {
            const raw = row[c.key];
            const text = c.format ? c.format(raw, row) : raw ?? "—";
            return `<td class="${c.align === "right" ? "num" : ""}">${
              c.html ? text : escapeHtml(text)
            }</td>`;
          })
          .join("");
        return `<tr data-index="${i}" ${onRowClick ? 'style="cursor:pointer"' : ""}>${cells}</tr>`;
      })
      .join("");
    table.innerHTML = `<thead><tr>${head}</tr></thead><tbody>${body}</tbody>`;

    table.querySelectorAll("th").forEach((th) => {
      th.style.cursor = "pointer";
      th.addEventListener("click", () => {
        const key = th.dataset.key;
        if (sortKey === key) sortDir = -sortDir;
        else { sortKey = key; sortDir = 1; }
        render();
      });
    });
    if (onRowClick) {
      table.querySelectorAll("tbody tr").forEach((tr) => {
        tr.addEventListener("click", () => {
          const sorted2 = [...rows];
          if (sortKey) {
            sorted2.sort((a, b) => {
              const av = a[sortKey], bv = b[sortKey];
              if (av === bv) return 0;
              if (av === null || av === undefined) return 1;
              if (bv === null || bv === undefined) return -1;
              return (av > bv ? 1 : -1) * sortDir;
            });
          }
          onRowClick(sorted2[Number(tr.dataset.index)]);
        });
      });
    }
  }

  render();
  wrap.appendChild(table);
  return wrap;
}
