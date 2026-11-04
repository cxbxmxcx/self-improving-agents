// Formatting helpers shared by every panel.

export function fmtScore(score, digits = 3) {
  return score === null || score === undefined ? "—" : Number(score).toFixed(digits);
}

export function fmtInt(n) {
  return n === null || n === undefined ? "—" : Number(n).toLocaleString("en-US");
}

export function fmtWhen(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? String(iso) : d.toLocaleString();
}

export function truncate(text, max = 120) {
  const s = String(text ?? "");
  return s.length > max ? s.slice(0, max - 1) + "…" : s;
}

export function escapeHtml(text) {
  return String(text ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

// Creator color rules, ported exactly from the Streamlit lineage panel.
const CREATOR_COLORS = {
  human: "var(--npg-slate)",
  gepa_crossover: "var(--npg-salmon)",
};

export function creatorColor(createdBy) {
  const c = String(createdBy ?? "");
  if (c in CREATOR_COLORS) return CREATOR_COLORS[c];
  if (c.startsWith("spo")) return "var(--npg-teal)";
  if (c.startsWith("gepa")) return "var(--npg-vermillion)";
  if (c.startsWith("dgm")) return "var(--dgm-brown)";
  if (c.startsWith("hillclimb")) return "var(--npg-cyan)";
  return "var(--c-unknown)";
}

// Raw hex versions for SVG export, where CSS variables are not available.
const CREATOR_HEX = {
  human: "#3C5488",
  gepa_crossover: "#F39B7F",
};

export function creatorHex(createdBy) {
  const c = String(createdBy ?? "");
  if (c in CREATOR_HEX) return CREATOR_HEX[c];
  if (c.startsWith("spo")) return "#00A087";
  if (c.startsWith("gepa")) return "#E64B35";
  if (c.startsWith("dgm")) return "#7E6148";
  if (c.startsWith("hillclimb")) return "#4DBBD5";
  return "#888888";
}
