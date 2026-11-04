// Transient notifications in the corner; errors linger a little longer.

export function toast(message, kind = "info", ms = 3200) {
  const host = document.getElementById("toasts");
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.textContent = message;
  host.appendChild(el);
  setTimeout(() => el.remove(), kind === "error" ? ms * 2 : ms);
}
