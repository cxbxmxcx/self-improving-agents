// App state: the URL hash is the single source of truth.
// #/panel?archive=<key>&focus=<id::kind>&sel=<ref>&... — every view is a link.

const subscribers = new Set();

export function parseHash() {
  const hash = window.location.hash.replace(/^#\/?/, "");
  const [route, queryString] = hash.split("?");
  const params = new URLSearchParams(queryString ?? "");
  return { route: route || "overview", params };
}

export function currentArchive() {
  return parseHash().params.get("archive") ?? "";
}

export function currentFocus() {
  const raw = parseHash().params.get("focus");
  if (!raw) return null;
  const [id, kind] = raw.split("::");
  return { id, kind: kind || null };
}

// Merge query-param changes into the hash without losing the rest.
export function navigate(route, patch = {}) {
  const { route: cur, params } = parseHash();
  const target = route ?? cur;
  for (const [k, v] of Object.entries(patch)) {
    if (v === null || v === undefined || v === "") params.delete(k);
    else params.set(k, String(v));
  }
  const qs = params.toString();
  window.location.hash = `#/${target}${qs ? "?" + qs : ""}`;
}

export function onRouteChange(fn) {
  subscribers.add(fn);
  return () => subscribers.delete(fn);
}

window.addEventListener("hashchange", () => {
  const state = parseHash();
  for (const fn of subscribers) fn(state);
});

// Session cache for cross-panel data (archive list, focus options).
export const cache = {
  archives: [],
  focusOptions: new Map(), // archiveKey -> [{id, kind, label}]
};
