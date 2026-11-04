// Hash router: resolves the route to a panel module and renders it.

import { abortInFlight } from "./api.js";
import { parseHash, currentArchive, currentFocus, navigate } from "./state.js";
import { escapeHtml } from "./format.js";

const PANELS = {
  overview: () => import("./panels/overview.js"),
  champion: () => import("./panels/champion.js"),
  lineage: () => import("./panels/lineage.js"),
  compare: () => import("./panels/compare.js"),
  verdicts: () => import("./panels/verdicts.js"),
  replay: () => import("./panels/replay.js"),
  rounds: () => import("./panels/rounds.js"),
  "lineage-compare": () => import("./panels/lineage_compare.js"),
  // Outline panels: planned layouts for chapters 5 to 7, data not yet wired.
  memory: () => import("./panels/memory.js"),
  scaffold: () => import("./panels/scaffold.js"),
  proposals: () => import("./panels/proposals.js"),
};

function markActiveNav(route) {
  document.querySelectorAll("nav.sidebar a").forEach((a) => {
    a.classList.toggle("active", a.dataset.route === route);
  });
  // Keep nav links carrying the current query so panel switches keep context.
  const { params } = parseHash();
  const qs = params.toString();
  document.querySelectorAll("nav.sidebar a").forEach((a) => {
    a.href = `#/${a.dataset.route}${qs ? "?" + qs : ""}`;
  });
}

export async function renderRoute() {
  const { route, params } = parseHash();
  markActiveNav(route);
  // Expanded mode is opt-in per render; panels that honor expand=1 re-add it.
  document.body.classList.remove("expanded");
  const container = document.getElementById("panel");
  const loader = PANELS[route];
  if (!loader) {
    container.innerHTML = `<div class="placeholder">Unknown panel: ${escapeHtml(route)}</div>`;
    return;
  }
  const signal = abortInFlight();
  container.innerHTML = `<div class="placeholder">Loading…</div>`;
  try {
    const mod = await loader();
    const ctx = {
      archiveKey: currentArchive(),
      focus: currentFocus(),
      params,
      navigate,
      signal,
    };
    await mod.render(container, ctx);
  } catch (err) {
    if (err.name === "AbortError") return;
    container.innerHTML = `<div class="banner error">${escapeHtml(err.message ?? String(err))}</div>`;
  }
}
