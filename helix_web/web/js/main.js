// Boot: load the archive list, wire the header pickers, start the router.

import * as api from "./api.js";
import { cache, currentArchive, currentFocus, navigate, onRouteChange } from "./state.js";
import { renderRoute } from "./router.js";
import { toast } from "./components/toast.js";
import { ensureStream } from "./stream.js";

async function loadArchives() {
  const body = await api.get("/api/archives");
  cache.archives = body.archives.filter((a) => a.exists);
  const select = document.getElementById("archive-select");
  select.innerHTML = cache.archives
    .map((a) => {
      const counts = a.valid ? ` (${a.n_artifacts} artifacts)` : " (not an archive)";
      const label = a.rel_path ?? a.path;
      return `<option value="${a.key}">${label}${counts}</option>`;
    })
    .join("");
  if (!currentArchive() && cache.archives.length) {
    const preferred =
      cache.archives.find((a) => a.valid && (a.n_measured ?? 0) > 0) ??
      cache.archives[0];
    navigate(null, { archive: preferred.key });
  }
  select.value = currentArchive();
}

async function loadFocusOptions() {
  const key = currentArchive();
  const select = document.getElementById("focus-select");
  if (!key) {
    select.innerHTML = `<option value="">(all artifacts)</option>`;
    return;
  }
  if (!cache.focusOptions.has(key)) {
    try {
      const body = await api.get(`/api/archives/${key}/variants`);
      const seen = new Map();
      for (const v of body.items) {
        seen.set(`${v.id}::${v.kind}`, { id: v.id, kind: v.kind });
      }
      cache.focusOptions.set(key, [...seen.values()]);
    } catch {
      cache.focusOptions.set(key, []);
    }
  }
  const options = cache.focusOptions.get(key);
  const focus = currentFocus();
  const current = focus ? `${focus.id}::${focus.kind ?? ""}` : "";
  select.innerHTML =
    `<option value="">(all artifacts)</option>` +
    options
      .map((o) => `<option value="${o.id}::${o.kind}">${o.id} (${o.kind})</option>`)
      .join("");
  select.value = [...select.options].some((opt) => opt.value === current)
    ? current
    : "";
}

function wireHeader() {
  document.getElementById("archive-select").addEventListener("change", (e) => {
    navigate(null, { archive: e.target.value, focus: null, sel: null });
  });
  document.getElementById("focus-select").addEventListener("change", (e) => {
    navigate(null, { focus: e.target.value || null, sel: null });
  });
}

async function boot() {
  wireHeader();
  try {
    await loadArchives();
  } catch (err) {
    toast(`Could not load archives: ${err.message}`, "error");
  }
  await loadFocusOptions();
  ensureStream();
  await renderRoute();
  onRouteChange(async () => {
    document.getElementById("archive-select").value = currentArchive();
    await loadFocusOptions();
    ensureStream();
    await renderRoute();
  });
}

boot();
