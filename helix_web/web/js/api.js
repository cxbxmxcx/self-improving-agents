// Fetch wrapper for the Observatory API: JSON, errors, and route-scoped abort.

export class ApiError extends Error {
  constructor(status, detail) {
    super(`API ${status}: ${detail}`);
    this.status = status;
    this.detail = detail;
  }
}

let currentController = null;

// Abort in-flight panel fetches when the route changes.
export function abortInFlight() {
  if (currentController) currentController.abort();
  currentController = new AbortController();
  return currentController.signal;
}

function signalOr(sig) {
  if (sig) return sig;
  if (!currentController) currentController = new AbortController();
  return currentController.signal;
}

async function parseError(resp) {
  let detail = resp.statusText;
  try {
    const body = await resp.json();
    if (body && body.detail) detail = body.detail;
  } catch { /* keep statusText */ }
  return new ApiError(resp.status, detail);
}

export async function get(path, params = {}, { signal } = {}) {
  const url = new URL(path, window.location.origin);
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== "") url.searchParams.set(k, v);
  }
  let resp = await fetch(url, { signal: signalOr(signal) });
  if (resp.status === 503) {
    // The archive was mid-write; one quiet retry before surfacing it.
    await new Promise((r) => setTimeout(r, 1100));
    resp = await fetch(url, { signal: signalOr(signal) });
  }
  if (!resp.ok) throw await parseError(resp);
  return resp.json();
}

export async function postJSON(path, body) {
  const resp = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });
  if (!resp.ok) throw await parseError(resp);
  if (resp.status === 204) return null;
  return resp.json();
}

export async function del(path) {
  const resp = await fetch(path, { method: "DELETE" });
  if (!resp.ok) throw await parseError(resp);
  return null;
}
