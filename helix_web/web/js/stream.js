// One EventSource per tab, following the archive in the URL.
// Panels subscribe to typed events; the header pill reflects the connection.

import { currentArchive } from "./state.js";

let source = null;
let streamedKey = null;
const listeners = new Map(); // event type -> Set<fn>

function pill(state, text) {
  const el = document.getElementById("live-pill");
  el.className = `live-pill ${state}`;
  el.textContent = text;
}

function dispatch(type, data) {
  for (const fn of listeners.get(type) ?? []) fn(data);
  for (const fn of listeners.get("*") ?? []) fn(type, data);
}

export function onStreamEvent(type, fn) {
  if (!listeners.has(type)) listeners.set(type, new Set());
  listeners.get(type).add(fn);
  return () => listeners.get(type)?.delete(fn);
}

const EVENT_TYPES = [
  "hello",
  "artifact",
  "measurement",
  "trajectory",
  "promotion",
  "champions",
  "round",
  "resync",
];

export function ensureStream() {
  // Screenshot and automation runs pass ?nostream=1 (before the hash) so the
  // page can reach network-idle; everything else behaves identically.
  if (new URLSearchParams(window.location.search).has("nostream")) {
    pill("", "static");
    return;
  }
  const key = currentArchive();
  if (!key) {
    stopStream();
    pill("", "static");
    return;
  }
  if (source && streamedKey === key) return;
  stopStream();
  streamedKey = key;
  pill("", "connecting…");
  source = new EventSource(`/api/archives/${key}/events`);
  source.onopen = () => pill("live", "● live");
  source.onerror = () => {
    // EventSource retries on its own; reflect the gap honestly meanwhile.
    pill("offline", "offline");
  };
  for (const type of EVENT_TYPES) {
    source.addEventListener(type, (e) => {
      if (type === "hello") pill("live", "● live");
      let data = {};
      try {
        data = JSON.parse(e.data);
      } catch {
        /* ignore malformed frames */
      }
      dispatch(type, data);
    });
  }
}

export function stopStream() {
  if (source) source.close();
  source = null;
  streamedKey = null;
}
