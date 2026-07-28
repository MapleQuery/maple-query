/**
 * package_id → dataset title, so anywhere the UI would print a UUID it
 * can print a name instead.
 *
 * Titles arrive free on every `datasets_ranked` frame, so a turn fills
 * the cache for its own chips before they render. Ids seen without a
 * title — a scope pinned from a suggestion, a notebook saved before
 * titles were recorded — are backfilled one `GET /datasets/{id}` at a
 * time, kept in localStorage, and never re-fetched: the title of a
 * package does not change between runs.
 *
 * A miss is never fatal. The caller falls back to the id, which is what
 * it used to show anyway.
 */

import * as React from "react";
import { getDataset } from "./api";

export type DatasetTitleMap = Readonly<Record<string, string>>;

const KEY = "mq:dataset-titles:v1";
const EMPTY: DatasetTitleMap = Object.freeze({});
/** Ids are UUIDs, so any non-hex character separates them safely. */
const SEP = "|";

let cache: DatasetTitleMap = EMPTY;
let hydrated = false;
const listeners = new Set<() => void>();
const inflight = new Set<string>();
/** Ids the API had no title for. In-memory only, so a transient failure
 * is retried on the next page load but not on every re-render. */
const unresolved = new Set<string>();

function emit(): void {
  for (const l of listeners) l();
}

function persist(): void {
  try {
    window.localStorage.setItem(KEY, JSON.stringify(cache));
  } catch {
    // Quota, or a browser with storage locked down. The in-memory map
    // still serves this session; nothing here is worth a failed render.
  }
}

/** Read the stored map once, after mount. Deferred rather than done at
 * module load so the first client render matches the server's. */
function hydrate(): void {
  if (hydrated) return;
  hydrated = true;
  if (typeof window === "undefined") return;
  try {
    const raw = window.localStorage.getItem(KEY);
    if (!raw) return;
    const parsed: unknown = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") return;
    const next: Record<string, string> = { ...cache };
    let changed = false;
    for (const [id, title] of Object.entries(parsed as Record<string, unknown>)) {
      if (typeof title === "string" && title && next[id] !== title) {
        next[id] = title;
        changed = true;
      }
    }
    if (changed) {
      cache = next;
      emit();
    }
  } catch {
    // Corrupt payload — start from empty rather than trip the page.
  }
}

/** Record titles seen in a stream frame or a REST response. */
export function rememberDatasetTitles(
  entries: { package_id: string; title?: string | null }[],
): void {
  const next: Record<string, string> = { ...cache };
  let changed = false;
  for (const e of entries) {
    const title = e.title?.trim();
    if (!title || !e.package_id || next[e.package_id] === title) continue;
    next[e.package_id] = title;
    unresolved.delete(e.package_id);
    changed = true;
  }
  if (!changed) return;
  cache = next;
  persist();
  emit();
}

/** Snapshot for callers outside a render, such as export builders. */
export function getCachedDatasetTitles(): DatasetTitleMap {
  hydrate();
  return cache;
}

function fetchTitle(packageId: string): void {
  if (!packageId || cache[packageId] || unresolved.has(packageId)) return;
  if (inflight.has(packageId)) return;
  inflight.add(packageId);
  void getDataset(packageId)
    .then((d) => {
      const title = d.title?.trim();
      if (title) rememberDatasetTitles([{ package_id: packageId, title }]);
      else unresolved.add(packageId);
    })
    .catch(() => {
      unresolved.add(packageId);
    })
    .finally(() => {
      inflight.delete(packageId);
    });
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

function getSnapshot(): DatasetTitleMap {
  return cache;
}

function getServerSnapshot(): DatasetTitleMap {
  return EMPTY;
}

/**
 * Titles for `packageIds`, backfilling any that are still unknown.
 * Re-renders when a backfill lands.
 *
 * `known` is titles the caller already holds — a notebook block keeps
 * its own copy — and wins over the cache. Anything it covers is never
 * looked up, so reopening a saved notebook costs no requests.
 */
export function useDatasetTitles(
  packageIds: string[],
  known?: Record<string, string>,
): DatasetTitleMap {
  const cached = React.useSyncExternalStore(
    subscribe,
    getSnapshot,
    getServerSnapshot,
  );
  // Ids arrive as a fresh array each render; key on the contents so the
  // effect fires when the set changes, not when the array identity does.
  const key = packageIds
    .filter((id) => !known?.[id] && !cached[id])
    .join(SEP);

  React.useEffect(() => {
    // Hydrating before the loop means an id the stored map already
    // answers never reaches fetchTitle, even on the first render.
    hydrate();
    for (const id of key ? key.split(SEP) : []) fetchTitle(id);
  }, [key]);

  return known ? { ...cached, ...known } : cached;
}
