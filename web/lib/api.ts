import { API_BASE_URL, authHeaders } from "./config";
import type {
  ColumnsResponse,
  DatasetSummary,
  DatasetsResponse,
  SqlRunResponse,
} from "./types";

export class ApiError extends Error {
  constructor(
    public status: number,
    public body: string,
    message?: string,
  ) {
    super(message ?? `API ${status}`);
  }
}

async function jsonFetch<T>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...authHeaders(),
      ...(init.headers ?? {}),
    },
  });
  const text = await res.text();
  if (!res.ok) throw new ApiError(res.status, text);
  return text ? (JSON.parse(text) as T) : (undefined as T);
}

export function listDatasets(params: {
  q?: string;
  limit?: number;
  offset?: number;
  signal?: AbortSignal;
}): Promise<DatasetsResponse> {
  const qs = new URLSearchParams();
  if (params.q) qs.set("q", params.q);
  if (params.limit != null) qs.set("limit", String(params.limit));
  if (params.offset != null) qs.set("offset", String(params.offset));
  const suffix = qs.toString() ? `?${qs.toString()}` : "";
  return jsonFetch<DatasetsResponse>(`/datasets${suffix}`, {
    signal: params.signal,
  });
}

export function getDatasetColumns(
  packageId: string,
  signal?: AbortSignal,
): Promise<ColumnsResponse> {
  return jsonFetch<ColumnsResponse>(
    `/datasets/${encodeURIComponent(packageId)}/columns`,
    { signal },
  );
}

export function runSql(
  sql: string,
  rationale?: string,
  signal?: AbortSignal,
): Promise<SqlRunResponse> {
  return jsonFetch<SqlRunResponse>(`/sql/run`, {
    method: "POST",
    body: JSON.stringify({ sql, rationale }),
    signal,
  });
}

export interface CorpusStats {
  documents: number;
  rows: number;
}

/**
 * Live counts for the hero. `raw.documents.row_count` is the per-doc row
 * count populated by the warehouse loader — summing it is the honest
 * substitute for `COUNT(*) FROM raw.rows`, which the SQL guard rejects
 * (raw.rows requires a document_id IN-list to keep cost bounded).
 */
export async function getCorpusStats(
  signal?: AbortSignal,
): Promise<CorpusStats> {
  const res = await runSql(
    "SELECT COUNT(*) AS documents, IFNULL(SUM(row_count), 0) AS rows FROM raw.documents",
    "landing page hero stats",
    signal,
  );
  const first = res.rows[0] ?? {};
  return {
    documents: toNumber(first["documents"]),
    rows: toNumber(first["rows"]),
  };
}

function toNumber(v: unknown): number {
  if (typeof v === "number") return v;
  if (typeof v === "string") {
    const n = Number(v);
    return Number.isFinite(n) ? n : 0;
  }
  return 0;
}

export type { DatasetSummary };
