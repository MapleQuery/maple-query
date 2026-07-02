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

export type { DatasetSummary };
