const trimSlash = (s: string) => s.replace(/\/+$/, "");

const rawBase =
  process.env.NEXT_PUBLIC_MAPLEQUERY_API_BASE_URL ?? "http://localhost:8080";

export const API_BASE_URL = trimSlash(rawBase);

export const API_TOKEN = process.env.NEXT_PUBLIC_MAPLEQUERY_API_TOKEN ?? "";

export const APP_ENV = process.env.NEXT_PUBLIC_MAPLEQUERY_ENV ?? "dev";

export function authHeaders(
  extra?: Record<string, string>,
): Record<string, string> {
  const h: Record<string, string> = { ...(extra ?? {}) };
  if (API_TOKEN) h["Authorization"] = `Bearer ${API_TOKEN}`;
  return h;
}
