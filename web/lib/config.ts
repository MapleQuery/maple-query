/**
 * The browser talks to agent-service only through this app's own
 * `/api/mq/*` relay (app/api/mq/[...path]/route.ts), which adds the
 * bearer token server-side. Nothing secret is read here: this module is
 * bundled into the client.
 */
export const API_BASE_URL = "/api/mq";

export const APP_ENV = process.env.NEXT_PUBLIC_MAPLEQUERY_ENV ?? "dev";

export function authHeaders(
  extra?: Record<string, string>,
): Record<string, string> {
  return { ...(extra ?? {}) };
}
