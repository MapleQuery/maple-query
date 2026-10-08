/**
 * Server-side relay to agent-service.
 *
 * The bearer token used to ship in the browser bundle as a
 * `NEXT_PUBLIC_` variable, which made it public by construction: anyone
 * could read it out of the JS and call the API directly. The browser now
 * calls this route, and the token is added here, on the server, from an
 * env var the client bundle never sees.
 *
 * Only the paths the web app uses are relayed, so this is not a general
 * proxy onto the service. Edge runtime: it streams the `/chat` SSE body
 * straight through and only has to start responding within Vercel's
 * edge window, not finish (a research turn can run ~90 s).
 */

export const runtime = "edge";
export const dynamic = "force-dynamic";

const ALLOWED = [/^chat$/, /^sql\/run$/, /^corpus\/stats$/, /^datasets(\/[^/]+(\/(columns|documents))?)?$/];

function upstreamBase(): string | null {
  const raw = process.env.MAPLEQUERY_API_BASE_URL;
  return raw ? raw.replace(/\/+$/, "") : null;
}

function token(): string {
  return process.env.MAPLEQUERY_API_TOKEN ?? "";
}

async function relay(
  req: Request,
  { params }: { params: Promise<{ path: string[] }> },
): Promise<Response> {
  const { path } = await params;
  const joined = path.join("/");
  if (!ALLOWED.some((re) => re.test(joined))) {
    return new Response("not found", { status: 404 });
  }
  const base = upstreamBase();
  if (!base) {
    return new Response("MAPLEQUERY_API_BASE_URL is not configured", { status: 500 });
  }
  const search = new URL(req.url).search;
  const headers: Record<string, string> = {};
  const contentType = req.headers.get("content-type");
  if (contentType) headers["Content-Type"] = contentType;
  const accept = req.headers.get("accept");
  if (accept) headers["Accept"] = accept;
  const t = token();
  if (t) headers["Authorization"] = `Bearer ${t}`;

  const upstream = await fetch(`${base}/${joined}${search}`, {
    method: req.method,
    headers,
    body: req.method === "GET" || req.method === "HEAD" ? undefined : await req.text(),
    signal: req.signal,
  });

  const out = new Headers();
  for (const name of ["content-type", "cache-control"]) {
    const v = upstream.headers.get(name);
    if (v) out.set(name, v);
  }
  // Keep proxies between here and the browser from buffering the stream.
  if (out.get("content-type")?.includes("text/event-stream")) {
    out.set("cache-control", "no-cache, no-transform");
    out.set("x-accel-buffering", "no");
  }
  return new Response(upstream.body, { status: upstream.status, headers: out });
}

export { relay as GET, relay as POST };
