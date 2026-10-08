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

  const init: RequestInit = {
    method: req.method,
    headers,
    body: req.method === "GET" || req.method === "HEAD" ? undefined : await req.text(),
    signal: req.signal,
  };

  if (joined === "chat") return chatStream(`${base}/chat${search}`, init);

  const upstream = await fetch(`${base}/${joined}${search}`, init);

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

/**
 * `/chat`, answered before agent-service has answered.
 *
 * The service scales to zero, and a cold start can take longer than the
 * window an edge function has to send its first byte: the first question
 * after a quiet spell came back as a 504. So the stream opens at once
 * with an SSE comment (which the client ignores), then the upstream body
 * is piped through. An upstream failure becomes an `error` event, which
 * the chat already renders, since the status line has been sent by then.
 */
function chatStream(url: string, init: RequestInit): Response {
  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      controller.enqueue(encoder.encode(": connecting\n\n"));
      // Keep the connection visibly alive through a slow cold start.
      const beat = setInterval(() => {
        controller.enqueue(encoder.encode(": waiting\n\n"));
      }, 10_000);
      try {
        const upstream = await fetch(url, init);
        clearInterval(beat);
        if (!upstream.ok || !upstream.body) {
          const detail = (await upstream.text()).slice(0, 300);
          const frame = {
            message: `agent-service ${upstream.status}: ${detail}`,
            retryable: upstream.status >= 500,
          };
          controller.enqueue(
            encoder.encode(`event: error\ndata: ${JSON.stringify(frame)}\n\n`),
          );
          controller.close();
          return;
        }
        const reader = upstream.body.getReader();
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          controller.enqueue(value);
        }
        controller.close();
      } catch (err) {
        clearInterval(beat);
        const frame = { message: `agent-service unreachable: ${String(err)}`, retryable: true };
        try {
          controller.enqueue(
            encoder.encode(`event: error\ndata: ${JSON.stringify(frame)}\n\n`),
          );
          controller.close();
        } catch {
          // The client already went away.
        }
      }
    },
  });
  return new Response(stream, {
    status: 200,
    headers: {
      "content-type": "text/event-stream",
      "cache-control": "no-cache, no-transform",
      "x-accel-buffering": "no",
    },
  });
}

export { relay as GET, relay as POST };
