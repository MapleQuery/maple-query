import { fetchEventSource } from "@microsoft/fetch-event-source";
import { API_BASE_URL, authHeaders } from "./config";
import {
  AgentEventSchemas,
  type AgentEvent,
  type AgentEventName,
  type ChatRequest,
  type DoneEvent,
  type ErrorEvent,
} from "./types";

export interface StreamHandlers {
  onEvent: (event: AgentEvent) => void;
  onDone?: (summary: DoneEvent) => void;
  onError?: (err: ErrorEvent | { message: string; retryable: boolean }) => void;
  onOpen?: () => void;
  onMalformed?: (name: string, raw: string, err: unknown) => void;
}

class RetryableError extends Error {}
class FatalError extends Error {}

function isKnownEvent(name: string): name is AgentEventName {
  return name in AgentEventSchemas;
}

/**
 * Stream POST /chat with typed event dispatch.
 *
 * A malformed frame is logged and skipped — the stream keeps going.
 * On network failure the caller decides whether to retry; we don't
 * auto-reconnect (retries would replay tool calls the server already ran).
 */
export async function streamChat(
  request: ChatRequest,
  handlers: StreamHandlers,
  signal: AbortSignal,
): Promise<void> {
  let sawDone = false;

  await fetchEventSource(`${API_BASE_URL}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
      ...authHeaders(),
    },
    body: JSON.stringify(request),
    signal,
    openWhenHidden: true,

    onopen: async (res) => {
      const ct = res.headers.get("content-type") ?? "";
      if (res.ok && ct.includes("text/event-stream")) {
        handlers.onOpen?.();
        return;
      }
      if (res.status === 401 || res.status === 403 || res.status === 400) {
        throw new FatalError(`chat ${res.status}: ${await res.text()}`);
      }
      throw new RetryableError(`chat ${res.status}`);
    },

    onmessage: (msg) => {
      const name = msg.event || "message";
      if (!isKnownEvent(name)) {
        return;
      }
      try {
        const parsed = JSON.parse(msg.data);
        const schema = AgentEventSchemas[name];
        const payload = schema.parse(parsed);
        handlers.onEvent({ name, payload } as AgentEvent);
        if (name === "done") {
          sawDone = true;
          handlers.onDone?.(payload as DoneEvent);
        }
        if (name === "error") {
          handlers.onError?.(payload as ErrorEvent);
        }
      } catch (err) {
        handlers.onMalformed?.(name, msg.data, err);
      }
    },

    onerror: (err) => {
      if (err instanceof FatalError) {
        handlers.onError?.({ message: err.message, retryable: false });
        throw err;
      }
      handlers.onError?.({ message: String(err), retryable: true });
      throw err;
    },

    onclose: () => {
      if (!sawDone) {
        // Server closed without a terminal event — surface as retryable.
        handlers.onError?.({
          message: "stream closed before done",
          retryable: true,
        });
      }
    },
  });
}
