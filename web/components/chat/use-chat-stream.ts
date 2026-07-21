"use client";

import * as React from "react";
import { streamChat } from "@/lib/sse";
import type { AgentEvent, HistoryMessage } from "@/lib/types";
import type { RailCard } from "@/components/evidence/evidence-rail";
import { uuid } from "@/lib/utils";

export interface StreamState {
  status: "idle" | "streaming" | "done" | "error";
  cards: RailCard[];
  assistantText: string;
  toolCalls: number;
  dollars: number;
  elapsedMs: number | null;
  cached: boolean;
  error: string | null;
  /** The server-built memory record for this turn, echoed back on the
   * next request as `turn_records`. */
  turnRecord: Record<string, unknown> | null;
}

const initialState: StreamState = {
  status: "idle",
  cards: [],
  assistantText: "",
  toolCalls: 0,
  dollars: 0,
  elapsedMs: null,
  cached: false,
  error: null,
  turnRecord: null,
};

type Action =
  | { type: "reset" }
  | { type: "start" }
  | { type: "event"; event: AgentEvent }
  | { type: "error"; message: string };

function upsertBySqlId(cards: RailCard[], mutator: (c: RailCard) => RailCard | null): RailCard[] {
  // Merge SQL generated → guarded → executed → rows into a single card by
  // walking backwards to the most recent sql_generated card without executed.
  for (let i = cards.length - 1; i >= 0; i--) {
    const c = cards[i];
    const next = mutator(c);
    if (next) {
      const copy = cards.slice();
      copy[i] = next;
      return copy;
    }
  }
  return cards;
}

function reducer(state: StreamState, action: Action): StreamState {
  switch (action.type) {
    case "reset":
      return { ...initialState };

    case "start":
      return { ...initialState, status: "streaming" };

    case "error":
      return {
        ...state,
        status: "error",
        error: action.message,
      };

    case "event": {
      const { name, payload } = action.event;
      switch (name) {
        case "turn_start":
          return { ...state, cached: payload.cached, status: "streaming" };

        case "cache_hit":
          return { ...state, cached: true };

        case "retrieval_started":
          return {
            ...state,
            cards: [
              ...state.cards,
              {
                id: uuid(),
                kind: "retrieval_started",
                query: payload.query,
                k: payload.k,
              },
            ],
          };

        case "datasets_ranked":
          return {
            ...state,
            cards: [
              ...state.cards,
              {
                id: uuid(),
                kind: "datasets_ranked",
                candidates: payload.candidates,
              },
            ],
          };

        case "columns_ranked":
          return {
            ...state,
            cards: [
              ...state.cards,
              {
                id: uuid(),
                kind: "columns_ranked",
                packageIds: payload.package_ids,
                candidates: payload.candidates,
              },
            ],
          };

        case "sample_rows":
          return {
            ...state,
            cards: [
              ...state.cards,
              {
                id: uuid(),
                kind: "sample_rows",
                packageId: payload.package_id,
                rows: payload.rows,
              },
            ],
          };

        case "derivation":
          return {
            ...state,
            cards: [
              ...state.cards,
              { id: uuid(), kind: "derivation", derivation: payload },
            ],
          };

        case "sql_generated":
          return {
            ...state,
            cards: [
              ...state.cards,
              {
                id: uuid(),
                kind: "sql_generated",
                sql: payload.sql,
                rationale: payload.rationale,
              },
            ],
          };

        case "sql_guarded":
          return {
            ...state,
            cards: upsertBySqlId(state.cards, (c) => {
              if (c.kind === "sql_generated" && !c.guard) {
                return {
                  ...c,
                  guard: {
                    accepted: payload.accepted,
                    reason: payload.reason,
                    sql_final: payload.sql_final,
                  },
                };
              }
              return null;
            }),
          };

        case "sql_executed":
          return {
            ...state,
            cards: upsertBySqlId(state.cards, (c) => {
              if (c.kind === "sql_generated" && !c.executed) {
                return {
                  ...c,
                  executed: {
                    row_count: payload.row_count,
                    elapsed_ms: payload.elapsed_ms ?? null,
                    rows: payload.sample_rows ?? [],
                  },
                };
              }
              return null;
            }),
          };

        case "rows":
          return {
            ...state,
            cards: upsertBySqlId(state.cards, (c) => {
              if (c.kind === "sql_generated" && c.executed) {
                return {
                  ...c,
                  executed: {
                    ...c.executed,
                    rows: [...c.executed.rows, ...payload.rows],
                  },
                };
              }
              return null;
            }),
          };

        case "message_delta":
          return { ...state, assistantText: state.assistantText + payload.delta };

        case "cost_update":
          return {
            ...state,
            dollars: payload.dollars_spent,
          };

        case "tool_error":
          return {
            ...state,
            cards: [
              ...state.cards,
              {
                id: uuid(),
                kind: "tool_error",
                tool: payload.tool,
                message: payload.message,
              },
            ],
          };

        case "budget_exceeded":
          return {
            ...state,
            cards: [
              ...state.cards,
              {
                id: uuid(),
                kind: "budget_exceeded",
                which: payload.which,
                value: payload.value,
                cap: payload.cap,
              },
            ],
          };

        case "turn_timeout":
          return {
            ...state,
            cards: [
              ...state.cards,
              {
                id: uuid(),
                kind: "turn_timeout",
                elapsed_ms: payload.elapsed_ms,
                cap_ms: payload.cap_ms,
              },
            ],
          };

        case "turn_record":
          return { ...state, turnRecord: payload.record };

        case "done":
          return {
            ...state,
            status: "done",
            toolCalls: payload.total_tool_calls,
            dollars: payload.total_dollars,
            elapsedMs: payload.elapsed_ms,
          };

        case "error":
          return {
            ...state,
            status: "error",
            error: payload.message,
          };

        default:
          return state;
      }
    }
  }
}

export interface UseChatStreamOptions {
  conversationId: string;
  onDone?: (finalState: StreamState) => void;
}

export function useChatStream({ conversationId, onDone }: UseChatStreamOptions) {
  const [state, dispatch] = React.useReducer(reducer, initialState);
  const abortRef = React.useRef<AbortController | null>(null);
  const stateRef = React.useRef(state);
  stateRef.current = state;

  React.useEffect(() => {
    return () => abortRef.current?.abort();
  }, []);

  const abort = React.useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
  }, []);

  const reset = React.useCallback(() => {
    abort();
    dispatch({ type: "reset" });
  }, [abort]);

  const send = React.useCallback(
    async (
      question: string,
      history: HistoryMessage[],
      turnRecords: Record<string, unknown>[] = [],
    ) => {
      abort();
      dispatch({ type: "start" });
      const controller = new AbortController();
      abortRef.current = controller;

      try {
        await streamChat(
          {
            conversation_id: conversationId,
            question,
            history,
            turn_records: turnRecords,
          },
          {
            onEvent: (event) => dispatch({ type: "event", event }),
            onDone: () => {
              onDone?.(stateRef.current);
            },
            onError: (err) => {
              // reducer already set status:"error" via the `error` event; only
              // update here for transport-level failures.
              if (stateRef.current.status !== "error") {
                dispatch({ type: "error", message: err.message });
              }
            },
            onMalformed: (name, raw, err) => {
              // eslint-disable-next-line no-console
              console.warn("Malformed SSE event", { name, raw, err });
            },
          },
          controller.signal,
        );
      } catch (err) {
        if (controller.signal.aborted) return;
        dispatch({
          type: "error",
          message: (err as Error).message ?? "Stream failed",
        });
      }
    },
    [conversationId, onDone, abort],
  );

  return { state, send, abort, reset };
}
