"use client";

import * as React from "react";
import { useRouter } from "next/navigation";
import { Message } from "@/components/evidence/message";
import { EvidenceRail } from "@/components/evidence/evidence-rail";
import { CostBadge } from "@/components/evidence/cost-badge";
import { ChatComposer } from "./chat-composer";
import { ConversationSwitcher } from "./conversation-switcher";
import { useChatStream } from "./use-chat-stream";
import {
  conversations,
  type EvidenceCard,
  type StoredConversation,
} from "@/lib/storage";
import type { HistoryMessage } from "@/lib/types";
import { appendUserTurn, isAtMessageCap } from "@/lib/history";
import { useToast } from "@/components/ui/toast";
import { truncate, uuid } from "@/lib/utils";
import { track } from "@/lib/analytics";

const SUGGESTIONS = [
  "Which federal departments spent the most on IT consulting in 2023?",
  "Compare housing grant approvals across provinces since 2020.",
  "How has immigration PR processing time changed since 2019?",
];

export interface ChatContainerProps {
  conversationId: string;
  /** Question forwarded from a `?q=` link on the landing page. */
  initialQuestion?: string;
}

interface Turn {
  id: string;
  question: string;
  assistantText: string;
  cards: EvidenceCard[];
  meta?: { dollars: number; toolCalls: number; elapsedMs: number | null; cached: boolean };
  status: "complete" | "streaming" | "error";
  errorMessage?: string;
}

export function ChatContainer({
  conversationId,
  initialQuestion,
}: ChatContainerProps) {
  const router = useRouter();
  const toast = useToast();

  const [conversation, setConversation] = React.useState<StoredConversation | null>(null);
  const [turns, setTurns] = React.useState<Turn[]>([]);
  const [index, setIndex] = React.useState<
    { id: string; title: string; updatedAt: string }[]
  >([]);
  const [currentTurnId, setCurrentTurnId] = React.useState<string | null>(null);
  const threadRef = React.useRef<HTMLDivElement>(null);

  // Refresh conversation index whenever storage changes below.
  const refreshIndex = React.useCallback(() => {
    setIndex(conversations.list());
  }, []);

  // Restore on mount / when conversationId changes.
  React.useEffect(() => {
    const stored = conversations.load(conversationId);
    refreshIndex();

    if (stored) {
      setConversation(stored);
      setTurns(rehydrateTurns(stored));
    } else {
      const now = new Date().toISOString();
      const fresh: StoredConversation = {
        id: conversationId,
        title: "New conversation",
        createdAt: now,
        updatedAt: now,
        history: [],
        evidenceByTurnId: {},
      };
      setConversation(fresh);
      setTurns([]);
    }
    setCurrentTurnId(null);
  }, [conversationId, refreshIndex]);

  const { state, send, abort, reset } = useChatStream({
    conversationId,
    onDone: () => {
      // The stream reducer holds the final state; the persistence pass in the
      // effect below picks it up.
    },
  });

  // Live-mirror the streaming turn into the visible thread.
  React.useEffect(() => {
    if (!currentTurnId) return;
    setTurns((prev) =>
      prev.map((t) =>
        t.id === currentTurnId
          ? {
              ...t,
              assistantText: state.assistantText,
              cards: state.cards.map(cardToStored),
              status:
                state.status === "streaming"
                  ? "streaming"
                  : state.status === "error"
                    ? "error"
                    : "complete",
              errorMessage: state.error ?? undefined,
              meta: {
                dollars: state.dollars,
                toolCalls: state.toolCalls,
                elapsedMs: state.elapsedMs,
                cached: state.cached,
              },
            }
          : t,
      ),
    );

    // Persist once the turn terminates.
    if (state.status === "done" || state.status === "error") {
      persistConversation();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    state.assistantText,
    state.cards,
    state.status,
    state.dollars,
    state.toolCalls,
    state.elapsedMs,
    state.cached,
    state.error,
    currentTurnId,
  ]);

  React.useEffect(() => {
    // Scroll the thread to bottom on new content.
    const el = threadRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [turns.length, state.assistantText.length, state.cards.length]);

  const persistConversation = React.useCallback(() => {
    setConversation((prev) => {
      if (!prev) return prev;
      const nowIso = new Date().toISOString();
      const nextTurns = turns.map((t) =>
        t.id === currentTurnId
          ? {
              ...t,
              assistantText: state.assistantText,
              cards: state.cards.map(cardToStored),
              status:
                state.status === "streaming"
                  ? ("streaming" as const)
                  : state.status === "error"
                    ? ("error" as const)
                    : ("complete" as const),
              meta: {
                dollars: state.dollars,
                toolCalls: state.toolCalls,
                elapsedMs: state.elapsedMs,
                cached: state.cached,
              },
            }
          : t,
      );

      const history = buildHistoryFromTurns(nextTurns);
      const title =
        prev.title !== "New conversation" && prev.title
          ? prev.title
          : truncate(nextTurns[0]?.question ?? "New conversation", 60);

      const evidenceByTurnId: Record<string, EvidenceCard[]> = {};
      for (const t of nextTurns) evidenceByTurnId[t.id] = t.cards;

      // Append this turn's memory record (dedup by turn_id — the
      // persistence pass can fire more than once per turn), capped to
      // the server's ingest limit.
      const prevRecords = prev.turnRecords ?? [];
      const record = state.turnRecord as { turn_id?: unknown } | null;
      const alreadyStored =
        record != null &&
        prevRecords.some(
          (r) => (r as { turn_id?: unknown }).turn_id === record.turn_id,
        );
      const turnRecords =
        record != null && !alreadyStored
          ? [...prevRecords, record].slice(-50)
          : prevRecords;

      const next: StoredConversation = {
        ...prev,
        title,
        history,
        evidenceByTurnId,
        turnRecords,
        updatedAt: nowIso,
      };
      conversations.save(next);
      refreshIndex();
      return next;
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    currentTurnId,
    state.assistantText,
    state.cards,
    state.status,
    state.dollars,
    state.toolCalls,
    state.elapsedMs,
    state.cached,
    state.turnRecord,
    turns,
    refreshIndex,
  ]);

  const handleSubmit = async (question: string) => {
    if (!conversation) return;
    const history = buildHistoryFromTurns(turns);
    if (isAtMessageCap(history)) {
      toast.show(
        "This conversation is at the 200-message cap. Start a new one.",
        "error",
      );
      return;
    }

    const turnId = uuid();
    setCurrentTurnId(turnId);
    setTurns((prev) => [
      ...prev,
      {
        id: turnId,
        question,
        assistantText: "",
        cards: [],
        status: "streaming",
      },
    ]);
    reset();

    track("chat_message_sent", {
      conversation_id: conversationId,
      turn_id: turnId,
      question_length: question.length,
      history_length: history.length,
    });

    const nextHistory = appendUserTurn(history, question);
    await send(question, nextHistory, conversation.turnRecords ?? []);
  };

  const handleNewConversation = () => {
    abort();
    const id = uuid();
    router.push(`/chat/${id}`);
  };

  const handleDelete = (id: string) => {
    conversations.remove(id);
    refreshIndex();
    if (id === conversationId) {
      handleNewConversation();
    }
  };

  return (
    <div className="flex h-[calc(100vh-4rem)] min-h-0">
      <ConversationSwitcher
        currentId={conversationId}
        index={index}
        onNew={handleNewConversation}
        onDelete={handleDelete}
      />

      <div className="flex min-w-0 flex-1 border-r border-hairline">
        <div className="flex min-w-0 flex-1 flex-col">
          <div
            ref={threadRef}
            className="flex-1 space-y-8 overflow-y-auto px-4 py-8 md:px-6 lg:px-8"
            aria-live="polite"
          >
            {turns.length === 0 && (
              <EmptyState />
            )}
            {turns.map((t) => (
              <React.Fragment key={t.id}>
                <Message role="user" content={t.question} />
                <Message
                  role="assistant"
                  content={t.assistantText}
                  streaming={t.status === "streaming"}
                  meta={
                    t.meta && (t.status === "complete" || t.status === "error") ? (
                      <CostBadge
                        dollars={t.meta.dollars}
                        toolCalls={t.meta.toolCalls}
                        elapsedMs={t.meta.elapsedMs}
                        cached={t.meta.cached}
                      />
                    ) : undefined
                  }
                />
                {t.status === "error" && (
                  <div className="rounded-lg border border-error/30 bg-error/10 px-4 py-3 text-sm text-error">
                    {t.errorMessage ?? "Something went wrong. Try again."}
                  </div>
                )}
              </React.Fragment>
            ))}
          </div>

          <ChatComposer
            onSubmit={handleSubmit}
            onAbort={abort}
            streaming={state.status === "streaming"}
            suggestions={turns.length === 0 ? SUGGESTIONS : []}
            initialText={turns.length === 0 ? initialQuestion : undefined}
          />
        </div>
      </div>

      <aside className="hidden w-[420px] shrink-0 bg-surface-soft/70 lg:block">
        <EvidenceRail
          cards={
            currentTurnId
              ? (turns.find((t) => t.id === currentTurnId)?.cards.map(
                  storedToCard,
                ) ?? [])
              : (turns[turns.length - 1]?.cards.map(storedToCard) ?? [])
          }
          isStreaming={state.status === "streaming"}
          cached={state.cached}
        />
      </aside>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="mx-auto max-w-2xl pt-16 text-center">
      <p className="mb-3 inline-flex items-center gap-2 rounded-full border border-hairline bg-white px-3 py-1 text-xs font-medium text-muted">
        <span className="h-1.5 w-1.5 rounded-full bg-coral" />
        Chat + evidence rail
      </p>
      <h1 className="font-display text-3xl font-medium tracking-tight text-ink md:text-4xl">
        What do you want to know about the corpus?
      </h1>
      <p className="mt-3 text-body">
        Ask in plain language. MapleQuery will search datasets, generate SQL,
        pass it through the guard, and stream the answer with a live trace on
        the right.
      </p>
    </div>
  );
}

function cardToStored(c: import("@/components/evidence/evidence-rail").RailCard): EvidenceCard {
  return {
    id: c.id,
    kind: c.kind,
    payload: c,
    createdAt: new Date().toISOString(),
  };
}

function storedToCard(
  c: EvidenceCard,
): import("@/components/evidence/evidence-rail").RailCard {
  return c.payload as import("@/components/evidence/evidence-rail").RailCard;
}

function rehydrateTurns(s: StoredConversation): Turn[] {
  const turns: Turn[] = [];
  let i = 0;
  const h = s.history;
  const evidence = s.evidenceByTurnId ?? {};
  const turnIds = Object.keys(evidence);
  let turnIdx = 0;
  while (i < h.length) {
    const msg = h[i];
    if (msg.role === "user") {
      const nextAssistantIdx = h.findIndex(
        (m, j) => j > i && m.role === "assistant",
      );
      const assistantMsg =
        nextAssistantIdx >= 0
          ? (h[nextAssistantIdx] as {
              role: "assistant";
              content: string | null;
            })
          : null;
      const turnId = turnIds[turnIdx++] ?? uuid();
      turns.push({
        id: turnId,
        question: msg.content,
        assistantText: assistantMsg?.content ?? "",
        cards: evidence[turnId] ?? [],
        status: "complete",
      });
      i = nextAssistantIdx >= 0 ? nextAssistantIdx + 1 : i + 1;
    } else {
      i += 1;
    }
  }
  return turns;
}

function buildHistoryFromTurns(turns: Turn[]): HistoryMessage[] {
  const h: HistoryMessage[] = [];
  for (const t of turns) {
    h.push({ role: "user", content: t.question });
    if (t.assistantText) {
      h.push({ role: "assistant", content: t.assistantText });
    }
  }
  return h;
}
