import type { HistoryMessage } from "./types";

/**
 * Append the user's current question to the running transcript.
 *
 * The server owns compaction (§6 of 5.1); the client's job is to send up
 * whatever it stored last turn plus the new question. We never trim on
 * this side, but we do enforce the same 200-message sanity cap the server
 * rejects on, so we surface the wall a beat earlier.
 */
const MAX_MESSAGES = 200;

export function appendUserTurn(
  history: HistoryMessage[],
  question: string,
): HistoryMessage[] {
  return [...history, { role: "user", content: question }];
}

export function appendAssistantTurn(
  history: HistoryMessage[],
  content: string,
): HistoryMessage[] {
  return [...history, { role: "assistant", content }];
}

export function isNearMessageCap(history: HistoryMessage[]): boolean {
  return history.length >= MAX_MESSAGES - 10;
}

export function isAtMessageCap(history: HistoryMessage[]): boolean {
  return history.length >= MAX_MESSAGES;
}
