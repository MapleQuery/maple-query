"use client";

import * as React from "react";
import { MapleLeaf } from "@/components/ui/maple-leaf";
import { LogoMark } from "@/components/ui/logo";
import { cn } from "@/lib/utils";

/**
 * A looping, scripted recreation of the chat surface for the marketing
 * hero. Not an iframe — every element is a real DOM node so we get the
 * same fonts, colors, and motion vocabulary as the app itself.
 */
interface Turn {
  question: string;
  candidates: { title: string; distance: string; accent: string }[];
  answer: React.ReactNode;
  answerPlain: string;
  citation: string;
}

const TURNS: Turn[] = [
  {
    question:
      "Which departments spent the most on IT consulting in 2023?",
    candidates: [
      {
        title: "Federal contracts, disclosed",
        distance: "0.184",
        accent: "from-navy/70 to-navy/40",
      },
      {
        title: "Departmental spend, quarterly",
        distance: "0.211",
        accent: "from-teal/70 to-teal/30",
      },
      {
        title: "IT consulting, sector view",
        distance: "0.267",
        accent: "from-amber/80 to-amber/40",
      },
    ],
    answerPlain:
      "Public Services led at $612M, followed by National Defence and Shared Services. Every figure traces to the contract-level record in ",
    answer: (
      <>
        Public Services led at $612M, followed by National Defence and
        Shared Services. Every figure traces to the contract-level record in{" "}
      </>
    ),
    citation: "contracts-2023",
  },
  {
    question: "How did housing grant approvals shift after 2020?",
    candidates: [
      {
        title: "Housing grants, provincial rollup",
        distance: "0.142",
        accent: "from-navy/70 to-navy/40",
      },
      {
        title: "CMHC program disbursements",
        distance: "0.198",
        accent: "from-teal/70 to-teal/30",
      },
      {
        title: "Federal transfers to provinces",
        distance: "0.281",
        accent: "from-amber/80 to-amber/40",
      },
    ],
    answerPlain:
      "Approvals climbed 34% between 2020 and 2023 with the largest gains in BC and Ontario. Provincial rollups live in ",
    answer: (
      <>
        Approvals climbed 34% between 2020 and 2023 with the largest gains
        in BC and Ontario. Provincial rollups live in{" "}
      </>
    ),
    citation: "housing-grants-rollup",
  },
];

type Phase =
  | { kind: "typing-question"; turnIdx: number; charIdx: number }
  | { kind: "loading"; turnIdx: number }
  | { kind: "retrieval"; turnIdx: number; barIdx: number }
  | { kind: "typing-answer"; turnIdx: number; charIdx: number }
  | { kind: "hold"; turnIdx: number };

const CANADIAN_VERBS = [
  "Portaging",
  "Zambonieing",
  "Snowshoeing",
  "Percolating",
  "Deliberating",
];

export function MiniChat() {
  const [phase, setPhase] = React.useState<Phase>({
    kind: "typing-question",
    turnIdx: 0,
    charIdx: 0,
  });
  const [verbIdx, setVerbIdx] = React.useState(0);

  const turn = TURNS[phase.turnIdx];

  React.useEffect(() => {
    let timeout: number;

    switch (phase.kind) {
      case "typing-question": {
        if (phase.charIdx >= turn.question.length) {
          timeout = window.setTimeout(
            () => setPhase({ kind: "loading", turnIdx: phase.turnIdx }),
            420,
          );
        } else {
          timeout = window.setTimeout(
            () =>
              setPhase({
                kind: "typing-question",
                turnIdx: phase.turnIdx,
                charIdx: phase.charIdx + 1,
              }),
            26,
          );
        }
        break;
      }
      case "loading": {
        timeout = window.setTimeout(
          () =>
            setPhase({
              kind: "retrieval",
              turnIdx: phase.turnIdx,
              barIdx: 0,
            }),
          1200,
        );
        break;
      }
      case "retrieval": {
        if (phase.barIdx >= turn.candidates.length) {
          timeout = window.setTimeout(
            () =>
              setPhase({
                kind: "typing-answer",
                turnIdx: phase.turnIdx,
                charIdx: 0,
              }),
            360,
          );
        } else {
          timeout = window.setTimeout(
            () =>
              setPhase({
                kind: "retrieval",
                turnIdx: phase.turnIdx,
                barIdx: phase.barIdx + 1,
              }),
            220,
          );
        }
        break;
      }
      case "typing-answer": {
        if (phase.charIdx >= turn.answerPlain.length) {
          timeout = window.setTimeout(
            () => setPhase({ kind: "hold", turnIdx: phase.turnIdx }),
            240,
          );
        } else {
          timeout = window.setTimeout(
            () =>
              setPhase({
                kind: "typing-answer",
                turnIdx: phase.turnIdx,
                charIdx: phase.charIdx + 1,
              }),
            14,
          );
        }
        break;
      }
      case "hold": {
        timeout = window.setTimeout(
          () =>
            setPhase({
              kind: "typing-question",
              turnIdx: (phase.turnIdx + 1) % TURNS.length,
              charIdx: 0,
            }),
          3600,
        );
        break;
      }
    }

    return () => window.clearTimeout(timeout);
  }, [phase, turn]);

  React.useEffect(() => {
    const id = window.setInterval(
      () => setVerbIdx((i) => (i + 1) % CANADIAN_VERBS.length),
      900,
    );
    return () => window.clearInterval(id);
  }, []);

  const questionShown =
    phase.kind === "typing-question"
      ? turn.question.slice(0, phase.charIdx)
      : turn.question;

  const answerShownChars =
    phase.kind === "typing-answer" ? phase.charIdx : 0;
  const answerFullyShown =
    phase.kind === "typing-answer"
      ? phase.charIdx >= turn.answerPlain.length
      : phase.kind === "hold";
  const showingLoader = phase.kind === "loading";
  const showingRetrieval =
    phase.kind === "retrieval" ||
    phase.kind === "typing-answer" ||
    phase.kind === "hold";
  const showingAnswerBlock =
    phase.kind === "typing-answer" || phase.kind === "hold";
  const revealedBars =
    phase.kind === "retrieval"
      ? phase.barIdx
      : phase.kind === "typing-answer" || phase.kind === "hold"
        ? turn.candidates.length
        : 0;

  return (
    <div className="relative mx-auto w-full max-w-md">
      <div
        aria-hidden="true"
        className="pointer-events-none absolute -inset-8 rounded-[2rem] bg-gradient-to-br from-coral/10 via-transparent to-navy/10 blur-2xl"
      />
      <div className="relative flex flex-col gap-4 rounded-2xl border border-hairline bg-white p-5 shadow-xl">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <LogoMark className="h-6 w-6" />
            <span className="text-xs font-semibold uppercase tracking-[0.14em] text-muted">
              Ask
            </span>
          </div>
          <span className="inline-flex h-2 w-2 animate-pulse rounded-full bg-coral" />
        </div>

        <div className="flex justify-end">
          <div className="max-w-[85%] rounded-2xl rounded-br-md border border-hairline bg-surface-soft/80 px-3.5 py-2 text-[13px] leading-relaxed text-ink">
            {questionShown}
            {phase.kind === "typing-question" && <TypingCaret />}
          </div>
        </div>

        <div className="flex gap-2.5">
          <span className="mt-0.5 grid h-7 w-7 shrink-0 place-items-center rounded-full bg-coral/15">
            <MapleLeaf size={16} />
          </span>
          <div className="min-w-0 flex-1 space-y-3">
            {showingLoader && (
              <div className="inline-flex items-center gap-2 rounded-2xl rounded-tl-md border border-hairline bg-white px-3 py-2">
                <MapleLeaf pulse size={14} />
                <span className="font-mono text-[9px] font-medium uppercase tracking-[0.14em] text-navy">
                  {CANADIAN_VERBS[verbIdx]}
                </span>
              </div>
            )}

            {showingRetrieval && (
              <div className="space-y-2">
                <span className="text-[10px] font-semibold uppercase tracking-[0.16em] text-muted">
                  Retrieval
                </span>
                <div className="space-y-1.5">
                  {turn.candidates.map((c, i) => {
                    const active = i < revealedBars;
                    return (
                      <div
                        key={c.title}
                        className={cn(
                          "flex items-center gap-2 transition-all duration-500",
                          active
                            ? "opacity-100 translate-y-0"
                            : "opacity-0 translate-y-1",
                        )}
                      >
                        <div className="h-5 w-5 shrink-0 rounded-md border border-hairline bg-surface-soft" />
                        <div
                          className={cn(
                            "h-2 rounded-full bg-gradient-to-r",
                            c.accent,
                          )}
                          style={{
                            width: `${88 - i * 14}%`,
                          }}
                        />
                        <span className="ml-auto shrink-0 font-mono text-[9px] text-muted">
                          d={c.distance}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {showingAnswerBlock && (
              <div className="rounded-xl bg-surface-soft/70 p-3">
                <span className="text-[10px] font-semibold uppercase tracking-[0.16em] text-muted">
                  Answer
                </span>
                <p className="mt-1 text-[12.5px] leading-relaxed text-body">
                  {turn.answerPlain.slice(0, answerShownChars)}
                  {answerFullyShown && (
                    <span className="ml-0.5 inline-flex items-center rounded bg-coral/25 px-1.5 py-0.5 align-middle font-mono text-[9px] font-semibold text-navy">
                      {turn.citation}
                    </span>
                  )}
                  {phase.kind === "typing-answer" && <TypingCaret />}
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function TypingCaret() {
  return (
    <span
      aria-hidden="true"
      className="ml-0.5 inline-block h-3 w-[2px] animate-dot-blink bg-coral align-middle"
    />
  );
}
