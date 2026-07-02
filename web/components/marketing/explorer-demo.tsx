"use client";

import * as React from "react";
import { Check, ShieldCheck } from "lucide-react";
import { useInView } from "./reveal";

const SQL_STEPS = [
  "SELECT",
  "\n  department,",
  "\n  SUM(spend_cad) AS total",
  "\nFROM raw.rows",
  "\nWHERE document_id IN (",
  "\n  'contracts-2023-q1',",
  "\n  'contracts-2023-q2',",
  "\n  'contracts-2023-q3',",
  "\n  'contracts-2023-q4'",
  "\n)",
  "\nGROUP BY department",
  "\nORDER BY total DESC",
  "\nLIMIT 10;",
];

const FULL_SQL = SQL_STEPS.join("");

type Phase = "idle" | "typing" | "guard" | "dryRun" | "done";

export function ExplorerDemo() {
  const [ref, inView] = useInView<HTMLDivElement>();
  const [phase, setPhase] = React.useState<Phase>("idle");
  const [chars, setChars] = React.useState(0);

  React.useEffect(() => {
    if (!inView) {
      setPhase("idle");
      setChars(0);
      return;
    }
    setPhase("typing");
  }, [inView]);

  React.useEffect(() => {
    if (phase !== "typing") return;
    if (chars >= FULL_SQL.length) {
      const t = window.setTimeout(() => setPhase("guard"), 320);
      return () => window.clearTimeout(t);
    }
    const step = FULL_SQL[chars] === "\n" ? 60 : 22;
    const t = window.setTimeout(() => setChars((c) => c + 1), step);
    return () => window.clearTimeout(t);
  }, [phase, chars]);

  React.useEffect(() => {
    if (phase === "guard") {
      const t = window.setTimeout(() => setPhase("dryRun"), 700);
      return () => window.clearTimeout(t);
    }
    if (phase === "dryRun") {
      const t = window.setTimeout(() => setPhase("done"), 900);
      return () => window.clearTimeout(t);
    }
    if (phase === "done") {
      const t = window.setTimeout(() => {
        setChars(0);
        setPhase("typing");
      }, 4200);
      return () => window.clearTimeout(t);
    }
  }, [phase]);

  const shown = FULL_SQL.slice(0, chars);

  return (
    <div
      ref={ref}
      className="relative overflow-hidden rounded-2xl border border-hairline bg-white shadow-xl"
    >
      <div className="flex items-center justify-between border-b border-hairline bg-surface-soft/60 px-4 py-2.5">
        <div className="flex items-center gap-2">
          <span className="h-2.5 w-2.5 rounded-full bg-coral/70" />
          <span className="h-2.5 w-2.5 rounded-full bg-amber/70" />
          <span className="h-2.5 w-2.5 rounded-full bg-teal/70" />
        </div>
        <span className="font-mono text-[10px] uppercase tracking-[0.14em] text-muted">
          Explorer · SQL step
        </span>
      </div>

      <pre className="min-h-[248px] px-5 py-4 font-mono text-[12.5px] leading-relaxed text-ink">
        <SqlHighlighted text={shown} />
        {phase === "typing" && (
          <span className="ml-0.5 inline-block h-4 w-[2px] animate-dot-blink bg-coral align-middle" />
        )}
      </pre>

      <div className="flex flex-wrap items-center gap-2 border-t border-hairline bg-surface-soft/50 px-4 py-3">
        <Chip
          active={phase === "guard" || phase === "dryRun" || phase === "done"}
          icon={<ShieldCheck className="h-3 w-3" />}
          label="Guard · SELECT-only"
        />
        <Chip
          active={phase === "dryRun" || phase === "done"}
          icon={<span className="text-[10px]">⛽</span>}
          label="Dry run · 42 MB"
        />
        <Chip
          active={phase === "done"}
          icon={<Check className="h-3 w-3" />}
          label="Executed · 12 rows"
        />
      </div>
    </div>
  );
}

function Chip({
  active,
  icon,
  label,
}: {
  active: boolean;
  icon: React.ReactNode;
  label: string;
}) {
  return (
    <span
      className={
        "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 font-mono text-[10px] transition-all duration-500 " +
        (active
          ? "border-navy/20 bg-white text-navy opacity-100"
          : "border-hairline bg-white/40 text-muted opacity-40")
      }
    >
      {icon}
      {label}
    </span>
  );
}

const KEYWORDS = new Set([
  "SELECT",
  "FROM",
  "WHERE",
  "GROUP",
  "BY",
  "ORDER",
  "LIMIT",
  "IN",
  "AS",
  "AND",
  "SUM",
  "DESC",
  "ASC",
]);

function SqlHighlighted({ text }: { text: string }) {
  const parts = text.split(/(\s+|,|\(|\)|;)/);
  return (
    <>
      {parts.map((p, i) => {
        const upper = p.toUpperCase();
        if (KEYWORDS.has(upper)) {
          return (
            <span key={i} className="font-semibold text-navy">
              {p}
            </span>
          );
        }
        if (/^'[^']*'$/.test(p)) {
          return (
            <span key={i} className="text-coral">
              {p}
            </span>
          );
        }
        return <React.Fragment key={i}>{p}</React.Fragment>;
      })}
    </>
  );
}
