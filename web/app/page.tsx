import Link from "next/link";
import { ArrowRight, Compass, MessagesSquare, Notebook } from "lucide-react";
import { cn } from "@/lib/utils";

const surfaces = [
  {
    href: "/chat",
    label: "Ask",
    title: "Chat + evidence rail",
    body: "Type a question. Watch the answer build with a live trace of retrieval, guardrails, and SQL on the right.",
    icon: MessagesSquare,
    accent: "bg-coral/15 text-navy",
    cta: "Open the chat",
  },
  {
    href: "/notebook",
    label: "Notebook",
    title: "Prose + live query",
    body: "Mix Markdown with runnable questions. Export the finished thread as a report.",
    icon: Notebook,
    accent: "bg-teal/15 text-teal",
    cta: "Open the notebook",
  },
  {
    href: "/explorer",
    label: "Explore",
    title: "Split explorer",
    body: "Ask, review the SQL step, then edit it directly and re-run against the guarded executor.",
    icon: Compass,
    accent: "bg-amber/25 text-[#b7791f]",
    cta: "Open the explorer",
  },
];

export default function LandingPage() {
  return (
    <>
      <Hero />
      <SurfaceGrid />
      <TrustBand />
      <CallToAction />
      <SiteFooter />
    </>
  );
}

function Hero() {
  return (
    <section className="hero-grad relative overflow-hidden">
      <span
        aria-hidden="true"
        className="pointer-events-none absolute left-[8%] top-24 h-24 w-24 rounded-2xl bg-coral/10"
      />
      <span
        aria-hidden="true"
        className="pointer-events-none absolute right-[10%] top-40 h-16 w-16 rounded-full bg-teal/15"
      />
      <span
        aria-hidden="true"
        className="pointer-events-none absolute bottom-10 right-[22%] h-10 w-10 rounded-lg bg-amber/20"
      />

      <div className="relative mx-auto grid max-w-6xl items-center gap-10 px-4 py-20 md:px-6 md:py-24 lg:grid-cols-[1.1fr_0.9fr]">
        <div>
          <h1 className="max-w-3xl text-4xl font-semibold leading-[1.08] tracking-tight text-ink sm:text-5xl md:text-6xl">
            Ask hard questions of{" "}
            <span className="relative inline-block">
              <span className="relative z-10">government data</span>
              <span
                aria-hidden="true"
                className="absolute inset-x-0 bottom-1 -z-0 h-[10px] bg-coral/40"
              />
            </span>
            . Get answers you can cite.
          </h1>
          <p className="mt-6 max-w-xl text-lg leading-relaxed text-body">
            MapleQuery turns fragmented Canadian open data into a plain-language
            conversation. Every figure carries a footnote that traces straight
            back to the original record.
          </p>
          <div className="mt-8 flex flex-wrap items-center gap-3">
            <Link
              href="/chat"
              className="inline-flex items-center gap-2 rounded-md bg-coral px-5 py-2.5 text-sm font-medium text-ink transition-colors hover:bg-coral-active focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-navy focus-visible:ring-offset-2 focus-visible:ring-offset-canvas"
            >
              Ask a question
              <ArrowRight className="h-4 w-4" />
            </Link>
            <Link
              href="/datasets"
              className="rounded-md border border-hairline bg-white px-5 py-2.5 text-sm font-medium text-ink transition-colors hover:bg-surface-soft focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-navy"
            >
              Browse the corpus
            </Link>
          </div>

          <dl className="mt-12 grid max-w-lg grid-cols-3 gap-6">
            <Stat n="3,700+" label="Documents" />
            <Stat n="Millions" label="Rows of data" />
            <Stat n="100%" label="Answers cited or refused" />
          </dl>
        </div>

        <HeroSchematic />
      </div>
    </section>
  );
}

/**
 * An abstract schematic of the app's shape: ask → retrieve → answer.
 * Deliberately non-representational. No fabricated numbers, dataset names,
 * or facts. It's a visual anchor for the hero, not a screenshot.
 */
function HeroSchematic() {
  return (
    <div className="relative mx-auto w-full max-w-md">
      <div
        aria-hidden="true"
        className="pointer-events-none absolute -inset-8 rounded-[2rem] bg-gradient-to-br from-coral/10 via-transparent to-navy/10 blur-2xl"
      />
      <div className="relative flex flex-col gap-3 rounded-2xl border border-hairline bg-white p-5 shadow-xl">
        <div className="flex items-center justify-between">
          <span className="text-[11px] font-medium uppercase tracking-[0.16em] text-muted">
            Question
          </span>
          <span className="h-1.5 w-1.5 rounded-full bg-coral" />
        </div>
        <div className="h-2 w-4/5 rounded-full bg-surface-card" />
        <div className="h-2 w-2/3 rounded-full bg-surface-card" />

        <div className="mt-4 grid gap-2">
          <span className="text-[11px] font-medium uppercase tracking-[0.16em] text-muted">
            Retrieval
          </span>
          <div className="grid gap-2">
            <RetrievalBar accent="from-navy/70 to-navy/40" width="w-11/12" />
            <RetrievalBar accent="from-teal/70 to-teal/30" width="w-9/12" />
            <RetrievalBar accent="from-amber/80 to-amber/40" width="w-7/12" />
          </div>
        </div>

        <div className="mt-4 grid gap-2 rounded-xl bg-surface-soft/70 p-4">
          <span className="text-[11px] font-medium uppercase tracking-[0.16em] text-muted">
            Answer
          </span>
          <div className="h-2 w-full rounded-full bg-white" />
          <div className="h-2 w-5/6 rounded-full bg-white" />
          <div className="flex items-center gap-2 pt-1">
            <div className="h-2 flex-1 rounded-full bg-white" />
            <span className="rounded bg-coral/25 px-1.5 py-0.5 text-[9px] font-semibold text-navy">
              cite
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

function RetrievalBar({
  accent,
  width,
}: {
  accent: string;
  width: string;
}) {
  return (
    <div className="flex items-center gap-2">
      <div className="h-6 w-6 shrink-0 rounded-md border border-hairline bg-surface-soft" />
      <div
        className={cn(
          "h-2 rounded-full bg-gradient-to-r",
          accent,
          width,
        )}
      />
    </div>
  );
}

function Stat({ n, label }: { n: string; label: string }) {
  return (
    <div>
      <dd className="text-3xl font-semibold tracking-tight text-ink">{n}</dd>
      <dt className="mt-1 text-xs text-muted">{label}</dt>
    </div>
  );
}

function SurfaceGrid() {
  return (
    <section className="mx-auto max-w-6xl px-4 py-16 md:px-6">
      <p className="mb-3 text-xs font-semibold uppercase tracking-[0.18em] text-navy">
        Three ways in
      </p>
      <h2 className="max-w-3xl text-3xl font-semibold leading-tight tracking-tight text-ink md:text-4xl">
        Ask, remix, or drop to raw SQL. Same guarded corpus behind every
        surface.
      </h2>
      <div className="mt-10 grid gap-5 md:grid-cols-3">
        {surfaces.map((s) => {
          const Icon = s.icon;
          return (
            <Link
              key={s.href}
              href={s.href}
              className="group flex flex-col rounded-2xl border border-hairline bg-white p-6 transition-all hover:-translate-y-0.5 hover:shadow-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-navy"
            >
              <span
                className={cn(
                  "mb-4 inline-grid h-10 w-10 place-items-center rounded-lg",
                  s.accent,
                )}
              >
                <Icon className="h-5 w-5" />
              </span>
              <p className="text-[11px] font-semibold uppercase tracking-wider text-muted">
                {s.label}
              </p>
              <h3 className="mt-1 text-xl font-semibold tracking-tight text-ink">
                {s.title}
              </h3>
              <p className="mt-2 flex-1 text-sm leading-relaxed text-body">
                {s.body}
              </p>
              <span className="mt-4 inline-flex items-center gap-1 text-sm font-medium text-navy group-hover:underline">
                {s.cta}
                <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" />
              </span>
            </Link>
          );
        })}
      </div>
    </section>
  );
}

function TrustBand() {
  const items = [
    {
      title: "First-class citations",
      body: "Every figure links back to the source dataset in the corpus.",
    },
    {
      title: "Honest uncertainty",
      body: "When the evidence is thin, MapleQuery says so instead of guessing.",
    },
    {
      title: "Guarded execution",
      body: "Every SQL statement passes a static guard before it hits BigQuery.",
    },
  ];
  return (
    <section className="border-y border-hairline bg-surface-soft/60">
      <div className="mx-auto grid max-w-6xl gap-10 px-4 py-16 md:grid-cols-2 md:px-6">
        <div>
          <h2 className="text-3xl font-semibold tracking-tight text-ink">
            Trust is a feature, not a footnote.
          </h2>
          <p className="mt-3 max-w-md text-body">
            The corpus is Canadian federal open data. Every answer traces back
            to a published record, and the guardrails are non-negotiable.
          </p>
        </div>
        <ul className="grid gap-4">
          {items.map((it) => (
            <li
              key={it.title}
              className="rounded-xl border border-hairline bg-white p-5"
            >
              <p className="text-sm font-semibold text-ink">{it.title}</p>
              <p className="mt-1 text-sm text-body">{it.body}</p>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}

function CallToAction() {
  return (
    <section className="mx-auto max-w-6xl px-4 py-16 md:px-6">
      <div className="rounded-2xl border border-hairline bg-white p-8 md:p-12">
        <div className="flex flex-col items-start gap-6 md:flex-row md:items-center md:justify-between">
          <div>
            <h2 className="text-2xl font-semibold tracking-tight text-ink md:text-3xl">
              Start with a question.
            </h2>
            <p className="mt-2 max-w-xl text-body">
              Ask in plain language, watch the answer build with live citations,
              and click any card to trace it.
            </p>
          </div>
          <div className="flex flex-wrap gap-3">
            <Link
              href="/chat"
              className="inline-flex items-center gap-2 rounded-md bg-coral px-5 py-2.5 text-sm font-medium text-ink transition-colors hover:bg-coral-active focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-navy focus-visible:ring-offset-2 focus-visible:ring-offset-canvas"
            >
              Open the chat
              <ArrowRight className="h-4 w-4" />
            </Link>
            <Link
              href="/datasets"
              className="rounded-md border border-hairline bg-white px-5 py-2.5 text-sm font-medium text-ink transition-colors hover:bg-surface-soft focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-navy"
            >
              Browse the corpus
            </Link>
          </div>
        </div>
      </div>
    </section>
  );
}

function SiteFooter() {
  return (
    <footer className="border-t border-hairline bg-canvas">
      <div className="mx-auto max-w-6xl px-4 py-6 text-xs text-muted md:px-6">
        MapleQuery · answers only from cited datasets.
      </div>
    </footer>
  );
}
