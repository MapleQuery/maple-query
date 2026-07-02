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
    cta: "Launch the demo",
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
      <Problem />
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
          <p className="mb-4 inline-flex items-center gap-2 rounded-full border border-hairline bg-white/70 px-3 py-1 text-xs font-medium text-muted">
            <span className="h-1.5 w-1.5 rounded-full bg-coral" />
            Live prototype · chat + evidence rail
          </p>
          <h1 className="max-w-3xl font-display text-4xl font-medium leading-[1.08] tracking-tight text-ink sm:text-5xl md:text-6xl">
            Ask hard questions of{" "}
            <span className="relative inline-block">
              <span className="relative z-10">government data</span>
              <span
                aria-hidden="true"
                className="absolute inset-x-0 bottom-1 -z-0 h-[10px] bg-coral/40"
              />
            </span>{" "}
            — and get answers you can cite.
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
              Launch the live demo
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
            <Stat n="~3,700" label="Datasets indexed" />
            <Stat n="120K+" label="Columns semantically tagged" />
            <Stat n="100%" label="Answers cited or refused" />
          </dl>
        </div>

        <div>
          <div className="overflow-hidden rounded-2xl border border-hairline bg-white shadow-xl">
            <div className="flex items-center gap-1.5 border-b border-hairline bg-surface-soft px-4 py-2.5">
              <span className="h-2.5 w-2.5 rounded-full bg-coral/40" />
              <span className="h-2.5 w-2.5 rounded-full bg-amber/40" />
              <span className="h-2.5 w-2.5 rounded-full bg-teal/40" />
              <span className="ml-2 font-mono text-[11px] text-muted">
                maplequery — ask
              </span>
            </div>
            <div className="space-y-3 p-4">
              <div className="flex justify-end">
                <div className="max-w-[80%] rounded-2xl rounded-tr-sm bg-surface-card px-3 py-2 text-sm text-ink">
                  How has federal IT contract spending changed since 2018?
                </div>
              </div>
              <div className="rounded-2xl rounded-tl-sm bg-canvas px-3 py-2.5 text-sm leading-relaxed text-body ring-1 ring-hairline">
                Spending rose from{" "}
                <span className="font-medium text-ink">$3.1B</span> to{" "}
                <span className="font-medium text-ink">$5.4B</span>
                <span className="ml-0.5 rounded bg-coral/25 px-1 align-super text-[10px] font-semibold text-navy">
                  [1]
                </span>{" "}
                — about <span className="font-medium text-ink">74%</span>.
              </div>
              <div className="rounded-lg border border-hairline bg-surface-soft p-3">
                <p className="mb-2 flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wider text-muted">
                  <span className="grid h-4 w-4 place-items-center rounded-full bg-navy text-[8px] text-white">
                    1
                  </span>{" "}
                  Evidence
                </p>
                <p className="text-xs font-medium text-ink">
                  Public Accounts of Canada — Vol. III, Contracts
                </p>
                <p className="font-mono text-[11px] text-muted">
                  2018–2024 · 14,228 rows · guarded SELECT
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function Stat({ n, label }: { n: string; label: string }) {
  return (
    <div>
      <dd className="font-display text-3xl font-medium text-ink">{n}</dd>
      <dt className="mt-1 text-xs text-muted">{label}</dt>
    </div>
  );
}

function Problem() {
  const cards = [
    {
      title: "Fragmented",
      body: "One topic is scattered across departments, portals, and formats.",
      accent: "bg-coral/15 text-navy",
    },
    {
      title: "Inconsistent",
      body: "Different structures, standards, gaps, and stale records.",
      accent: "bg-amber/25 text-[#b7791f]",
    },
    {
      title: "Technical",
      body: "Joining and cleaning needs SQL, warehousing, and time.",
      accent: "bg-teal/15 text-teal",
    },
    {
      title: "Slow",
      body: "Prep eats the hours meant for analysis and writing.",
      accent: "bg-success/15 text-success",
    },
  ];
  return (
    <section className="border-y border-hairline bg-surface-soft">
      <div className="mx-auto max-w-6xl px-4 py-16 md:px-6">
        <h2 className="font-display text-3xl font-medium tracking-tight text-ink">
          The data is public. Using it isn&rsquo;t.
        </h2>
        <p className="mt-3 max-w-2xl text-body">
          Access to government data has improved; usability hasn&rsquo;t kept
          pace. MapleQuery closes the last mile.
        </p>
        <div className="mt-10 grid gap-5 md:grid-cols-2 lg:grid-cols-4">
          {cards.map((c) => (
            <div
              key={c.title}
              className="rounded-xl border border-hairline bg-white p-5"
            >
              <span
                className={cn(
                  "mb-3 inline-grid h-9 w-9 place-items-center rounded-lg",
                  c.accent,
                )}
              >
                <span className="text-sm font-semibold">·</span>
              </span>
              <h3 className="text-sm font-semibold text-ink">{c.title}</h3>
              <p className="mt-1 text-sm text-body">{c.body}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function SurfaceGrid() {
  return (
    <section className="mx-auto max-w-6xl px-4 py-16 md:px-6">
      <p className="mb-3 text-xs font-semibold uppercase tracking-[0.18em] text-navy">
        Three ways in
      </p>
      <h2 className="max-w-3xl font-display text-3xl font-medium leading-tight tracking-tight text-ink md:text-4xl">
        Ask, remix, or drop to raw SQL — same guarded corpus behind every
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
              <p className="font-mono text-[11px] uppercase tracking-wider text-muted">
                {s.label}
              </p>
              <h3 className="mt-1 font-display text-xl font-medium text-ink">
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
          <h2 className="font-display text-3xl font-medium tracking-tight text-ink">
            Trust is a feature, not a footnote.
          </h2>
          <p className="mt-3 max-w-md text-body">
            The corpus is Canadian federal open data. The stack is BigQuery,
            OpenAI, and a semantic index. The guardrails are non-negotiable.
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
      <div className="relative overflow-hidden rounded-2xl bg-navy p-8 text-white md:p-12">
        <span
          aria-hidden="true"
          className="pointer-events-none absolute -right-20 -top-24 h-72 w-72 rounded-full"
          style={{
            background:
              "radial-gradient(circle, rgba(253,137,115,.85), rgba(255,191,101,.45) 55%, transparent 72%)",
          }}
        />
        <div className="relative">
          <h2 className="font-display text-2xl font-medium md:text-3xl">
            Try the chat + evidence rail
          </h2>
          <p className="mt-3 max-w-2xl text-white/85">
            Ask a question, watch the answer build with live citations, and
            click any card to trace it. It&rsquo;s a working prototype.
          </p>
          <div className="mt-6 flex flex-wrap gap-3">
            <Link
              href="/chat"
              className="rounded-md bg-white px-5 py-2.5 text-sm font-medium text-navy transition-transform hover:-translate-y-0.5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white"
            >
              Launch the demo
            </Link>
            <Link
              href="/datasets"
              className="rounded-md border border-white/40 px-5 py-2.5 text-sm font-medium text-white transition-colors hover:bg-white/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white"
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
        MapleQuery · Milestone-4 prototype · answers only from cited datasets.
      </div>
    </footer>
  );
}
