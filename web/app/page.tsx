import Link from "next/link";
import { ArrowRight, ShieldCheck, Sparkles, Zap } from "lucide-react";
import { CorpusStats } from "@/components/marketing/corpus-stats";
import { EvidenceDemo } from "@/components/marketing/evidence-demo";
import { ExampleQuestions } from "@/components/marketing/example-questions";
import { ExplorerDemo } from "@/components/marketing/explorer-demo";
import { MiniChat } from "@/components/marketing/mini-chat";
import { NotebookDemo } from "@/components/marketing/notebook-demo";
import { Reveal } from "@/components/marketing/reveal";

export default function LandingPage() {
  return (
    <>
      <Hero />
      <FeatureShowcase />
      <TrustBand />
      <ExampleQuestions />
      <SiteFooter />
    </>
  );
}

function Hero() {
  return (
    <section className="relative overflow-hidden">
      <div className="relative mx-auto grid max-w-6xl items-center gap-10 px-4 py-20 md:px-6 md:py-24 lg:grid-cols-[1.1fr_0.9fr]">
        <div>
          <h1 className="max-w-3xl text-4xl font-semibold leading-[1.04] tracking-tight text-ink sm:text-5xl md:text-6xl">
            Ask hard questions of government data. Get answers you can
            cite.
          </h1>
          <p className="mt-6 max-w-xl text-lg leading-relaxed text-body">
            MapleQuery turns fragmented Canadian open data into a
            plain-language conversation. Every figure carries a footnote
            that traces straight back to the original record.
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

          <CorpusStats />
        </div>

        <div className="relative">
          <MiniChat />
        </div>
      </div>
    </section>
  );
}

function FeatureShowcase() {
  return (
    <div className="border-t border-hairline">
      <Feature
        kicker="Chat"
        eyebrowIcon={<Sparkles className="h-3.5 w-3.5" />}
        title="Ask in plain language. Watch the answer build itself."
        body="Every question runs through the same guarded loop: retrieve relevant datasets, generate SQL, dry-run for cost, execute, cite. The evidence rail shows the trace as it happens — never a black box."
        cta={{ href: "/chat", label: "Open the chat" }}
        visual={<EvidenceDemo />}
      />

      <Feature
        reverse
        kicker="Explorer"
        eyebrowIcon={<ShieldCheck className="h-3.5 w-3.5" />}
        title="Edit the SQL. The guard still has your back."
        body="Drop into the generated SQL, tweak it, and re-run. Every statement passes a static guard: SELECT-only, allow-listed datasets, cost dry-runs, and an automatic LIMIT clamp. No footguns."
        cta={{ href: "/explorer", label: "Open the explorer" }}
        visual={<ExplorerDemo />}
      />

      <Feature
        kicker="Notebook"
        eyebrowIcon={<Zap className="h-3.5 w-3.5" />}
        title="Prose and live queries in one document."
        body="Draft a brief that stays honest as the data changes. Queries live inline; the prose around them updates as the numbers do. Export the whole thing when you're ready to share."
        cta={{ href: "/notebook", label: "Open the notebook" }}
        visual={<NotebookDemo />}
      />
    </div>
  );
}

function Feature({
  kicker,
  eyebrowIcon,
  title,
  body,
  cta,
  visual,
  reverse = false,
}: {
  kicker: string;
  eyebrowIcon: React.ReactNode;
  title: string;
  body: string;
  cta: { href: string; label: string };
  visual: React.ReactNode;
  reverse?: boolean;
}) {
  return (
    <section className="border-b border-hairline">
      <div className="mx-auto grid max-w-6xl items-center gap-10 px-4 py-20 md:grid-cols-2 md:px-6 md:py-28">
        <Reveal className={reverse ? "md:order-2" : ""}>
          <div className="mb-3 inline-flex items-center gap-1.5 rounded-full border border-hairline bg-white px-2.5 py-1 font-mono text-[10px] font-semibold uppercase tracking-[0.14em] text-navy">
            {eyebrowIcon}
            {kicker}
          </div>
          <h2 className="max-w-lg text-3xl font-semibold leading-tight tracking-tight text-ink md:text-4xl">
            {title}
          </h2>
          <p className="mt-4 max-w-lg text-body">{body}</p>
          <Link
            href={cta.href}
            className="mt-6 inline-flex items-center gap-1.5 text-sm font-medium text-navy hover:underline"
          >
            {cta.label}
            <ArrowRight className="h-4 w-4" />
          </Link>
        </Reveal>

        <Reveal delayMs={120} className={reverse ? "md:order-1" : ""}>
          {visual}
        </Reveal>
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
    <section className="border-b border-hairline bg-surface-soft/40 backdrop-blur-[2px]">
      <div className="mx-auto grid max-w-6xl gap-10 px-4 py-16 md:grid-cols-2 md:px-6">
        <Reveal>
          <h2 className="text-3xl font-semibold tracking-tight text-ink">
            Trust is a feature, not a footnote.
          </h2>
          <p className="mt-3 max-w-md text-body">
            The corpus is Canadian federal open data. Every answer traces
            back to a published record, and the guardrails are
            non-negotiable.
          </p>
        </Reveal>
        <Reveal delayMs={120}>
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
        </Reveal>
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
