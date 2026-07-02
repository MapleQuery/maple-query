import Link from "next/link";
import { ArrowUpRight } from "lucide-react";
import { Reveal } from "./reveal";

interface ExampleQuestion {
  question: string;
  angle: string;
  tag: string;
}

/**
 * Six ways in. Each chip is a real question the corpus can answer, and
 * clicking through drops the visitor into a fresh chat with the composer
 * pre-filled — no blank canvas problem.
 */
const QUESTIONS: ExampleQuestion[] = [
  {
    question:
      "Which departments spent the most on IT consulting in 2023?",
    angle: "Contract spend",
    tag: "Fiscal 2023",
  },
  {
    question: "How did housing grant approvals shift after 2020?",
    angle: "Program outcomes",
    tag: "Trend",
  },
  {
    question:
      "Which provinces received the largest federal transfers last year?",
    angle: "Transfers",
    tag: "Provincial",
  },
  {
    question:
      "What is the average PR application processing time by year?",
    angle: "Immigration",
    tag: "Processing time",
  },
  {
    question:
      "Compare defence procurement spend across the last three fiscal years.",
    angle: "Procurement",
    tag: "Multi-year",
  },
  {
    question:
      "Which grant programs had the highest rejection rates in 2022?",
    angle: "Program rigour",
    tag: "Rejections",
  },
];

export function ExampleQuestions() {
  return (
    <section className="border-t border-hairline">
      <div className="mx-auto max-w-6xl px-4 py-20 md:px-6 md:py-28">
        <Reveal>
          <p className="mb-3 text-xs font-semibold uppercase tracking-[0.18em] text-navy">
            Try one on
          </p>
          <h2 className="max-w-3xl text-3xl font-semibold leading-tight tracking-tight text-ink md:text-4xl">
            Real questions the corpus can answer.
          </h2>
          <p className="mt-3 max-w-xl text-body">
            Click any card. MapleQuery opens with the question pre-filled;
            you decide when to send.
          </p>
        </Reveal>

        <div className="mt-10 grid gap-3 md:grid-cols-2 lg:grid-cols-3">
          {QUESTIONS.map((q, i) => (
            <Reveal key={q.question} delayMs={60 + i * 40}>
              <Link
                href={`/chat?q=${encodeURIComponent(q.question)}`}
                className="group flex h-full flex-col justify-between gap-6 rounded-xl border border-hairline bg-white/70 p-5 backdrop-blur-sm transition-all hover:-translate-y-0.5 hover:border-navy hover:bg-white hover:shadow-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-navy"
              >
                <div>
                  <div className="mb-3 flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.14em] text-muted">
                    <span className="rounded bg-surface-soft px-1.5 py-0.5 text-navy">
                      {q.angle}
                    </span>
                    <span>{q.tag}</span>
                  </div>
                  <p className="text-[15px] font-medium leading-snug text-ink">
                    {q.question}
                  </p>
                </div>
                <span className="inline-flex items-center gap-1 text-xs font-medium text-navy group-hover:underline">
                  Open in chat
                  <ArrowUpRight className="h-3.5 w-3.5 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
                </span>
              </Link>
            </Reveal>
          ))}
        </div>
      </div>
    </section>
  );
}
