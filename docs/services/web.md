# web

Next.js 15 App Router frontend for MapleQuery. Deployed to Vercel, hits the
agent-service Cloud Run backend over SSE + REST. Stateless client; all
conversation, notebook, and explorer state lives in the browser's
`localStorage`.

Repo location: `/web` (repo root, not under `services/`; this isn't a
Python service and doesn't fit the services layout).

---

## Layout

```
web/
├── app/                 # App Router entrypoints
│   ├── layout.tsx       # Root layout + font wiring + header
│   ├── page.tsx         # Landing
│   ├── chat/            # /chat and /chat/[conversationId]
│   ├── notebook/        # /notebook and /notebook/[notebookId]
│   ├── explorer/        # /explorer (single session in localStorage)
│   └── datasets/        # /datasets list + /datasets/[packageId] detail
├── components/
│   ├── ui/              # shadcn-style primitives (Button, Input, Toast, …)
│   ├── layout/          # Site header
│   ├── evidence/        # Shared across all three surfaces:
│   │                    # Message, EvidenceRail, DatasetCard, DatasetChip,
│   │                    # ColumnList, SqlBlock, RowsTable, CostBadge
│   ├── chat/            # ChatContainer + composer + conversation switcher
│   ├── notebook/        # NotebookContainer + Markdown/PDF export
│   └── explorer/        # ExplorerContainer + step chain
├── lib/
│   ├── api.ts           # REST wrappers (datasets, columns, /sql/run)
│   ├── sse.ts           # POST-capable SSE via @microsoft/fetch-event-source,
│   │                    # dispatches typed AgentEvent unions after zod-validating
│   ├── types.ts         # zod schemas mirroring 5.1's event dataclasses
│   ├── storage.ts       # localStorage with per-collection LRU index (max 50)
│   │                    # + quota-exceeded eviction
│   ├── dataset-titles.ts # package_id → title cache + backfill hook
│   ├── history.ts       # append helpers + 200-message wall
│   ├── highlight.ts     # shiki singleton for SQL highlighting
│   ├── config.ts        # NEXT_PUBLIC_* env reading + bearer header builder
│   └── utils.ts         # cn, uuid, formatters
└── styles/globals.css   # Tailwind base + concept palette + shiki block styles
```

---

## Design system

Palette lifted from `docs/UIUX/maplequery-concepts/concept-1-*.html`. Every
color token, font, and radius should match those prototypes; the frontend's
job is to make them functional, not to redesign.

Tokens live in `tailwind.config.ts` (`canvas`, `surface-soft`, `surface-card`,
`ink`, `body`, `muted`, `hairline`, `coral`, `coral-active`, `navy`, `polar`,
`teal`, `amber`, `success`, `error`).

One typeface across the whole app: **Inter**, loaded via `next/font/google`.
The Tailwind `font-display` and `font-mono` aliases are kept but resolve to
the same Inter variable so `font-mono` classes in existing markup keep
working without a rename pass. Weight and tabular-figure feature settings
carry the visual distinction that separate serif / mono families would.

`prefers-reduced-motion` is respected globally in `styles/globals.css`.

---

## Surfaces

### /chat · primary
- Two columns on `lg+`: message thread (left), evidence rail (right).
- Conversation switcher sidebar (LRU up to 50, backed by
  `mq:conversations:v1:*` in localStorage).
- SSE events dispatch through `useChatStream` (reducer keyed on event name).
- History is client-supplied per turn; the server owns compaction (5.1 §6).
- `/chat` (no id) redirects client-side to the most recent conversation or
  spawns a new one.

### /notebook · secondary
- Ordered list of prose (Markdown) and query (single-turn `/chat`) blocks.
- A single **Export** control with a format menu. Both formats build the
  same Markdown document — every block, including SQL fenced blocks and
  result tables (first 20 rows):
  - **Markdown** downloads it as `.md`.
  - **PDF** renders it with the same ReactMarkdown/remark-gfm pair used on
    screen into an off-screen iframe and calls `print()` on that iframe,
    so the output is real text with real pagination. It lands on the
    browser's print dialog, where the user picks "Save as PDF".
- Dataset references (the `Scoped to:` and `Datasets:` rows, and the
  `Sources:` line in an export) are named by title, not package UUID. See
  "Dataset titles" below.
- Re-running a query block clears its result and re-streams.

### /explorer · secondary
- Left column: prompt input + step chain (prompt cards + SQL cards).
- Right column: active step's SQL editor + rows table.
- Step 1 always comes from `/chat` (single-turn); subsequent SQL edits go
  through `/sql/run` directly.
- One active session per browser (localStorage key `explorer:current-v1`).

### / and /datasets · landing / corpus browser
- Static-ish surfaces built from `docs/UIUX/maplequery-concepts/concept-1-landing.html`
  and `data-viewer.html`.
- `/datasets` hits `GET /datasets?q=` for semantic search.
- `/datasets/[packageId]` also hits `GET /datasets/{id}/documents` for the
  "Source files" table — per-file open.canada.ca download links, with an
  "Enriched" badge on the representative document. The section hides
  itself when the fetch fails or returns no files.

---

## Dataset titles

`lib/dataset-titles.ts` keeps a `package_id → title` map so no surface has
to print a UUID at the user. It is filled for free from three places that
already carry titles: `datasets_ranked` stream frames, the `/datasets`
list, and the `/datasets/{id}` detail fetch. The map is cached in
localStorage under `mq:dataset-titles:v1` and read through
`useDatasetTitles(ids, known?)`.

An id still unknown after all that — a scope pinned from a suggestion, a
notebook saved before titles were recorded — is backfilled with one
`GET /datasets/{id}`, deduped across components and never retried once
answered. `known` lets a caller pass titles it already holds (a notebook
block stores its own in `packageTitles`), and those ids are never looked
up at all, so reopening a saved notebook costs no requests.

Every failure mode degrades to the raw id, which is what these surfaces
showed before.

---

## Configuration

Every env var is `NEXT_PUBLIC_*` because it needs to reach the browser. See
`.env.example` for the shape.

| Var                                  | Purpose                                  |
| ------------------------------------ | ---------------------------------------- |
| `NEXT_PUBLIC_MAPLEQUERY_API_BASE_URL`| Cloud Run agent-service URL, no trailing `/`. |
| `NEXT_PUBLIC_MAPLEQUERY_API_TOKEN`   | Bearer token from Secret Manager (see below). |
| `NEXT_PUBLIC_MAPLEQUERY_ENV`         | `prod` / `preview` / `dev` label.        |
| `NEXT_PUBLIC_POSTHOG_KEY`            | PostHog project key. Absent → capture no-ops, provider passes through. |
| `NEXT_PUBLIC_POSTHOG_HOST`           | PostHog host (default `https://us.i.posthog.com`). |

Retrieve the bearer token once for local dev:

```
gcloud secrets versions access latest \
  --secret=mqagent-api-token \
  --project=maplequery
```

Paste into `web/.env.local`. Do not commit `.env.local`.

Vercel injects env vars at build time; a deployed bundle is pinned to the
values present at build. Changing a var requires a redeploy.

---

## Local dev

Prereqs: Node ≥ 20, pnpm 10.

```
cd web
pnpm install
cp .env.example .env.local  # then fill in the bearer token
pnpm dev
```

App is at `http://localhost:3000`. Backend defaults to
`http://localhost:8080` when unset; either run agent-service locally or
point at the deployed Cloud Run URL.

Type-check and build:

```
pnpm typecheck
pnpm build
```

---

## Deployment

Vercel project: `maplequery-web`. Root directory `web/`. Framework preset:
Next.js. Build command: `pnpm build`. Install: `pnpm install --frozen-lockfile`.

- **Preview** on every PR touching `web/**`. URL pattern
  `https://maplequery-web-<hash>-<team>.vercel.app`.
- **Production** on merge to `main` touching `web/**`, at
  `https://maple-query.vercel.app`.
- **Rollback** via Deployments → prior deploy → Promote to Production.

The backend's `MQAGENT_CORS_ORIGINS` allow-list must include every FE origin
that will call the API from a browser. Production + `localhost:3000` are the
defaults; preview URLs are per-PR unique and require an ad-hoc allow-list
update (or point preview builds at a local backend).

---

## Testing posture

Tests are not exhaustive; the FE is a supporting surface and the loop is
verified independently. What ships:

- Type safety via `pnpm typecheck` (strict).
- Build validation via `pnpm build`.
- Manual smoke of chat, notebook, explorer, landing, and datasets against a
  live agent-service.

Formal unit/component/E2E coverage is a follow-up when the workflow demands
regression protection.

---

## What this is not

- Not a general BI tool. Rows tables only; no charts.
- Not shared or collaborative. One user, one browser, one localStorage scope.
- Not authenticated beyond the public bearer token baked into the bundle.
- Not mobile-optimized for chat / notebook / explorer. Landing and
  `/datasets` are responsive; the workbench surfaces are desktop-first.
- Not internationalized. English UI, bilingual corpus.
