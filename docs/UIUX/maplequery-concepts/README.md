<<<<<<< HEAD
# MapleQuery

Open government data, made queryable.

MapleQuery ingests public datasets from CKAN-style open-data portals
(Government of Canada, UK, etc.), lands them in BigQuery, and exposes
the corpus through an LLM agent that answers natural-language questions
with cited sources.

## Status

Pre-implementation. The repo currently contains only the documentation
scaffold for agent-driven development. No service code exists yet.

## Working on this repo

If you're an agent, **read [`AGENTS.md`](AGENTS.md) before any task.**

If you're a human:

- [`AGENTS.md`](AGENTS.md) — how agents should work in this repo.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — top-level pipeline and layering.
- [`docs/index.md`](docs/index.md) — full doc index.

## Setting up infra

We use Terraform to provision the project's cloud infrastructure (the
GCS bucket, service accounts, and — as later milestones land —
BigQuery datasets and the Cloud Run job). Everything lives in
`infra/terraform/`.

**You probably don't need to run this.** The infra is already up.
The steps below are only relevant if you're standing up a fresh GCP
project — team rotation, separate environment, DR rebuild, etc.

First-time setup:

```bash
# 1. Auth (gcloud CLI AND application-default — they are separate)
gcloud auth login
gcloud auth application-default login
gcloud config set project <your-project-id>

# 2. Enable the three APIs we need
gcloud services enable storage.googleapis.com iam.googleapis.com \
                       cloudresourcemanager.googleapis.com \
                       --project=<your-project-id>

# 3. Fill in your values (terraform.tfvars is gitignored)
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars
$EDITOR terraform.tfvars

# 4. Apply
terraform init
terraform plan -out=tfplan
terraform apply tfplan
```

To add an admin: edit the `admin_users` list in `terraform.tfvars` and
re-apply.

To run the test suite (no GCP calls, dummy token works offline):

```bash
GOOGLE_OAUTH_ACCESS_TOKEN=fake terraform -chdir=infra/terraform test
```

## License

TBD.
=======
# MapleQuery — Concept Prototypes

Static HTML/CSS prototypes for **MapleQuery**, a conversational BI platform that turns
Canadian government open data into plain-language answers and browsable tables.

These are clickable, front-end-only mockups (no backend). They explore two core tools
and three richer "concept" directions built on top of them.

## Quick start

No build step. Everything is static HTML styled with the Tailwind CDN and Google Fonts,
so it needs an internet connection to render but nothing to install.

- **Easiest:** open `index.html` in any modern browser.
- **Or serve locally** (avoids any file:// quirks):

  ```bash
  python3 -m http.server 8000
  # then visit http://localhost:8000
  ```

Start at **`index.html`** — it's the hub that links to every other page.

## Page map

`index.html` is the entry point. Each page below links back to it, and each concept
demo links to its own "About" landing page.

| Page | Role | Description |
|------|------|-------------|
| `index.html` | **Hub** | "Choose a Concept" — links to all tools and concepts |
| `simple-chat.html` | Core tool | **Ask** — a friendly RAG chatbot that returns clear, sourced answers |
| `data-viewer.html` | Core tool | **Data viewer** — browse official datasets as clean, searchable tables |
| `concept-1-landing.html` | Concept 1 landing | Intro for "Chat + evidence rail" |
| `concept-1-chat-evidence.html` | Concept 1 demo | Conversation with a persistent source panel; every claim is a traceable footnote |
| `concept-2-landing.html` | Concept 2 landing | Intro for "Notebook workspace" |
| `concept-2-notebook.html` | Concept 2 demo | A document of prose + live query blocks that promotes into a sourced report |
| `concept-3-landing.html` | Concept 3 landing | Intro for "Split explorer" |
| `concept-3-explorer.html` | Concept 3 demo | NL chat beside a spreadsheet, with each transform shown as an editable step |
| `maplequery-logo.html` | Standalone | Logo / wordmark exploration (not linked from the hub) |

## Navigation

```
index.html  (hub)
├─ Core tools
│  ├─ simple-chat.html
│  └─ data-viewer.html
└─ Concept explorations  (landing → demo)
   ├─ concept-1-landing.html → concept-1-chat-evidence.html
   ├─ concept-2-landing.html → concept-2-notebook.html
   └─ concept-3-landing.html → concept-3-explorer.html

maplequery-logo.html   (standalone, not in nav)
```

## Tech notes

- Plain HTML with [Tailwind CSS](https://tailwindcss.com/) via CDN — no bundler or package manager.
- Fonts (Fraunces, Inter, Fira Code, JetBrains Mono) loaded from Google Fonts.
- Interactions in the demos are prototyped in vanilla JavaScript inline in each file.
- Respects `prefers-reduced-motion` for accessibility.
>>>>>>> c7e6690 (docs: Add README with page map and quick start)
