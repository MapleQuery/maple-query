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
