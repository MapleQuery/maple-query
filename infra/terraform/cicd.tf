# Spec: docs/product-specs/milestone-4/5.2-agent-service-deploy.md
#
# Continuous-deployment plumbing for the agent-service:
#
# 1. Artifact Registry Docker repo (us-central1) — image target for CI.
# 2. Workload Identity Federation — GitHub Actions in this repo trades
#    its OIDC token for short-lived GCP credentials, no long-lived JSON
#    keys.
# 3. CD service account — impersonated by GitHub Actions; scoped to
#    push images + update the Cloud Run revision + impersonate the
#    runtime SA.
#
# This is provisioned by a one-time `terraform apply` from an operator's
# box. From that point on the workflow at
# .github/workflows/deploy-agent-service.yml uses these resources.

variable "github_owner" {
  description = "GitHub org or username that owns the maple-query repo. WIF binds pool→repo, so only pushes from this owner's repo can impersonate the CD SA."
  type        = string
  default     = "MapleQuery"
}

variable "github_repo" {
  description = "GitHub repo name (without owner). Combined with github_owner for the WIF attribute condition."
  type        = string
  default     = "maple-query"
}

# ── Required APIs ──────────────────────────────────────────────────
# The APIs below aren't managed by `google_project_service` in the
# existing repo, so operators enable them manually via `gcloud services
# enable` (see docs/services/agent-service.md — Bootstrap section).
#
# artifactregistry.googleapis.com  — Artifact Registry
# run.googleapis.com               — Cloud Run
# iamcredentials.googleapis.com    — WIF token exchange
# secretmanager.googleapis.com     — Secrets

# ── Artifact Registry ──────────────────────────────────────────────

resource "google_artifact_registry_repository" "agent_service" {
  project       = var.gcp_project_id
  location      = "us-central1"
  repository_id = "agent-service"
  description   = "MapleQuery agent-service container images. One repo, one image, tag per commit SHA."
  format        = "DOCKER"

  labels = {
    component  = "agent-service"
    managed_by = "terraform"
  }
}

resource "google_artifact_registry_repository_iam_member" "agent_service_puller" {
  # The runtime SA needs to pull its own image at revision-start time.
  project    = var.gcp_project_id
  location   = google_artifact_registry_repository.agent_service.location
  repository = google_artifact_registry_repository.agent_service.repository_id
  role       = "roles/artifactregistry.reader"
  member     = "serviceAccount:${google_service_account.agent_service.email}"
}

# ── Workload Identity Federation for GitHub Actions ────────────────

resource "google_iam_workload_identity_pool" "github" {
  project                   = var.gcp_project_id
  workload_identity_pool_id = "github-actions"
  display_name              = "GitHub Actions"
  description               = "Federation pool for GitHub Actions workflows in the ${var.github_owner}/${var.github_repo} repo. Enables OIDC-token-based auth with no long-lived key material."
}

resource "google_iam_workload_identity_pool_provider" "github" {
  project                            = var.gcp_project_id
  workload_identity_pool_id          = google_iam_workload_identity_pool.github.workload_identity_pool_id
  workload_identity_pool_provider_id = "github"
  display_name                       = "GitHub OIDC"
  description                        = "OIDC provider for GitHub Actions. `attribute_condition` locks the trust to this repo only."

  # attribute_condition is REQUIRED when audiences aren't set; without
  # it any GitHub workflow anywhere could exchange tokens against this
  # provider. Restrict to our repo.
  attribute_condition = "assertion.repository == \"${var.github_owner}/${var.github_repo}\""

  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.repository" = "assertion.repository"
    "attribute.ref"        = "assertion.ref"
    "attribute.actor"      = "assertion.actor"
  }

  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}

# ── CD service account ─────────────────────────────────────────────

resource "google_service_account" "cd_agent_service" {
  project      = var.gcp_project_id
  account_id   = "sa-cd-agent-service"
  display_name = "MapleQuery CD — agent-service"
  description  = "Impersonated by GitHub Actions via WIF. Pushes images to Artifact Registry and updates the Cloud Run agent-service revision."
}

# Allow the WIF pool (scoped to this repo by attribute_condition above)
# to impersonate the CD SA. `principalSet://` grants to every identity
# matching `attribute.repository == owner/repo`.
resource "google_service_account_iam_member" "cd_wif_impersonation" {
  service_account_id = google_service_account.cd_agent_service.name
  role               = "roles/iam.workloadIdentityUser"
  member = format(
    "principalSet://iam.googleapis.com/%s/attribute.repository/%s/%s",
    google_iam_workload_identity_pool.github.name,
    var.github_owner,
    var.github_repo,
  )
}

# Push images to Artifact Registry.
resource "google_artifact_registry_repository_iam_member" "cd_pusher" {
  project    = var.gcp_project_id
  location   = google_artifact_registry_repository.agent_service.location
  repository = google_artifact_registry_repository.agent_service.repository_id
  role       = "roles/artifactregistry.writer"
  member     = "serviceAccount:${google_service_account.cd_agent_service.email}"
}

# Update the Cloud Run revision. `roles/run.admin` covers `services
# update` + reading service state; `run.developer` would work too but
# admin keeps the door open for future revision-tag / traffic-split
# work without a permission bump.
resource "google_project_iam_member" "cd_run_admin" {
  project = var.gcp_project_id
  role    = "roles/run.admin"
  member  = "serviceAccount:${google_service_account.cd_agent_service.email}"
}

# `run services update` sets the container image, which Cloud Run
# validates by pulling with the *runtime* SA — but the update itself
# needs the CD SA to `actAs` the runtime SA. Without this bind the
# update returns "iam.serviceAccounts.actAs" denied.
resource "google_service_account_iam_member" "cd_actas_runtime" {
  service_account_id = google_service_account.agent_service.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.cd_agent_service.email}"
}

# ── Outputs ────────────────────────────────────────────────────────

output "cd_workload_identity_provider" {
  description = "Full WIF provider path. Paste into GitHub repo variable MQAGENT_WIF_PROVIDER."
  value       = google_iam_workload_identity_pool_provider.github.name
}

output "cd_service_account_email" {
  description = "CD SA email. Paste into GitHub repo variable MQAGENT_CD_SERVICE_ACCOUNT."
  value       = google_service_account.cd_agent_service.email
}

output "agent_service_image_base" {
  description = "Artifact Registry base path for the agent-service image. CI concatenates `:<sha>` to produce the full image URL."
  value       = "${google_artifact_registry_repository.agent_service.location}-docker.pkg.dev/${var.gcp_project_id}/${google_artifact_registry_repository.agent_service.repository_id}/agent-service"
}
