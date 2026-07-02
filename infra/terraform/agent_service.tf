# Spec: docs/product-specs/milestone-4/5.2-agent-service-deploy.md
#
# Cloud Run service that wraps the 5.1 agent loop as an HTTP surface.
# One region (us-central1) — same as the BQ US multi-region so reads
# stay same-region-cheap. min_instances=1 during the demo period to
# avoid cold starts on the first question of a session.

variable "agent_service_image" {
  description = "Fully qualified container image for agent-service. Only consumed by the FIRST terraform apply that creates the revision; subsequent deploys land the image via `gcloud run services update` from the CI workflow and Terraform ignores the drift (see `lifecycle.ignore_changes` below). The `placeholder` default keeps `terraform test` (plan-mode) working without a real image."
  type        = string
  default     = "us-central1-docker.pkg.dev/PROJECT_ID/agent-service/agent-service:placeholder"
}

variable "agent_service_cors_origins" {
  description = "Comma-separated CORS allow-list. Includes the production Vercel URL, preview URLs, and localhost for dev. Ends up in MQAGENT_CORS_ORIGINS."
  type        = string
  default     = "https://maple-query.vercel.app,http://localhost:3000"
}

# ── Service account ────────────────────────────────────────────────

resource "google_service_account" "agent_service" {
  project      = var.gcp_project_id
  account_id   = "sa-agent-service"
  display_name = "MapleQuery agent service"
  description  = "Identity for the Cloud Run agent-service. Reads raw.* + semantic.* via VECTOR_SEARCH; runs guard-approved SELECTs against BQ; consumes OpenAI + the shared bearer token from Secret Manager."
}

# ── BQ grants ──────────────────────────────────────────────────────
# raw.* + semantic.* read-only. Project-level jobUser to run queries.

resource "google_bigquery_dataset_iam_member" "agent_service_raw_viewer" {
  project    = var.gcp_project_id
  dataset_id = google_bigquery_dataset.raw.dataset_id
  role       = "roles/bigquery.dataViewer"
  member     = "serviceAccount:${google_service_account.agent_service.email}"
}

resource "google_bigquery_dataset_iam_member" "agent_service_semantic_viewer" {
  project    = var.gcp_project_id
  dataset_id = google_bigquery_dataset.semantic.dataset_id
  role       = "roles/bigquery.dataViewer"
  member     = "serviceAccount:${google_service_account.agent_service.email}"
}

resource "google_project_iam_member" "agent_service_job_user" {
  project = var.gcp_project_id
  role    = "roles/bigquery.jobUser"
  member  = "serviceAccount:${google_service_account.agent_service.email}"
}

resource "google_project_iam_member" "agent_service_log_writer" {
  project = var.gcp_project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.agent_service.email}"
}

# ── Secrets ────────────────────────────────────────────────────────
# The service reads four secrets:
#   - openai-api-key (shared with the semantic-enrich reembed pipeline).
#   - mqagent-api-token (bearer token shared with the FE bundle).
#   - braintrust-api-key (LLM tracing; the service degrades to no-op
#     traces when the value is empty).
#   - posthog-api-key (server-side product analytics; capture calls
#     no-op when the value is empty).
#
# Secrets themselves are created here so a fresh apply provisions them;
# operators manage versions manually via `gcloud secrets versions add`.

resource "google_secret_manager_secret" "openai_api_key" {
  project   = var.gcp_project_id
  secret_id = "openai-api-key"

  replication {
    auto {}
  }

  labels = {
    component  = "agent-service"
    managed_by = "terraform"
  }
}

resource "google_secret_manager_secret" "mqagent_api_token" {
  project   = var.gcp_project_id
  secret_id = "mqagent-api-token"

  replication {
    auto {}
  }

  labels = {
    component  = "agent-service"
    managed_by = "terraform"
  }
}

resource "google_secret_manager_secret" "braintrust_api_key" {
  project   = var.gcp_project_id
  secret_id = "braintrust-api-key"

  replication {
    auto {}
  }

  labels = {
    component  = "agent-service"
    managed_by = "terraform"
  }
}

resource "google_secret_manager_secret" "posthog_api_key" {
  project   = var.gcp_project_id
  secret_id = "posthog-api-key"

  replication {
    auto {}
  }

  labels = {
    component  = "agent-service"
    managed_by = "terraform"
  }
}

resource "google_secret_manager_secret_iam_member" "agent_service_openai_key_accessor" {
  project   = var.gcp_project_id
  secret_id = google_secret_manager_secret.openai_api_key.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.agent_service.email}"
}

resource "google_secret_manager_secret_iam_member" "agent_service_api_token_accessor" {
  project   = var.gcp_project_id
  secret_id = google_secret_manager_secret.mqagent_api_token.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.agent_service.email}"
}

resource "google_secret_manager_secret_iam_member" "agent_service_braintrust_key_accessor" {
  project   = var.gcp_project_id
  secret_id = google_secret_manager_secret.braintrust_api_key.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.agent_service.email}"
}

resource "google_secret_manager_secret_iam_member" "agent_service_posthog_key_accessor" {
  project   = var.gcp_project_id
  secret_id = google_secret_manager_secret.posthog_api_key.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.agent_service.email}"
}

# ── Cloud Run service ──────────────────────────────────────────────

resource "google_cloud_run_v2_service" "agent_service" {
  project  = var.gcp_project_id
  name     = "agent-service"
  location = "us-central1"
  ingress  = "INGRESS_TRAFFIC_ALL"

  # A destroy on a serving revision is a demo-visible outage — require
  # an explicit `-target` to nuke it.
  deletion_protection = true

  template {
    service_account                  = google_service_account.agent_service.email
    max_instance_request_concurrency = 80
    # Loop's own 60s wall-clock timeout wins by design; this 5-minute
    # cap is the backstop so a wedged tool call can't hold a slot
    # forever.
    timeout = "300s"

    scaling {
      # Avoid cold starts on the first question of every demo session.
      # ~$15/mo idle cost; drop to 0 after the demo period.
      min_instance_count = 1
      max_instance_count = 10
    }

    containers {
      image = var.agent_service_image

      resources {
        limits = {
          cpu    = "2"
          memory = "2Gi"
        }
        # Fluid CPU stays hot during requests only; matches the I/O-bound
        # nature of the loop (mostly waiting on OpenAI + BQ).
        cpu_idle          = true
        startup_cpu_boost = true
      }

      ports {
        container_port = 8080
      }

      env {
        name  = "MQAGENT_GCP_PROJECT_ID"
        value = var.gcp_project_id
      }

      env {
        name  = "MQAGENT_CORS_ORIGINS"
        value = var.agent_service_cors_origins
      }

      env {
        name = "MQAGENT_OPENAI_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.openai_api_key.secret_id
            version = "latest"
          }
        }
      }

      env {
        name = "MQAGENT_API_TOKEN"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.mqagent_api_token.secret_id
            version = "latest"
          }
        }
      }

      env {
        name = "BRAINTRUST_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.braintrust_api_key.secret_id
            version = "latest"
          }
        }
      }

      env {
        name = "POSTHOG_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.posthog_api_key.secret_id
            version = "latest"
          }
        }
      }

      startup_probe {
        http_get {
          path = "/readyz"
        }
        initial_delay_seconds = 10
        timeout_seconds       = 5
        period_seconds        = 5
        failure_threshold     = 6
      }

      liveness_probe {
        http_get {
          path = "/healthz"
        }
        period_seconds    = 30
        timeout_seconds   = 3
        failure_threshold = 3
      }
    }
  }

  # Wait until the SA has BQ + Secret access before Cloud Run starts
  # pulling the image — otherwise the first startup_probe against
  # /readyz can flap while IAM propagates.
  depends_on = [
    google_bigquery_dataset_iam_member.agent_service_raw_viewer,
    google_bigquery_dataset_iam_member.agent_service_semantic_viewer,
    google_project_iam_member.agent_service_job_user,
    google_secret_manager_secret_iam_member.agent_service_openai_key_accessor,
    google_secret_manager_secret_iam_member.agent_service_api_token_accessor,
    google_secret_manager_secret_iam_member.agent_service_braintrust_key_accessor,
    google_secret_manager_secret_iam_member.agent_service_posthog_key_accessor,
  ]

  # CD updates the container image directly via `gcloud run services
  # update` (see .github/workflows/deploy-agent-service.yml). Ignoring
  # the drift here means the CI workflow can roll revisions without
  # requiring an out-of-band `terraform apply -var agent_service_image=…`
  # on every push. Structural changes (concurrency, scaling, probes,
  # env vars) still go through Terraform.
  lifecycle {
    ignore_changes = [
      template[0].containers[0].image,
      client,
      client_version,
    ]
  }
}

# Allow unauthenticated invocations. The bearer-token check inside the
# app is what actually gates traffic; Cloud Run's own IAM would 401
# every FE call otherwise (browsers can't present a Google IAM token).
resource "google_cloud_run_v2_service_iam_member" "agent_service_public" {
  project  = var.gcp_project_id
  location = google_cloud_run_v2_service.agent_service.location
  name     = google_cloud_run_v2_service.agent_service.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# ── Outputs ────────────────────────────────────────────────────────

output "agent_service_url" {
  description = "Public URL for the Cloud Run agent-service. FE points at this."
  value       = google_cloud_run_v2_service.agent_service.uri
}

output "agent_service_sa_email" {
  description = "Service account email for the agent-service. Referenced by ad-hoc IAM audits."
  value       = google_service_account.agent_service.email
}
