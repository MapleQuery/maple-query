# Pins the Cloud Run agent-service infra. Drift fails CI.
# Run offline with a dummy token:
#   GOOGLE_OAUTH_ACCESS_TOKEN=fake terraform test

variables {
  gcp_project_id = "maplequery-test"
  admin_users    = ["alice@example.com", "bob@example.com"]
}

run "cloud_run_service_shape" {
  command = plan

  assert {
    condition     = google_cloud_run_v2_service.agent_service.name == "agent-service"
    error_message = "Cloud Run service must be named 'agent-service' (FE URL contract)."
  }

  assert {
    condition     = google_cloud_run_v2_service.agent_service.location == "us-central1"
    error_message = "Cloud Run service must be in us-central1 (co-located with BQ US multi-region)."
  }

  assert {
    condition     = google_cloud_run_v2_service.agent_service.ingress == "INGRESS_TRAFFIC_ALL"
    error_message = "Cloud Run service must accept all ingress; the FE calls it directly from the browser."
  }

  assert {
    condition     = google_cloud_run_v2_service.agent_service.deletion_protection == true
    error_message = "Cloud Run service must have deletion protection (a destroy is a demo-visible outage)."
  }
}

run "cloud_run_scaling_and_concurrency" {
  command = plan

  assert {
    condition = alltrue([
      for t in google_cloud_run_v2_service.agent_service.template :
      t.max_instance_request_concurrency == 80
    ])
    error_message = "Per-instance concurrency must be 80 — matches Cloud Run default for I/O-bound loops."
  }

  # min_instances=1 keeps a warm instance during the demo period so the
  # first question of every session doesn't eat a 5-10s cold start.
  assert {
    condition = alltrue([
      for t in google_cloud_run_v2_service.agent_service.template :
      alltrue([for s in t.scaling : s.min_instance_count == 1])
    ])
    error_message = "min_instance_count must be 1 during the M4 demo period."
  }

  assert {
    condition = alltrue([
      for t in google_cloud_run_v2_service.agent_service.template :
      alltrue([for s in t.scaling : s.max_instance_count == 10])
    ])
    error_message = "max_instance_count must be 10 (caps runaway cost)."
  }
}

run "cloud_run_public_invoker" {
  command = plan

  assert {
    condition     = google_cloud_run_v2_service_iam_member.agent_service_public.member == "allUsers"
    error_message = "Cloud Run invoker must be allUsers (bearer-token check inside the app gates traffic; browsers can't present IAM tokens)."
  }

  assert {
    condition     = google_cloud_run_v2_service_iam_member.agent_service_public.role == "roles/run.invoker"
    error_message = "Public invoker binding must be roles/run.invoker."
  }
}

run "agent_sa_bq_grants" {
  command = plan

  assert {
    condition     = google_service_account.agent_service.account_id == "sa-agent-service"
    error_message = "Agent SA must be sa-agent-service (referenced by ad-hoc IAM audits)."
  }

  assert {
    condition     = google_bigquery_dataset_iam_member.agent_service_raw_viewer.role == "roles/bigquery.dataViewer"
    error_message = "Agent SA must have dataViewer on raw."
  }

  assert {
    condition     = google_bigquery_dataset_iam_member.agent_service_semantic_viewer.role == "roles/bigquery.dataViewer"
    error_message = "Agent SA must have dataViewer on semantic."
  }

  assert {
    condition     = google_project_iam_member.agent_service_job_user.role == "roles/bigquery.jobUser"
    error_message = "Agent SA must have jobUser at the project level to run BQ queries."
  }
}

run "agent_sa_secret_access" {
  command = plan

  assert {
    condition     = google_secret_manager_secret.openai_api_key.secret_id == "openai-api-key"
    error_message = "OpenAI key secret must be 'openai-api-key'."
  }

  assert {
    condition     = google_secret_manager_secret.mqagent_api_token.secret_id == "mqagent-api-token"
    error_message = "Bearer token secret must be 'mqagent-api-token'."
  }

  assert {
    condition     = google_secret_manager_secret_iam_member.agent_service_openai_key_accessor.role == "roles/secretmanager.secretAccessor"
    error_message = "Agent SA must have secretAccessor on the OpenAI key."
  }

  assert {
    condition     = google_secret_manager_secret_iam_member.agent_service_api_token_accessor.role == "roles/secretmanager.secretAccessor"
    error_message = "Agent SA must have secretAccessor on the bearer token."
  }
}
