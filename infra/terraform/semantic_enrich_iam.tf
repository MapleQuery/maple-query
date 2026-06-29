# Identity for the semantic enrichment pipeline (text generation +
# embedding + load). Pairs with semantic_iam.tf which holds the
# BQ dataset-scoped grants.
#
# No GCS grant is configured. Enrichment is BQ-in/BQ-out: it reads
# sampled rows from raw.rows (already-parsed JSON) and writes to the
# semantic.* tables. The source CSV bytes are not on the read path.

resource "google_service_account" "semantic_enrich" {
  project      = var.gcp_project_id
  account_id   = "sa-semantic-enrich"
  display_name = "MapleQuery semantic enricher"
  description  = "Identity for the semantic enrichment pipeline (text generation, embedding, and load). Reads raw.documents + raw.rows for sampling; writes semantic.datasets + semantic.columns."
}

# Required at project level for the SA to start BQ jobs (load, query).
# Dataset-level dataEditor is necessary but not sufficient.
resource "google_project_iam_member" "semantic_enrich_job_user" {
  project = var.gcp_project_id
  role    = "roles/bigquery.jobUser"
  member  = "serviceAccount:${google_service_account.semantic_enrich.email}"
}
