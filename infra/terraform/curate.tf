# Normalize (M3): the curate service and the curated tables it owns.
# Design: docs/design/people-spine.md. Service: services/curate.
#
# Schemas live in schemas/curated_*.json, read by both this file and
# services/curate (schema-as-code, one source of truth).

resource "google_bigquery_table" "curated_people" {
  project             = var.gcp_project_id
  dataset_id          = google_bigquery_dataset.curated.dataset_id
  table_id            = "people"
  description         = "One row per person (federal politicians, from openparliament.ca). Key: person_id."
  deletion_protection = true
  clustering          = ["person_id"]
  schema              = file("${path.module}/schemas/curated_people.json")
}

resource "google_bigquery_table" "curated_person_names" {
  project             = var.gcp_project_id
  dataset_id          = google_bigquery_dataset.curated.dataset_id
  table_id            = "person_names"
  description         = "Name variants per person, normalised for linking. Key: (person_id, name_norm, origin)."
  deletion_protection = true
  clustering          = ["name_norm"]
  schema              = file("${path.module}/schemas/curated_person_names.json")
}

resource "google_bigquery_table" "curated_person_terms" {
  project             = var.gcp_project_id
  dataset_id          = google_bigquery_dataset.curated.dataset_id
  table_id            = "person_terms"
  description         = "House of Commons terms per person. Key: term_id. Riding is context, never a join key."
  deletion_protection = true
  clustering          = ["person_id"]
  schema              = file("${path.module}/schemas/curated_person_terms.json")
}

resource "google_bigquery_table" "curated_person_contributions" {
  project             = var.gcp_project_id
  dataset_id          = google_bigquery_dataset.curated.dataset_id
  table_id            = "person_contributions"
  description         = "Elections Canada contributions to candidates and contestants, aggregated and linked to people. Individual donors are never named. Key: contribution_key."
  deletion_protection = true
  clustering          = ["person_id", "status"]
  schema              = file("${path.module}/schemas/curated_person_contributions.json")
}

# ── sa-curate ───────────────────────────────────────────────────────

resource "google_service_account" "curate" {
  project      = var.gcp_project_id
  account_id   = "sa-curate"
  display_name = "MapleQuery curate (Normalize, M3)"
  description  = "Builds curated.* from openparliament.ca and raw GCS archives. Reads raw/ objects; writes the curated dataset only."
}

resource "google_bigquery_dataset_iam_member" "curate_curated_editor" {
  project    = var.gcp_project_id
  dataset_id = google_bigquery_dataset.curated.dataset_id
  role       = "roles/bigquery.dataEditor"
  member     = "serviceAccount:${google_service_account.curate.email}"
}

resource "google_project_iam_member" "curate_job_user" {
  project = var.gcp_project_id
  role    = "roles/bigquery.jobUser"
  member  = "serviceAccount:${google_service_account.curate.email}"
}

# Read gs://maplequery-raw/raw/ only (the Elections Canada archive);
# same prefix scoping as sa-warehouse-load.
resource "google_storage_bucket_iam_member" "curate_raw_reader" {
  bucket = google_storage_bucket.raw.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.curate.email}"

  condition {
    title       = "raw prefix only"
    description = "Read access does not extend to quarantine/ or sandbox/."
    expression  = <<-EOT
      resource.name.startsWith("projects/_/buckets/${google_storage_bucket.raw.name}/objects/raw/")
    EOT
  }
}

# The agent reads curated.* through its person_record tool.
resource "google_bigquery_dataset_iam_member" "agent_service_curated_viewer" {
  project    = var.gcp_project_id
  dataset_id = google_bigquery_dataset.curated.dataset_id
  role       = "roles/bigquery.dataViewer"
  member     = "serviceAccount:${google_service_account.agent_service.email}"
}
