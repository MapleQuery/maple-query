# Spec: docs/product-specs/milestone-2/3.1-bq-warehouse-infrastructure.md §6.3 - §6.5
#
# BQ-side grants. Isolated in its own file so widening review is a
# single-file diff. SA + cross-resource grants live in
# warehouse_load_iam.tf alongside the GCS condition.

# --- Warehouse loader (3.2 + 3.3): write to raw ----------------------

resource "google_project_iam_member" "warehouse_load_job_user" {
  project = var.gcp_project_id
  role    = "roles/bigquery.jobUser"
  member  = "serviceAccount:${google_service_account.warehouse_load.email}"
}

resource "google_bigquery_dataset_iam_member" "warehouse_load_raw_editor" {
  project    = var.gcp_project_id
  dataset_id = google_bigquery_dataset.raw.dataset_id
  role       = "roles/bigquery.dataEditor"
  member     = "serviceAccount:${google_service_account.warehouse_load.email}"
}

# --- Ingest reader (M3 agent, pre-wired): read raw + curated --------

resource "google_bigquery_dataset_iam_member" "ingest_reader_raw_viewer" {
  project    = var.gcp_project_id
  dataset_id = google_bigquery_dataset.raw.dataset_id
  role       = "roles/bigquery.dataViewer"
  member     = "serviceAccount:${google_service_account.ingest_reader.email}"
}

resource "google_bigquery_dataset_iam_member" "ingest_reader_curated_viewer" {
  project    = var.gcp_project_id
  dataset_id = google_bigquery_dataset.curated.dataset_id
  role       = "roles/bigquery.dataViewer"
  member     = "serviceAccount:${google_service_account.ingest_reader.email}"
}

resource "google_project_iam_member" "ingest_reader_job_user" {
  project = var.gcp_project_id
  role    = "roles/bigquery.jobUser"
  member  = "serviceAccount:${google_service_account.ingest_reader.email}"
}

# --- Human admins ---------------------------------------------------
# Small-team deviation from 2.1 §5.2 (groups-only); see variables.tf.

resource "google_bigquery_dataset_iam_member" "admin_raw" {
  for_each = var.admin_users

  project    = var.gcp_project_id
  dataset_id = google_bigquery_dataset.raw.dataset_id
  role       = "roles/bigquery.dataOwner"
  member     = "user:${each.value}"
}

resource "google_bigquery_dataset_iam_member" "admin_curated" {
  for_each = var.admin_users

  project    = var.gcp_project_id
  dataset_id = google_bigquery_dataset.curated.dataset_id
  role       = "roles/bigquery.dataOwner"
  member     = "user:${each.value}"
}
