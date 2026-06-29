# BQ-side grants for the semantic dataset. Isolated in its own file so
# widening review is a single-file diff. The semantic enricher service
# account and its project-level grant live in semantic_enrich_iam.tf.

# --- Semantic enricher: read raw for sampling -----------------------
# dataViewer is sufficient: the enrichment loop samples raw.rows and
# reads raw.documents metadata. It never writes back to raw.

resource "google_bigquery_dataset_iam_member" "semantic_enrich_raw_viewer" {
  project    = var.gcp_project_id
  dataset_id = google_bigquery_dataset.raw.dataset_id
  role       = "roles/bigquery.dataViewer"
  member     = "serviceAccount:${google_service_account.semantic_enrich.email}"
}

# --- Semantic enricher: write semantic -----------------------------
# dataEditor covers bq load jobs against both tables in the dataset.

resource "google_bigquery_dataset_iam_member" "semantic_enrich_semantic_editor" {
  project    = var.gcp_project_id
  dataset_id = google_bigquery_dataset.semantic.dataset_id
  role       = "roles/bigquery.dataEditor"
  member     = "serviceAccount:${google_service_account.semantic_enrich.email}"
}

# --- Ingest reader (future agent, pre-wired): read semantic --------
# Pre-wiring this grant now avoids a retroactive permission change
# later. sa-ingest-reader already has project-level jobUser from its
# existing wiring; no second grant needed here.

resource "google_bigquery_dataset_iam_member" "ingest_reader_semantic_viewer" {
  project    = var.gcp_project_id
  dataset_id = google_bigquery_dataset.semantic.dataset_id
  role       = "roles/bigquery.dataViewer"
  member     = "serviceAccount:${google_service_account.ingest_reader.email}"
}

# --- Human admins --------------------------------------------------
# dataOwner is broader than dataEditor — it includes the right to
# delete the dataset and grant access to others. Admins need this
# for schema migrations (e.g., a future vector-index drop/recreate).

resource "google_bigquery_dataset_iam_member" "admin_semantic" {
  for_each = var.admin_users

  project    = var.gcp_project_id
  dataset_id = google_bigquery_dataset.semantic.dataset_id
  role       = "roles/bigquery.dataOwner"
  member     = "user:${each.value}"
}
