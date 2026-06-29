# Semantic warehouse layer: per-package summaries and per-column
# descriptions with inline embedding vectors. Consumed by the agent
# retrieval path; populated by the semantic enrichment pipeline
# (text generation + embedding service).
#
# IAM lives in semantic_iam.tf and semantic_enrich_iam.tf.

resource "google_bigquery_dataset" "semantic" {
  project                     = var.gcp_project_id
  dataset_id                  = "semantic"
  location                    = var.gcp_region
  description                 = "Semantic warehouse layer: per-package summaries and per-column descriptions with inline embedding vectors."
  default_table_expiration_ms = null
  delete_contents_on_destroy  = false

  labels = {
    env        = "prod"
    component  = "warehouse"
    layer      = "semantic"
    managed_by = "terraform"
  }
}

resource "google_bigquery_table" "semantic_datasets" {
  project             = var.gcp_project_id
  dataset_id          = google_bigquery_dataset.semantic.dataset_id
  table_id            = "datasets"
  description         = "Per-CKAN-package semantic summary. One row per package_id."
  deletion_protection = true

  clustering = ["package_id"]

  schema = file("${path.module}/schemas/semantic_datasets.json")
}

resource "google_bigquery_table" "semantic_columns" {
  project             = var.gcp_project_id
  dataset_id          = google_bigquery_dataset.semantic.dataset_id
  table_id            = "columns"
  description         = "Per-(package, column) semantic description. One row per distinct column name within a package."
  deletion_protection = true

  clustering = ["package_id"]

  schema = file("${path.module}/schemas/semantic_columns.json")
}
