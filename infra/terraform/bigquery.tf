# Spec: docs/product-specs/milestone-2/3.1-bq-warehouse-infrastructure.md
#
# Datasets, tables, and the search index for the M2 warehouse load.
# IAM lives in bigquery_iam.tf and warehouse_load_iam.tf.

resource "google_bigquery_dataset" "raw" {
  project                     = var.gcp_project_id
  dataset_id                  = "raw"
  location                    = var.gcp_region
  description                 = "Raw warehouse layer: per-file catalog (documents), per-row JSON body (rows), and the materialised column index. See docs/product-specs/milestone-2-warehouse-load.md §4."
  default_table_expiration_ms = null
  delete_contents_on_destroy  = false

  labels = {
    env        = "prod"
    component  = "warehouse"
    layer      = "raw"
    managed_by = "terraform"
  }
}

resource "google_bigquery_dataset" "curated" {
  project                    = var.gcp_project_id
  dataset_id                 = "curated"
  location                   = var.gcp_region
  description                = "Curated warehouse layer: typed per-series promotions (master M2 §4.3). Empty in milestone 2; promotions land as separate config-only PRDs."
  delete_contents_on_destroy = false

  labels = {
    env        = "prod"
    component  = "warehouse"
    layer      = "curated"
    managed_by = "terraform"
  }
}

resource "google_bigquery_table" "documents" {
  project             = var.gcp_project_id
  dataset_id          = google_bigquery_dataset.raw.dataset_id
  table_id            = "documents"
  description         = "Per-file catalog. One row per ingested resource (from JSONL runlog). See docs/product-specs/milestone-2/3.2-documents-loader.md."
  deletion_protection = true

  clustering = ["country_code", "source_code", "organization_code"]

  schema = file("${path.module}/schemas/raw_documents.json")
}

resource "google_bigquery_table" "rows" {
  project             = var.gcp_project_id
  dataset_id          = google_bigquery_dataset.raw.dataset_id
  table_id            = "rows"
  description         = "Per-CSV-row JSON store. One row per body row across all loaded CSVs. See docs/product-specs/milestone-2/3.3-rows-loader.md."
  deletion_protection = true

  clustering = ["document_id"]

  schema = file("${path.module}/schemas/raw_rows.json")
}

resource "google_bigquery_table" "rows_staging" {
  project             = var.gcp_project_id
  dataset_id          = google_bigquery_dataset.raw.dataset_id
  table_id            = "rows_staging"
  description         = "Staging table for raw.rows MERGE. Truncated and rewritten per load batch by 3.3. Not user-queryable."
  deletion_protection = false

  clustering = ["document_id"]

  # Same schema file as raw.rows so staging cannot drift from target.
  schema = file("${path.module}/schemas/raw_rows.json")
}

resource "google_bigquery_table" "column_index" {
  project             = var.gcp_project_id
  dataset_id          = google_bigquery_dataset.raw.dataset_id
  table_id            = "column_index"
  description         = "Materialised (col_name, file_count, document_ids) index over raw.rows JSON keys. Master M2 §4.5. Refreshed by 3.3 after each rows load."
  deletion_protection = true

  schema = file("${path.module}/schemas/raw_column_index.json")
}

# Search index on raw.rows.row. Master M2 §4.5 / §6 Q1 — canonical
# brute-search path, not optional. Provider has no native resource as
# of google ~> 6.0, so we issue the DDL via bq CLI and track state with
# triggers + replace_triggered_by. See 3.1 §5.
resource "null_resource" "rows_search_index" {
  triggers = {
    table_id    = google_bigquery_table.rows.id
    ddl_version = "v1" # bump to force re-issue if DDL changes
  }

  # Without this, dropping and recreating raw.rows would lose the index
  # (BQ drops search indexes implicitly on table replace) but the
  # trigger values would be unchanged, so the DDL would never re-run.
  lifecycle {
    replace_triggered_by = [google_bigquery_table.rows]
  }

  provisioner "local-exec" {
    # Pass the DDL as a single-quoted bash argument so bash performs
    # no expansion (Terraform interpolates ${var...} before bash sees
    # the string, and the backticks around the table identifier must
    # reach `bq` literally). SQL string literals use double quotes —
    # BQ accepts both, and double quotes survive inside bash single
    # quotes without escaping gymnastics.
    command = <<-EOT
      bq --project_id=${var.gcp_project_id} query \
        --use_legacy_sql=false \
        --location=${var.gcp_region} \
        'CREATE SEARCH INDEX IF NOT EXISTS rows_search ON `${var.gcp_project_id}.${google_bigquery_dataset.raw.dataset_id}.${google_bigquery_table.rows.table_id}`(row) OPTIONS(analyzer = "LOG_ANALYZER");'
    EOT
  }

  depends_on = [google_bigquery_table.rows]
}
