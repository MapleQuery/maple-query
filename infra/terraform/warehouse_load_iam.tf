# Spec: docs/product-specs/milestone-2/3.1-bq-warehouse-infrastructure.md §6.1 - §6.2
#
# sa-warehouse-load identity and its GCS-side grant. Pairs with
# bigquery_iam.tf which holds the BQ-side grants.

resource "google_service_account" "warehouse_load" {
  project      = var.gcp_project_id
  account_id   = "sa-warehouse-load"
  display_name = "MapleQuery warehouse loader"
  description  = "Identity for the documents loader (3.2) and rows loader (3.3). Reads runlogs + CSV bytes from GCS; writes to BQ raw dataset."
}

# Read access to gs://maplequery-raw/raw/ only. The condition keeps
# the SA out of quarantine/ and sandbox/ (matches 2.1 §5.1 scoping).
resource "google_storage_bucket_iam_member" "warehouse_load_raw" {
  bucket = google_storage_bucket.raw.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.warehouse_load.email}"

  condition {
    title       = "raw prefix only"
    description = "Scoped per 2.1 §5.1 — read access does not extend to quarantine/ or sandbox/."
    expression  = <<-EOT
      resource.name.startsWith("projects/_/buckets/${google_storage_bucket.raw.name}/objects/raw/")
    EOT
  }
}
