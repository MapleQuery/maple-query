resource "google_service_account" "ingest_job" {
  project      = var.gcp_project_id
  account_id   = "sa-ingest-job"
  display_name = "MapleQuery ingest job"
  description  = "Cloud Run ingest job identity. Consumed by 2.2."
}

resource "google_service_account" "ingest_reader" {
  project      = var.gcp_project_id
  account_id   = "sa-ingest-reader"
  display_name = "MapleQuery ingest reader (M2 extract)"
  description  = "Read-only consumer of raw/. Reserved for M2; created now to avoid retroactive grant."
}

# Ingest SA: objectAdmin scoped to raw/ + quarantine/ only (2.1 §5.1).
# objectAdmin is the least-permissive role that allows create + delete
# + overwrite, which the dedup ladder needs for failed partial uploads.
resource "google_storage_bucket_iam_member" "ingest_raw_quarantine" {
  bucket = google_storage_bucket.raw.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.ingest_job.email}"

  condition {
    title       = "raw and quarantine prefixes only"
    description = "Scoped per 2.1 §5.1"
    expression  = <<-EOT
      resource.name.startsWith("projects/_/buckets/${google_storage_bucket.raw.name}/objects/raw/")
      || resource.name.startsWith("projects/_/buckets/${google_storage_bucket.raw.name}/objects/quarantine/")
    EOT
  }
}

# Reader SA: objectViewer scoped to raw/ only (2.1 §5.1).
resource "google_storage_bucket_iam_member" "reader_raw" {
  bucket = google_storage_bucket.raw.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.ingest_reader.email}"

  condition {
    title       = "raw prefix only"
    description = "Scoped per 2.1 §5.1"
    expression  = <<-EOT
      resource.name.startsWith("projects/_/buckets/${google_storage_bucket.raw.name}/objects/raw/")
    EOT
  }
}

# Human admins as direct user bindings. Small-team deviation from
# 2.1 §5.2; see variable docs and PRD note.
resource "google_storage_bucket_iam_member" "admins" {
  for_each = var.admin_users

  bucket = google_storage_bucket.raw.name
  role   = "roles/storage.admin"
  member = "user:${each.value}"
}
