output "bucket_name" {
  description = "Name of the maplequery-raw bucket."
  value       = google_storage_bucket.raw.name
}

output "ingest_job_sa_email" {
  description = "Email of the ingest job service account, consumed by 2.2's Cloud Run job."
  value       = google_service_account.ingest_job.email
}

output "ingest_reader_sa_email" {
  description = "Email of the ingest reader service account, reserved for M2 extract."
  value       = google_service_account.ingest_reader.email
}

output "bq_raw_dataset" {
  description = "Fully-qualified raw dataset reference (project.dataset), consumed by services/warehouse-load."
  value       = "${var.gcp_project_id}.${google_bigquery_dataset.raw.dataset_id}"
}

output "bq_curated_dataset" {
  description = "Fully-qualified curated dataset reference (project.dataset)."
  value       = "${var.gcp_project_id}.${google_bigquery_dataset.curated.dataset_id}"
}

output "warehouse_load_sa_email" {
  description = "Email of the warehouse loader service account. Used by 3.2 and 3.3 for impersonation in CI / local runs."
  value       = google_service_account.warehouse_load.email
}

output "bq_semantic_dataset" {
  description = "Fully-qualified semantic dataset reference (project.dataset), consumed by the semantic enrichment service."
  value       = "${var.gcp_project_id}.${google_bigquery_dataset.semantic.dataset_id}"
}

output "semantic_enrich_sa_email" {
  description = "Email of the semantic enricher service account. Used by the enrichment pipeline for impersonation in CI / local runs."
  value       = google_service_account.semantic_enrich.email
}
