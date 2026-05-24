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
