variable "gcp_project_id" {
  description = "GCP project that owns the maplequery-raw bucket and ingest service accounts."
  type        = string
}

variable "gcp_region" {
  description = "Region the bucket lives in. Locked to northamerica-northeast1 by 2.1 §2.1; exposed only so tests can override."
  type        = string
  default     = "northamerica-northeast1"
}

variable "bucket_name" {
  description = "Raw-data bucket name. Locked to maplequery-raw by 2.1 §2.1; exposed only so tests can override."
  type        = string
  default     = "maplequery-raw"
}

variable "admin_users" {
  description = <<-EOT
    User emails granted roles/storage.admin on the bucket.

    Deviation from 2.1 §5.2 (which requires group bindings): while the
    team is pre-Workspace and ≤ a handful of devs, individual user
    bindings are acceptable. Revisit when adopting Cloud Identity /
    Workspace or when team > 3 — switch to a group binding then.
  EOT
  type        = set(string)
}
