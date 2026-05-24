resource "google_storage_bucket" "raw" {
  name                        = var.bucket_name
  project                     = var.gcp_project_id
  location                    = var.gcp_region
  storage_class               = "STANDARD"
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  force_destroy               = false

  versioning {
    enabled = false
  }

  # Soft-delete disabled: dedup is checksum-based, tombstones add cost
  # without value. See 2.1 §2.1.
  soft_delete_policy {
    retention_duration_seconds = 0
  }

  # Rule 1: raw/ STANDARD -> NEARLINE @ 90d
  lifecycle_rule {
    condition {
      age                   = 90
      matches_prefix        = ["raw/"]
      matches_storage_class = ["STANDARD"]
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }

  # Rule 2: raw/ NEARLINE -> COLDLINE @ 365d
  lifecycle_rule {
    condition {
      age                   = 365
      matches_prefix        = ["raw/"]
      matches_storage_class = ["NEARLINE"]
    }
    action {
      type          = "SetStorageClass"
      storage_class = "COLDLINE"
    }
  }

  # Rule 3: quarantine/ delete @ 30d
  lifecycle_rule {
    condition {
      age            = 30
      matches_prefix = ["quarantine/"]
    }
    action {
      type = "Delete"
    }
  }

  # Rule 4: sandbox/ delete @ 7d
  lifecycle_rule {
    condition {
      age            = 7
      matches_prefix = ["sandbox/"]
    }
    action {
      type = "Delete"
    }
  }

  # Rule 5: belt-and-braces. Catches noncurrent versions if versioning
  # ever gets flipped on by accident.
  lifecycle_rule {
    condition {
      days_since_noncurrent_time = 1
    }
    action {
      type = "Delete"
    }
  }

  labels = {
    env        = "prod"
    component  = "ingest"
    managed_by = "terraform"
    data_class = "public"
  }
}
