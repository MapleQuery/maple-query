# Pins the bucket configuration in 2.1 §2.1. Drift fails CI.

variables {
  gcp_project_id = "maplequery-test"
  admin_users    = ["test@example.com"]
}

run "bucket_identity_locked" {
  command = plan

  assert {
    condition     = google_storage_bucket.raw.name == "maplequery-raw"
    error_message = "Bucket name must be maplequery-raw (2.1 §2.1)."
  }

  assert {
    condition     = lower(google_storage_bucket.raw.location) == "northamerica-northeast1"
    error_message = "Bucket must be in northamerica-northeast1, single region (2.1 §2.1)."
  }

  assert {
    condition     = google_storage_bucket.raw.storage_class == "STANDARD"
    error_message = "Default storage class must be STANDARD (2.1 §2.1)."
  }
}

run "bucket_lockdown_flags" {
  command = plan

  assert {
    condition     = google_storage_bucket.raw.uniform_bucket_level_access == true
    error_message = "UBLA must be on (2.1 §2.1 / §5.3)."
  }

  assert {
    condition     = google_storage_bucket.raw.public_access_prevention == "enforced"
    error_message = "Public access prevention must be enforced (2.1 §2.1 / §5.3 / acceptance criterion §10.4)."
  }

  assert {
    condition     = google_storage_bucket.raw.versioning[0].enabled == false
    error_message = "Versioning must be off (2.1 §2.1)."
  }

  assert {
    condition     = google_storage_bucket.raw.force_destroy == false
    error_message = "force_destroy must be false — bucket is production state."
  }

  assert {
    condition     = google_storage_bucket.raw.soft_delete_policy[0].retention_duration_seconds == 0
    error_message = "Soft-delete must be disabled (2.1 §2.1)."
  }
}

run "labels_present" {
  command = plan

  assert {
    condition     = google_storage_bucket.raw.labels["env"] == "prod"
    error_message = "Label env=prod missing (2.1 §2.2)."
  }

  assert {
    condition     = google_storage_bucket.raw.labels["component"] == "ingest"
    error_message = "Label component=ingest missing (2.1 §2.2)."
  }

  assert {
    condition     = google_storage_bucket.raw.labels["managed_by"] == "terraform"
    error_message = "Label managed_by=terraform missing (2.1 §2.2)."
  }

  assert {
    condition     = google_storage_bucket.raw.labels["data_class"] == "public"
    error_message = "Label data_class=public missing (2.1 §2.2)."
  }
}
