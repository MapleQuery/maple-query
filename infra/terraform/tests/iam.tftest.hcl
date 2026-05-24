# Pins the service accounts and bucket IAM bindings in 2.1 §5.
# Drift fails CI. Live integration tests (acceptance criterion §10.3:
# "ingest SA can write to raw/ but not sandbox/") run separately
# against a real project — they need bytes-on-the-wire and can't be
# expressed in `terraform test`.

variables {
  gcp_project_id = "maplequery-test"
  admin_users    = ["alice@example.com", "bob@example.com"]
}

run "service_accounts_exist" {
  command = plan

  assert {
    condition     = google_service_account.ingest_job.account_id == "sa-ingest-job"
    error_message = "Ingest job SA must be sa-ingest-job (2.1 §5.1)."
  }

  assert {
    condition     = google_service_account.ingest_reader.account_id == "sa-ingest-reader"
    error_message = "Ingest reader SA must be sa-ingest-reader (2.1 §5.1)."
  }
}

run "ingest_sa_scoped_to_raw_and_quarantine" {
  command = plan

  assert {
    condition     = google_storage_bucket_iam_member.ingest_raw_quarantine.role == "roles/storage.objectAdmin"
    error_message = "Ingest SA must have roles/storage.objectAdmin — least-permissive role with delete (2.1 §5.1)."
  }

  assert {
    condition = strcontains(
      google_storage_bucket_iam_member.ingest_raw_quarantine.condition[0].expression,
      "objects/raw/"
    )
    error_message = "Ingest SA IAM condition must allow raw/ prefix (2.1 §5.1)."
  }

  assert {
    condition = strcontains(
      google_storage_bucket_iam_member.ingest_raw_quarantine.condition[0].expression,
      "objects/quarantine/"
    )
    error_message = "Ingest SA IAM condition must allow quarantine/ prefix (2.1 §5.1)."
  }

  # Negative: condition must NOT mention sandbox/. The forbid-sandbox-writes
  # invariant in 2.1 §3.3 lives at the IAM layer too, not just lint.
  assert {
    condition = !strcontains(
      google_storage_bucket_iam_member.ingest_raw_quarantine.condition[0].expression,
      "objects/sandbox/"
    )
    error_message = "Ingest SA must NOT be able to write to sandbox/ (2.1 §3.3 / §5.1)."
  }
}

run "reader_sa_scoped_to_raw_only" {
  command = plan

  assert {
    condition     = google_storage_bucket_iam_member.reader_raw.role == "roles/storage.objectViewer"
    error_message = "Reader SA must have roles/storage.objectViewer (2.1 §5.1)."
  }

  assert {
    condition = strcontains(
      google_storage_bucket_iam_member.reader_raw.condition[0].expression,
      "objects/raw/"
    )
    error_message = "Reader SA condition must allow raw/ (2.1 §5.1)."
  }

  assert {
    condition = !strcontains(
      google_storage_bucket_iam_member.reader_raw.condition[0].expression,
      "objects/quarantine/"
    )
    error_message = "Reader SA must NOT see quarantine/ — quarantine triage belongs to humans + ingest SA (2.1 §5.1)."
  }
}

run "admin_user_bindings" {
  command = plan

  # One binding per admin in admin_users (small-team deviation from
  # 2.1 §5.2 groups-only — see variable docs).
  assert {
    condition     = length(google_storage_bucket_iam_member.admins) == length(var.admin_users)
    error_message = "Admin bindings must match the admin_users set 1:1."
  }

  assert {
    condition = alltrue([
      for k, b in google_storage_bucket_iam_member.admins :
      b.role == "roles/storage.admin"
    ])
    error_message = "Every admin binding must use roles/storage.admin."
  }

  assert {
    condition = alltrue([
      for k, b in google_storage_bucket_iam_member.admins :
      startswith(b.member, "user:")
    ])
    error_message = "Admin bindings must be user: members (small-team deviation; switch to group: when adopting Workspace)."
  }
}
