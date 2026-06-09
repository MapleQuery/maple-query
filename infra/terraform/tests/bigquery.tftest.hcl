# Pins the BQ warehouse infra in 3.1. Drift fails CI.
# Mirrors the dummy-token pattern from 2.1's tests:
#   GOOGLE_OAUTH_ACCESS_TOKEN=fake terraform test

variables {
  gcp_project_id = "maplequery-test"
  admin_users    = ["alice@example.com", "bob@example.com"]
}

run "datasets_created" {
  command = plan

  assert {
    condition     = google_bigquery_dataset.raw.dataset_id == "raw"
    error_message = "raw dataset must be named 'raw' (3.1 §3.1)."
  }

  assert {
    condition     = lower(google_bigquery_dataset.raw.location) == "northamerica-northeast1"
    error_message = "raw dataset must be in northamerica-northeast1 (3.1 §3.1; co-located with bucket per 2.1 §2.1)."
  }

  assert {
    condition     = google_bigquery_dataset.curated.dataset_id == "curated"
    error_message = "curated dataset must be named 'curated' (3.1 §3.2)."
  }

  assert {
    condition     = lower(google_bigquery_dataset.curated.location) == "northamerica-northeast1"
    error_message = "curated dataset must be in northamerica-northeast1 (3.1 §3.2)."
  }

  assert {
    condition     = google_bigquery_dataset.raw.delete_contents_on_destroy == false
    error_message = "raw dataset must not delete contents on destroy (3.1 §3.1)."
  }
}

run "table_clustering_pinned" {
  command = plan

  assert {
    condition     = google_bigquery_table.rows.clustering[0] == "document_id"
    error_message = "raw.rows must cluster on document_id (3.1 §4.2 / master M2 §6 Q12)."
  }

  assert {
    condition     = length(google_bigquery_table.rows.clustering) == 1
    error_message = "raw.rows must have exactly one clustering column — document_id (master M2 §6 Q12)."
  }

  assert {
    condition     = google_bigquery_table.documents.clustering[0] == "country_code"
    error_message = "raw.documents must cluster on (country_code, source_code, organization_code); country_code first (3.1 §4.1)."
  }

  assert {
    condition     = length(google_bigquery_table.documents.clustering) == 3
    error_message = "raw.documents must have three clustering columns (3.1 §4.1)."
  }

  assert {
    condition     = google_bigquery_table.rows_staging.clustering[0] == "document_id"
    error_message = "raw.rows_staging must cluster on document_id to match raw.rows (3.1 §4.3)."
  }
}

run "no_partitioning" {
  command = plan

  # Master M2 §6 Q12: cluster-only, no time-partitioning. The
  # google_bigquery_table resource exposes time_partitioning as a list
  # block; length 0 means unset.
  assert {
    condition     = length(google_bigquery_table.documents.time_partitioning) == 0
    error_message = "raw.documents must not be time-partitioned (master M2 §6 Q12)."
  }

  assert {
    condition     = length(google_bigquery_table.rows.time_partitioning) == 0
    error_message = "raw.rows must not be time-partitioned (master M2 §6 Q12)."
  }

  assert {
    condition     = length(google_bigquery_table.rows_staging.time_partitioning) == 0
    error_message = "raw.rows_staging must not be time-partitioned."
  }

  assert {
    condition     = length(google_bigquery_table.column_index.time_partitioning) == 0
    error_message = "raw.column_index must not be time-partitioned."
  }
}

run "deletion_protection" {
  command = plan

  assert {
    condition     = google_bigquery_table.documents.deletion_protection == true
    error_message = "raw.documents must have deletion_protection enabled (3.1 §4)."
  }

  assert {
    condition     = google_bigquery_table.rows.deletion_protection == true
    error_message = "raw.rows must have deletion_protection enabled (3.1 §4)."
  }

  assert {
    condition     = google_bigquery_table.column_index.deletion_protection == true
    error_message = "raw.column_index must have deletion_protection enabled (3.1 §4)."
  }

  # Staging is the one exception — the loader truncates and rewrites
  # it every batch (3.1 §4.3).
  assert {
    condition     = google_bigquery_table.rows_staging.deletion_protection == false
    error_message = "raw.rows_staging must NOT have deletion_protection — loader truncates it (3.1 §4.3)."
  }
}

run "staging_shares_rows_schema" {
  command = plan

  # Schema-file reuse is the cheap structural guarantee against staging
  # drifting from target. If someone splits these into two files we
  # want CI to scream.
  assert {
    condition     = google_bigquery_table.rows.schema == google_bigquery_table.rows_staging.schema
    error_message = "raw.rows_staging must share raw.rows's schema file (3.1 §4.3)."
  }
}

run "warehouse_loader_iam" {
  command = plan

  assert {
    condition     = google_service_account.warehouse_load.account_id == "sa-warehouse-load"
    error_message = "Warehouse loader SA must be sa-warehouse-load (3.1 §6.1)."
  }

  assert {
    condition     = google_bigquery_dataset_iam_member.warehouse_load_raw_editor.role == "roles/bigquery.dataEditor"
    error_message = "Warehouse loader must have roles/bigquery.dataEditor on raw (3.1 §6.3)."
  }

  assert {
    condition     = google_project_iam_member.warehouse_load_job_user.role == "roles/bigquery.jobUser"
    error_message = "Warehouse loader must have roles/bigquery.jobUser at project level (3.1 §6.3)."
  }

  # GCS scoping: read access to raw/ only, not quarantine/ or sandbox/.
  assert {
    condition = strcontains(
      google_storage_bucket_iam_member.warehouse_load_raw.condition[0].expression,
      "objects/raw/"
    )
    error_message = "Warehouse loader GCS condition must allow raw/ (3.1 §6.2)."
  }

  assert {
    condition = !strcontains(
      google_storage_bucket_iam_member.warehouse_load_raw.condition[0].expression,
      "objects/quarantine/"
    )
    error_message = "Warehouse loader must NOT have access to quarantine/ (3.1 §6.2)."
  }

  assert {
    condition = !strcontains(
      google_storage_bucket_iam_member.warehouse_load_raw.condition[0].expression,
      "objects/sandbox/"
    )
    error_message = "Warehouse loader must NOT have access to sandbox/ (3.1 §6.2)."
  }
}

run "ingest_reader_extended_to_bq" {
  command = plan

  assert {
    condition     = google_bigquery_dataset_iam_member.ingest_reader_raw_viewer.role == "roles/bigquery.dataViewer"
    error_message = "Ingest reader must have roles/bigquery.dataViewer on raw (3.1 §6.4)."
  }

  assert {
    condition     = google_bigquery_dataset_iam_member.ingest_reader_curated_viewer.role == "roles/bigquery.dataViewer"
    error_message = "Ingest reader must have roles/bigquery.dataViewer on curated (3.1 §6.4 — pre-wired for M3 agent)."
  }

  assert {
    condition     = google_project_iam_member.ingest_reader_job_user.role == "roles/bigquery.jobUser"
    error_message = "Ingest reader must have roles/bigquery.jobUser at project level (3.1 §6.4)."
  }
}

run "admin_bindings_match_set" {
  command = plan

  assert {
    condition     = length(google_bigquery_dataset_iam_member.admin_raw) == length(var.admin_users)
    error_message = "Admin bindings on raw must match admin_users 1:1 (3.1 §6.5)."
  }

  assert {
    condition     = length(google_bigquery_dataset_iam_member.admin_curated) == length(var.admin_users)
    error_message = "Admin bindings on curated must match admin_users 1:1 (3.1 §6.5)."
  }

  assert {
    condition = alltrue([
      for k, b in google_bigquery_dataset_iam_member.admin_raw : b.role == "roles/bigquery.dataOwner"
    ])
    error_message = "Every admin binding on raw must use roles/bigquery.dataOwner (3.1 §6.5)."
  }
}

run "search_index_wiring" {
  command = plan

  # The null_resource is the only signal Terraform has that the DDL
  # should be issued; pin its presence + trigger shape so a future
  # refactor doesn't silently drop the recreation path. Can't compare
  # table.id because it's only known post-apply; checking the keys
  # and ddl_version is enough to catch a structural break.
  assert {
    condition     = contains(keys(null_resource.rows_search_index.triggers), "table_id")
    error_message = "Search index null_resource must trigger on table_id so table replacement re-issues the DDL (3.1 §5)."
  }

  assert {
    condition     = null_resource.rows_search_index.triggers.ddl_version == "v1"
    error_message = "Search index ddl_version pinned to v1; bump intentionally when DDL changes (3.1 §5)."
  }
}
