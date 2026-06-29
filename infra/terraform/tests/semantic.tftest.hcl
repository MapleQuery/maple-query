# Pins the BQ semantic-layer infra. Drift fails CI.
# Run offline with a dummy token:
#   GOOGLE_OAUTH_ACCESS_TOKEN=fake terraform test

variables {
  gcp_project_id = "maplequery-test"
  admin_users    = ["alice@example.com", "bob@example.com"]
}

run "semantic_dataset_created" {
  command = plan

  assert {
    condition     = google_bigquery_dataset.semantic.dataset_id == "semantic"
    error_message = "semantic dataset must be named 'semantic'."
  }

  assert {
    condition     = lower(google_bigquery_dataset.semantic.location) == "northamerica-northeast1"
    error_message = "semantic dataset must be in northamerica-northeast1 (co-located with raw to keep enrichment joins in-region)."
  }

  assert {
    condition     = google_bigquery_dataset.semantic.delete_contents_on_destroy == false
    error_message = "semantic dataset must not delete contents on destroy — enrichment is expensive to regenerate."
  }
}

run "semantic_table_clustering_pinned" {
  command = plan

  assert {
    condition     = google_bigquery_table.semantic_datasets.clustering[0] == "package_id"
    error_message = "semantic.datasets must cluster on package_id."
  }

  assert {
    condition     = length(google_bigquery_table.semantic_datasets.clustering) == 1
    error_message = "semantic.datasets must have exactly one clustering column — package_id."
  }

  assert {
    condition     = google_bigquery_table.semantic_columns.clustering[0] == "package_id"
    error_message = "semantic.columns must cluster on package_id."
  }

  assert {
    condition     = length(google_bigquery_table.semantic_columns.clustering) == 1
    error_message = "semantic.columns must have exactly one clustering column — package_id."
  }
}

run "semantic_no_partitioning" {
  command = plan

  # Cluster-only by design. Tables are small, write-once-per-enrichment,
  # and the read path does not filter by any time column. Adding a
  # generated_at partition would prune nothing and would break the
  # future re-enrichment MERGE pattern.
  assert {
    condition     = length(google_bigquery_table.semantic_datasets.time_partitioning) == 0
    error_message = "semantic.datasets must not be time-partitioned (cluster-only)."
  }

  assert {
    condition     = length(google_bigquery_table.semantic_columns.time_partitioning) == 0
    error_message = "semantic.columns must not be time-partitioned (cluster-only)."
  }
}

run "semantic_deletion_protection" {
  command = plan

  assert {
    condition     = google_bigquery_table.semantic_datasets.deletion_protection == true
    error_message = "semantic.datasets must have deletion_protection enabled."
  }

  assert {
    condition     = google_bigquery_table.semantic_columns.deletion_protection == true
    error_message = "semantic.columns must have deletion_protection enabled."
  }
}

run "semantic_embedding_shape" {
  command = plan

  # Static guarantee that the schema files contain a REPEATED FLOAT64
  # 'embedding' field on both tables. Catches accidental schema edits
  # (e.g. switching to STRING). The runtime ARRAY<FLOAT64> assertion
  # against INFORMATION_SCHEMA lives in scripts/check_semantic_embeddings.sh
  # and runs post-apply.
  assert {
    condition = anytrue([
      for f in jsondecode(file("${path.module}/schemas/semantic_datasets.json")) :
      f.name == "embedding" && f.type == "FLOAT64" && f.mode == "REPEATED"
    ])
    error_message = "semantic.datasets schema must contain REPEATED FLOAT64 'embedding'."
  }

  assert {
    condition = anytrue([
      for f in jsondecode(file("${path.module}/schemas/semantic_columns.json")) :
      f.name == "embedding" && f.type == "FLOAT64" && f.mode == "REPEATED"
    ])
    error_message = "semantic.columns schema must contain REPEATED FLOAT64 'embedding'."
  }
}

run "semantic_required_fields" {
  command = plan

  # package_id is the FK / PK column — must be REQUIRED on both tables.
  assert {
    condition = anytrue([
      for f in jsondecode(file("${path.module}/schemas/semantic_datasets.json")) :
      f.name == "package_id" && f.mode == "REQUIRED"
    ])
    error_message = "semantic.datasets.package_id must be REQUIRED."
  }

  assert {
    condition = anytrue([
      for f in jsondecode(file("${path.module}/schemas/semantic_columns.json")) :
      f.name == "package_id" && f.mode == "REQUIRED"
    ])
    error_message = "semantic.columns.package_id must be REQUIRED."
  }

  assert {
    condition = anytrue([
      for f in jsondecode(file("${path.module}/schemas/semantic_columns.json")) :
      f.name == "column_name" && f.mode == "REQUIRED"
    ])
    error_message = "semantic.columns.column_name must be REQUIRED."
  }
}

run "semantic_enrich_iam" {
  command = plan

  assert {
    condition     = google_service_account.semantic_enrich.account_id == "sa-semantic-enrich"
    error_message = "Semantic enricher SA must be sa-semantic-enrich."
  }

  assert {
    condition     = google_bigquery_dataset_iam_member.semantic_enrich_raw_viewer.role == "roles/bigquery.dataViewer"
    error_message = "Semantic enricher must have roles/bigquery.dataViewer on raw."
  }

  assert {
    condition     = google_bigquery_dataset_iam_member.semantic_enrich_semantic_editor.role == "roles/bigquery.dataEditor"
    error_message = "Semantic enricher must have roles/bigquery.dataEditor on semantic."
  }

  assert {
    condition     = google_project_iam_member.semantic_enrich_job_user.role == "roles/bigquery.jobUser"
    error_message = "Semantic enricher must have roles/bigquery.jobUser at project level."
  }
}

run "ingest_reader_extended_to_semantic" {
  command = plan

  assert {
    condition     = google_bigquery_dataset_iam_member.ingest_reader_semantic_viewer.role == "roles/bigquery.dataViewer"
    error_message = "Ingest reader must have roles/bigquery.dataViewer on semantic (pre-wired for the future agent read path)."
  }
}

run "admin_semantic_bindings" {
  command = plan

  assert {
    condition     = length(google_bigquery_dataset_iam_member.admin_semantic) == length(var.admin_users)
    error_message = "Admin bindings on semantic must match admin_users 1:1."
  }

  assert {
    condition = alltrue([
      for k, b in google_bigquery_dataset_iam_member.admin_semantic : b.role == "roles/bigquery.dataOwner"
    ])
    error_message = "Every admin binding on semantic must use roles/bigquery.dataOwner."
  }
}
