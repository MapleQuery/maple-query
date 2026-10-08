# Pins the curate (M3) tables and identity. Same dummy-token pattern:
#   GOOGLE_OAUTH_ACCESS_TOKEN=fake terraform test

variables {
  gcp_project_id = "maplequery-test"
  admin_users    = ["alice@example.com"]
}

run "curated_tables_live_in_curated" {
  command = plan

  assert {
    condition = alltrue([
      for t in [
        google_bigquery_table.curated_people,
        google_bigquery_table.curated_person_names,
        google_bigquery_table.curated_person_terms,
        google_bigquery_table.curated_person_contributions,
        google_bigquery_table.curated_person_expenses,
      ] : t.dataset_id == "curated"
    ])
    error_message = "Every curate-owned table lives in the curated dataset."
  }

  assert {
    condition     = google_bigquery_table.curated_people.deletion_protection == true
    error_message = "curated.people must be deletion-protected."
  }

  assert {
    condition     = google_bigquery_table.curated_person_contributions.clustering[0] == "person_id"
    error_message = "person_contributions clusters on person_id first (the agent reads by person)."
  }
}

run "curate_identity_scoped" {
  command = plan

  assert {
    condition     = google_service_account.curate.account_id == "sa-curate"
    error_message = "curate runs as sa-curate."
  }

  assert {
    condition     = google_bigquery_dataset_iam_member.curate_curated_editor.dataset_id == "curated"
    error_message = "sa-curate edits the curated dataset only."
  }

  assert {
    condition     = google_bigquery_dataset_iam_member.agent_service_curated_viewer.role == "roles/bigquery.dataViewer"
    error_message = "The agent reads curated.* and never writes it."
  }
}
