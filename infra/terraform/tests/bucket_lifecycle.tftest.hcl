# Pins the 5 lifecycle rules in 2.1 §4.1. Drift fails CI.
# Acceptance criterion §10.2.
#
# Note: in the google provider, lifecycle_rule.action and
# lifecycle_rule.condition are TypeSet with MaxItems=1, so we use
# one() rather than [0] to extract the single element.

variables {
  gcp_project_id = "maplequery-test"
  admin_users    = ["test@example.com"]
}

run "exactly_five_rules" {
  command = plan

  assert {
    condition     = length(google_storage_bucket.raw.lifecycle_rule) == 5
    error_message = "Expected exactly 5 lifecycle rules per 2.1 §4.1 — adding or removing one is a contract change, update the PRD first."
  }
}

run "rule_1_raw_to_nearline_90d" {
  command = plan

  assert {
    condition = length([
      for r in google_storage_bucket.raw.lifecycle_rule : r
      if try(one(r.action).type, "") == "SetStorageClass"
      && try(one(r.action).storage_class, "") == "NEARLINE"
      && try(one(r.condition).age, 0) == 90
      && contains(try(one(r.condition).matches_prefix, []), "raw/")
      && contains(try(one(r.condition).matches_storage_class, []), "STANDARD")
    ]) == 1
    error_message = "Rule 1 missing or drifted: raw/ STANDARD -> NEARLINE @ 90d (2.1 §4.1)."
  }
}

run "rule_2_raw_to_coldline_365d" {
  command = plan

  assert {
    condition = length([
      for r in google_storage_bucket.raw.lifecycle_rule : r
      if try(one(r.action).type, "") == "SetStorageClass"
      && try(one(r.action).storage_class, "") == "COLDLINE"
      && try(one(r.condition).age, 0) == 365
      && contains(try(one(r.condition).matches_prefix, []), "raw/")
      && contains(try(one(r.condition).matches_storage_class, []), "NEARLINE")
    ]) == 1
    error_message = "Rule 2 missing or drifted: raw/ NEARLINE -> COLDLINE @ 365d (2.1 §4.1)."
  }
}

run "rule_3_quarantine_delete_30d" {
  command = plan

  assert {
    condition = length([
      for r in google_storage_bucket.raw.lifecycle_rule : r
      if try(one(r.action).type, "") == "Delete"
      && try(one(r.condition).age, 0) == 30
      && contains(try(one(r.condition).matches_prefix, []), "quarantine/")
    ]) == 1
    error_message = "Rule 3 missing or drifted: quarantine/ delete @ 30d (2.1 §4.1)."
  }
}

run "rule_4_sandbox_delete_7d" {
  command = plan

  assert {
    condition = length([
      for r in google_storage_bucket.raw.lifecycle_rule : r
      if try(one(r.action).type, "") == "Delete"
      && try(one(r.condition).age, 0) == 7
      && contains(try(one(r.condition).matches_prefix, []), "sandbox/")
    ]) == 1
    error_message = "Rule 4 missing or drifted: sandbox/ delete @ 7d (2.1 §4.1)."
  }
}

run "rule_5_noncurrent_belt_and_braces" {
  command = plan

  assert {
    condition = length([
      for r in google_storage_bucket.raw.lifecycle_rule : r
      if try(one(r.action).type, "") == "Delete"
      && try(one(r.condition).days_since_noncurrent_time, 0) == 1
    ]) == 1
    error_message = "Rule 5 missing or drifted: delete noncurrent versions @ 1d (2.1 §4.1)."
  }
}
