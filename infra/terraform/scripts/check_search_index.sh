#!/usr/bin/env bash
# Spec: docs/product-specs/milestone-2/3.1-bq-warehouse-infrastructure.md §9.2
#
# Post-apply smoke check. The null_resource in bigquery.tf issues the
# search index DDL but does not track real index state in Terraform —
# a manual `DROP SEARCH INDEX rows_search` would not be detected by
# `terraform apply`. Run this after every apply and from CI on a
# schedule. Recovery is one bq query with the DDL from 3.1 §5.
#
# Usage: check_search_index.sh <project_id> [region]

set -euo pipefail

PROJECT="${1:?project id required}"
REGION="${2:-northamerica-northeast1}"

RESULT=$(bq --project_id="$PROJECT" query --use_legacy_sql=false --format=csv --quiet \
  "SELECT index_name FROM \`${PROJECT}.region-${REGION}.INFORMATION_SCHEMA.SEARCH_INDEXES\` WHERE table_schema = 'raw' AND table_name = 'rows' AND index_name = 'rows_search';")

if ! grep -q "^rows_search$" <<<"$RESULT"; then
  echo "FATAL: rows_search index missing on raw.rows. Re-run terraform apply or issue the DDL manually." >&2
  exit 1
fi

echo "rows_search OK"
