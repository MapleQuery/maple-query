#!/usr/bin/env bash
# Post-apply smoke check for the semantic-layer embedding columns.
#
# Terraform validates the JSON schema files at plan time, but the
# load-bearing assertion is that BigQuery actually reports
# ARRAY<FLOAT64> for the `embedding` column on both tables. A drift
# here (manual ALTER TABLE, or a future provider release changing how
# REPEATED FLOAT64 round-trips through HCL) costs hours to diagnose
# once rows have been written; catching it pre-load is cheap.
#
# Run after every terraform apply and on a CI schedule.
#
# Usage: check_semantic_embeddings.sh <project_id>

set -euo pipefail

PROJECT="${1:?project id required}"

# data_type is the SQL standard form: ARRAY<FLOAT64>. is_nullable is
# 'YES' because REPEATED fields in BQ are technically nullable (an
# empty array is the NULL representation); what we care about is the
# ARRAY<FLOAT64> shape, not nullability.
RESULT=$(bq --project_id="$PROJECT" query --use_legacy_sql=false --format=csv --quiet <<SQL
SELECT
  table_name,
  column_name,
  data_type
FROM \`${PROJECT}.semantic.INFORMATION_SCHEMA.COLUMNS\`
WHERE column_name = 'embedding'
ORDER BY table_name;
SQL
)

EXPECTED=$'table_name,column_name,data_type\ncolumns,embedding,ARRAY<FLOAT64>\ndatasets,embedding,ARRAY<FLOAT64>'

if [[ "$RESULT" != "$EXPECTED" ]]; then
  echo "FATAL: semantic.*.embedding shape mismatch." >&2
  echo "Expected:" >&2
  echo "$EXPECTED" >&2
  echo "Got:" >&2
  echo "$RESULT" >&2
  exit 1
fi

echo "semantic embedding columns OK"
