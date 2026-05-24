# Spec: docs/product-specs/milestone-1/2.1-gcs-storage-layer.md
#
# Apply:
#   terraform init
#   terraform apply -var "gcp_project_id=<project>" \
#                   -var "admins_group=<group>" \
#                   -var "readers_group=<group>"
#
# Tests (plan-mode, no GCP calls; dummy token works offline):
#   GOOGLE_OAUTH_ACCESS_TOKEN=fake terraform test
#
# State: local backend for bootstrap only. Swap to a GCS backend pointed
# at a hand-provisioned state bucket before a second human touches this.

provider "google" {
  project = var.gcp_project_id
  region  = var.gcp_region
}
