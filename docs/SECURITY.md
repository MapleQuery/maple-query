Last updated: 2026-05-21

# Security bars

## Source surface

- All in-scope source APIs are **public and read-only**. We never
  authenticate to a source. If a future source requires auth, it
  needs explicit design review before adoption.
- We never write back to a source. The pipeline is one-way.

## Service identity

- Each service runs as its own identity with the minimum permissions
  it needs.
- Broad project-level roles (`roles/editor`, `roles/owner`) are
  forbidden.
- No long-lived service-account keys. Workload identity only.

## Secrets

- No inline secrets. Anything sensitive is read at runtime from a
  secret store.
- Secrets are never logged. Structured-logging setup must strip values
  for known sensitive field names.

## Storage

- Buckets are non-public. No `allUsers` bindings, no public objects.
- Per-object ACLs are disabled in favour of bucket-level access.
- IAM bindings are checked in (infrastructure-as-code); ad-hoc console
  grants are forbidden.

## Dependencies

- New outbound dependencies are reviewed and noted in the PR
  description.
- Prefer well-known, well-maintained libraries over new unknowns.
- Direct dependencies are pinned; the resolved lockfile is committed.
