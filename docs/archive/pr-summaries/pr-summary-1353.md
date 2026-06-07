## Summary

Adds a `SECURITY.md` at the repository root to close the supply-chain
*readiness* (SCR-RUNBOOK) gap flagged in issue #1353. The repo's supply-chain
machinery was already strong (`cargo audit` + `rustsec/audit-check` +
`dependency-review-action` on every PR, Renovate's 24h quarantine with a
security fast-lane, and `bump-deps.sh` mirroring the same window) — the only
missing piece was the human-facing runbook tying it together and naming a
disclosure contact.

`SECURITY.md` provides the two missing elements:

1. **Disclosure contact** — `security@stsoftware.com.au` plus GitHub private
   vulnerability reporting, giving researchers a non-public channel.
2. **Emergency dependency-bump runbook** — points at the existing fast-lane:
   Renovate raises security PRs immediately via `vulnerabilityAlerts`
   (`minimumReleaseAge: "0"`, no quarantine wait), and a manual emergency bump
   can run `VIBE_BUMP_QUARANTINE_HOURS=0 ./bump-deps.sh`. Notes that the
   standing verification gate (`cargo audit` / `cargo deny check` from
   `deny.toml`) still applies.

This is a documentation-only addition; no source behaviour changes.

Closes #1353.

### Dependency bump

`quality.sh` auto-upgraded the dev-dependency `serial_test` 3.4 → 3.5 during
the quality run (the repo's sanctioned bump-on-build behaviour). The change
passed the `cargo deny check` audit gate and lands in this PR per the
bump-in-same-PR policy (Issue #1613).

## Evidence

Backend/docs change — no web interface to screenshot. Verified via the new
artefact tests and the full quality gate.

- `cargo test --test issue_1353_security_md` — 4 passed.
- `./quality.sh` — all checks passed (build, clippy, type check, full test
  suite, `cargo deny check`, doc build, release build).
- `markdownlint-cli2 SECURITY.md` — 0 errors.

```mermaid
flowchart TD
    R[Vulnerability discovered] --> C{Have a contact?}
    C -->|SECURITY.md| E[security@stsoftware.com.au / private report]
    E --> F{Emergency bump?}
    F -->|Renovate fast-lane| G[vulnerabilityAlerts: no quarantine wait]
    F -->|Manual| H[VIBE_BUMP_QUARANTINE_HOURS=0 ./bump-deps.sh]
    G --> V[Verify: cargo audit + cargo deny check]
    H --> V
    V --> M[Merge fix]
```

## Test Plan

Added `tests/issue_1353_security_md.rs` (asserts on the real committed
artefact, not source patterns):

- `security_md_exists_at_repo_root` — `SECURITY.md` exists and is non-empty.
- `security_md_names_disclosure_contact` — names `security@stsoftware.com.au`.
- `security_md_documents_emergency_bump_procedure` — references
  `vulnerabilityAlerts`, `VIBE_BUMP_QUARANTINE_HOURS=0`, and `bump-deps.sh`.
- `security_md_names_verification_gate` — names `cargo audit` and
  `cargo deny check`.
