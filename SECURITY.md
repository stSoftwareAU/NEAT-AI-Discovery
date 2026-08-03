# Security Policy

## Reporting a vulnerability

If you discover a security vulnerability in this repository, please report it
**privately** so we can address it before public disclosure. Do **not** open a
public GitHub issue for a suspected vulnerability.

- **Email:** [security@stsoftware.com.au](mailto:security@stsoftware.com.au)
- **GitHub:** alternatively, use
  [private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
  via the repository's **Security** tab.

Please include enough detail to reproduce the issue (affected version or
commit, steps, and impact). We aim to acknowledge reports promptly and will
keep you informed as we work through triage and remediation.

## Supported versions

This crate is released as a rolling line — security fixes land on the default
branch and are picked up by the next patch release. Always run the latest
published version.

## Supply-chain machinery

Routine supply-chain defence runs automatically; you do not need to invoke it
to report a vulnerability. It is summarised here so responders know what is
already in place:

- **Per-PR audit** — `cargo audit` and
  `actions/dependency-review-action` run on every pull request via
  `.github/workflows/security.yml` (invoked from `.github/workflows/ci.yml`).
- **Quarantine window** — Renovate holds external crates.io and GitHub Actions
  updates for 24h after publish (`renovate.json`), and `bump-deps.sh` mirrors
  the same window locally via `VIBE_BUMP_QUARANTINE_HOURS` (default `24`). The
  local gate covers the resolved `Cargo.lock` and every dependency table of
  every tracked manifest — `[build-dependencies]`, `[target.<spec>.*]` and
  `fuzz/Cargo.toml` included (Issue #1908).
- **Internal deps** — first-party `stSoftwareAU/*` releases bypass the
  quarantine window.
- **Expiring suppressions** — `deny.toml` sets
  `[advisories] unused-ignored-advisory = "deny"`, so an ignore that no longer
  matches any crate in the graph fails `cargo deny check` (Issue #1917). A
  suppression may not outlive the dependency it was written for: left in place
  it would silently re-suppress the advisory if that crate ever returned. For
  the same reason the `dependency-review-action` step carries no `allow-ghsas`
  list — add one only alongside a matching, live `deny.toml` ignore.

## Emergency dependency-bump runbook

When an actively-exploited CVE requires an immediate dependency bump, use the
existing fast-lane rather than waiting out the 24h quarantine window:

1. **Renovate security fast-lane (preferred).** Renovate raises security
   update PRs immediately — `vulnerabilityAlerts` in `renovate.json` sets
   `minimumReleaseAge: "0"`, so security advisories bypass the 24h quarantine.
   Review and merge the PR once CI is green.
2. **Manual emergency bump.** If you must bump locally without waiting, run:

   ```bash
   VIBE_BUMP_QUARANTINE_HOURS=0 ./bump-deps.sh
   ```

   Setting `VIBE_BUMP_QUARANTINE_HOURS=0` disables the quarantine wait for this
   run. The audit gate (`cargo deny check`) still runs and will reject a tree
   that introduces a new advisory.
3. **Verify before merging.** Whichever path you take, the standing
   verification remains:

   ```bash
   cargo audit
   cargo deny check
   ```

   `cargo deny check` uses the policy in `deny.toml`. Both must pass before the
   bump is merged.

This runbook documents the human-facing procedure only; the configuration it
references (`renovate.json`, `bump-deps.sh`, `deny.toml`,
`.github/workflows/security.yml`) is the source of truth.
