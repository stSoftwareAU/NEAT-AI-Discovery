#!/usr/bin/env bash
# Test script for Cargo Security Audit workflow completeness (Issue #1122).
#
# `.github/workflows/security.yml` must run a RustSec advisory audit of the
# repository's Cargo.lock on every pull request. The canonical single audit
# path is the version-pinned explicit install (CVSS 4.0 support, Issue #1223):
#   1. `cargo audit`  — invocation of the audit tool
#   2. `cargo-audit`  — crate name (install or reference)
#
# Issue #1657: the `rustsec/audit-check` GitHub Action was a *second* RustSec
# audit over the same lockfile in the same job — duplicate work. It has been
# removed, so this test no longer requires it and asserts it is absent to
# prevent the duplicate from being reintroduced.
#
# These tests read the real workflow file and assert each pattern's presence
# or absence.

set -euo pipefail

WORKFLOW_FILE=".github/workflows/security.yml"
PASS=0
FAIL=0

assert_pattern_present() {
  local description="$1"
  local pattern="$2"
  local file="$3"

  if grep -qE "$pattern" "$file"; then
    PASS=$((PASS + 1))
  else
    echo "FAIL: $description — expected pattern '$pattern' in $file"
    FAIL=$((FAIL + 1))
  fi
}

assert_pattern_absent() {
  local description="$1"
  local pattern="$2"
  local file="$3"

  if grep -qE "$pattern" "$file"; then
    echo "FAIL: $description — unexpected pattern '$pattern' in $file"
    FAIL=$((FAIL + 1))
  else
    PASS=$((PASS + 1))
  fi
}

# --- Test: workflow file exists ---
if [ ! -f "$WORKFLOW_FILE" ]; then
  echo "FAIL: $WORKFLOW_FILE not found"
  exit 1
fi

# --- Test: all three cargo audit patterns are present ---
assert_pattern_present \
  "cargo audit invocation present" \
  "cargo[[:space:]]+audit" \
  "$WORKFLOW_FILE"

assert_pattern_present \
  "cargo-audit crate reference present" \
  "cargo-audit" \
  "$WORKFLOW_FILE"

# Issue #1657: the duplicate rustsec/audit-check action must NOT be present —
# it audited the same Cargo.lock a second time in the same job.
assert_pattern_absent \
  "rustsec/audit-check duplicate removed" \
  "rustsec/audit-check" \
  "$WORKFLOW_FILE"

# --- Summary ---
echo ""
echo "Results: $PASS passed, $FAIL failed"

if [ "$FAIL" -gt 0 ]; then
  echo "❌ Security workflow validation failed"
  exit 1
fi

echo "✅ All security workflow validation tests passed"
