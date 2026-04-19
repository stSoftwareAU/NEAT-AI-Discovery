#!/usr/bin/env bash
# Test script for Cargo Security Audit workflow completeness (Issue #1122).
#
# The VibeCoding workflow sync expects `.github/workflows/security.yml` to
# reference all three canonical cargo-audit patterns so the audit is fully
# covered:
#   1. `cargo audit`          — invocation of the audit tool
#   2. `cargo-audit`          — crate name (install or reference)
#   3. `rustsec/audit-check`  — official GitHub Action from RustSec
#
# These tests read the real workflow file and assert each pattern is present.

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

assert_pattern_present \
  "rustsec/audit-check action reference present" \
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
