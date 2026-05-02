#!/usr/bin/env bash
# Test script for Cargo Quality workflow completeness (Issue #1180).
#
# The VibeCoding workflow sync expects `.github/workflows/cargo-quality.yml`
# to wire up the standard Rust quality gate:
#   1. `actions/checkout`        — fetch the source tree
#   2. `dtolnay/rust-toolchain`  — install the Rust toolchain
#   3. `rustfmt, clippy`         — toolchain components
#   4. `cargo fmt --check`       — formatting gate
#   5. `cargo clippy`            — lint gate with warnings denied
#   6. `cargo-llvm-cov`          — coverage instrumentation
#   7. `codecov/codecov-action`  — coverage upload
#
# These tests read the real workflow file and assert each pattern is present.

set -euo pipefail

WORKFLOW_FILE=".github/workflows/cargo-quality.yml"
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

# --- Test: triggers on pull_request ---
assert_pattern_present \
  "pull_request trigger present" \
  "pull_request:" \
  "$WORKFLOW_FILE"

# --- Test: required actions and components ---
assert_pattern_present \
  "actions/checkout reference present" \
  "actions/checkout@v[0-9]+" \
  "$WORKFLOW_FILE"

assert_pattern_present \
  "dtolnay/rust-toolchain reference present" \
  "dtolnay/rust-toolchain" \
  "$WORKFLOW_FILE"

assert_pattern_present \
  "rustfmt component present" \
  "rustfmt" \
  "$WORKFLOW_FILE"

assert_pattern_present \
  "clippy component present" \
  "clippy" \
  "$WORKFLOW_FILE"

# --- Test: cargo fmt and clippy invocations ---
assert_pattern_present \
  "cargo fmt --check invocation present" \
  "cargo[[:space:]]+fmt[[:space:]]+--check" \
  "$WORKFLOW_FILE"

assert_pattern_present \
  "cargo clippy with -D warnings present" \
  "cargo[[:space:]]+clippy.*-D[[:space:]]+warnings" \
  "$WORKFLOW_FILE"

# --- Test: coverage tooling present ---
assert_pattern_present \
  "cargo-llvm-cov install action present" \
  "cargo-llvm-cov" \
  "$WORKFLOW_FILE"

assert_pattern_present \
  "codecov/codecov-action reference present" \
  "codecov/codecov-action@v[0-9]+" \
  "$WORKFLOW_FILE"

# --- Test: contents read permission set ---
assert_pattern_present \
  "contents: read permission present" \
  "contents:[[:space:]]+read" \
  "$WORKFLOW_FILE"

# --- Summary ---
echo ""
echo "Results: $PASS passed, $FAIL failed"

if [ "$FAIL" -gt 0 ]; then
  echo "❌ Cargo Quality workflow validation failed"
  exit 1
fi

echo "✅ All Cargo Quality workflow validation tests passed"
