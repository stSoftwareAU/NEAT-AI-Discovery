#!/usr/bin/env bash
# Test script for Coverage workflow completeness (Issue #1180, #1289).
#
# Originally written for Issue #1180 when the workflow ran fmt + clippy
# + coverage. Issue #1289 trimmed the workflow to coverage only — the
# fmt and clippy gates are owned by `.github/workflows/ci.yml` and
# running them again here doubled CI runtime for two checks that always
# produce the same result.
#
# The workflow file is now expected to wire up only the coverage path:
#   1. `actions/checkout`        — fetch the source tree
#   2. `dtolnay/rust-toolchain`  — install the Rust toolchain
#   3. `cargo-llvm-cov`          — coverage instrumentation
#   4. `codecov/codecov-action`  — coverage upload
#
# These tests read the real workflow file and assert each pattern is
# present, and additionally assert that the duplicate fmt and clippy
# invocations have been removed.

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

assert_pattern_absent() {
  local description="$1"
  local pattern="$2"
  local file="$3"

  # Strip comment-only lines (starting with optional whitespace then `#`)
  # before searching — the workflow's header comment legitimately
  # mentions the removed steps to explain why they were removed.
  if grep -vE '^[[:space:]]*#' "$file" | grep -qE "$pattern"; then
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

# --- Test: triggers on pull_request ---
assert_pattern_present \
  "pull_request trigger present" \
  "pull_request:" \
  "$WORKFLOW_FILE"

# --- Test: required actions present ---
assert_pattern_present \
  "actions/checkout reference present" \
  "actions/checkout@" \
  "$WORKFLOW_FILE"

assert_pattern_present \
  "dtolnay/rust-toolchain reference present" \
  "dtolnay/rust-toolchain" \
  "$WORKFLOW_FILE"

# --- Test: coverage tooling present ---
assert_pattern_present \
  "cargo-llvm-cov install action present" \
  "cargo-llvm-cov" \
  "$WORKFLOW_FILE"

assert_pattern_present \
  "codecov/codecov-action reference present" \
  "codecov/codecov-action@" \
  "$WORKFLOW_FILE"

# --- Test: contents read permission set ---
assert_pattern_present \
  "contents: read permission present" \
  "contents:[[:space:]]+read" \
  "$WORKFLOW_FILE"

# --- Test: checkout does not persist credentials (Issue #1567) ---
# The coverage job only reads the tree and uploads coverage; it never
# pushes back or fetches private submodules, so the workflow's
# GITHUB_TOKEN must not be written to .git/config where a later
# compromised step could read it.
assert_pattern_present \
  "checkout sets persist-credentials: false" \
  "persist-credentials:[[:space:]]+false" \
  "$WORKFLOW_FILE"

# --- Test: duplicate fmt/clippy gates removed (Issue #1289) ---
# These steps live in ci.yml/quality and re-running them here doubles
# CI runtime for no extra signal. Their absence is part of the
# workflow's contract now.
assert_pattern_absent \
  "cargo fmt --check no longer invoked (handled by ci.yml/quality)" \
  "cargo[[:space:]]+fmt[[:space:]]+--check" \
  "$WORKFLOW_FILE"

assert_pattern_absent \
  "cargo clippy no longer invoked (handled by ci.yml/quality)" \
  "cargo[[:space:]]+clippy" \
  "$WORKFLOW_FILE"

# --- Summary ---
echo ""
echo "Results: $PASS passed, $FAIL failed"

if [ "$FAIL" -gt 0 ]; then
  echo "❌ Coverage workflow validation failed"
  exit 1
fi

echo "✅ All Coverage workflow validation tests passed"
