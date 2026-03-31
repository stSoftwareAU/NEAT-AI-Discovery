#!/usr/bin/env bash
# Test script for quality.sh dependency upgrade step (Issue #959)
#
# Validates that cargo upgrade --incompatible runs successfully
# as part of the quality workflow.

set -euo pipefail

PASS=0
FAIL=0

assert_pass() {
  local description="$1"
  PASS=$((PASS + 1))
  echo "PASS: $description"
}

assert_fail() {
  local description="$1"
  FAIL=$((FAIL + 1))
  echo "FAIL: $description"
}

# --- Test: cargo-upgrade is available ---
if command -v cargo-upgrade &> /dev/null; then
  assert_pass "cargo-upgrade command is available"
else
  assert_fail "cargo-upgrade command is not available (install with: cargo install cargo-edit)"
fi

# --- Test: cargo upgrade --incompatible --dry-run succeeds ---
if command -v cargo-upgrade &> /dev/null; then
  if cargo upgrade --incompatible --dry-run < /dev/null 2>&1; then
    assert_pass "cargo upgrade --incompatible --dry-run succeeds"
  else
    assert_fail "cargo upgrade --incompatible --dry-run failed"
  fi
else
  echo "SKIP: cargo upgrade --dry-run (cargo-edit not installed)"
fi

# --- Test: Cargo.toml exists and has valid dependency sections ---
if [ -f Cargo.toml ]; then
  assert_pass "Cargo.toml exists"

  if grep -qE '^\[dependencies\]' Cargo.toml; then
    assert_pass "Cargo.toml has [dependencies] section"
  else
    assert_fail "Cargo.toml missing [dependencies] section"
  fi
else
  assert_fail "Cargo.toml not found"
fi

# --- Test: Cargo.lock exists (required for reproducible builds) ---
if [ -f Cargo.lock ]; then
  assert_pass "Cargo.lock exists"
else
  assert_fail "Cargo.lock not found"
fi

# --- Summary ---
echo ""
echo "Results: $PASS passed, $FAIL failed"

if [ "$FAIL" -gt 0 ]; then
  echo "❌ Some tests failed"
  exit 1
fi

echo "✅ All quality upgrade tests passed"
