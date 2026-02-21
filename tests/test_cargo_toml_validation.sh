#!/usr/bin/env bash
# Test script for stricter Cargo.toml field validation (Issue #691)
#
# Validates that the CI grep pattern:
# 1. Skips comment lines (lines starting with optional whitespace + #)
# 2. Requires exact field match at start of line (e.g. "version" must not match "rust-version")
# 3. Accepts valid uncommented fields

set -euo pipefail

PASS=0
FAIL=0

assert_detected() {
  local description="$1"
  local toml_content="$2"
  local field="$3"

  if echo "$toml_content" | grep -vE "^[[:space:]]*#" | grep -qE "^[[:space:]]*${field}[[:space:]]*="; then
    PASS=$((PASS + 1))
  else
    echo "FAIL: $description — expected field '$field' to be detected"
    FAIL=$((FAIL + 1))
  fi
}

assert_not_detected() {
  local description="$1"
  local toml_content="$2"
  local field="$3"

  if echo "$toml_content" | grep -vE "^[[:space:]]*#" | grep -qE "^[[:space:]]*${field}[[:space:]]*="; then
    echo "FAIL: $description — expected field '$field' NOT to be detected"
    FAIL=$((FAIL + 1))
  else
    PASS=$((PASS + 1))
  fi
}

# --- Test: uncommented fields are detected ---
assert_detected \
  "Uncommented name field" \
  'name = "my-crate"' \
  "name"

assert_detected \
  "Uncommented version field" \
  'version = "1.0.0"' \
  "version"

assert_detected \
  "Uncommented edition field" \
  'edition = "2021"' \
  "edition"

# --- Test: commented-out fields are NOT detected ---
assert_not_detected \
  "Commented-out name field" \
  '# name = "my-crate"' \
  "name"

assert_not_detected \
  "Commented-out version field with leading whitespace" \
  '  # version = "1.0.0"' \
  "version"

assert_not_detected \
  "Commented-out edition field" \
  '# edition = "2021"' \
  "edition"

# --- Test: substring fields must NOT match ---
assert_not_detected \
  "rust-version must not match version" \
  'rust-version = "1.70"' \
  "version"

assert_not_detected \
  "build-edition must not match edition" \
  'build-edition = "2021"' \
  "edition"

# --- Test: fields with varying whitespace ---
assert_detected \
  "Field with spaces around equals" \
  'name   =   "my-crate"' \
  "name"

assert_detected \
  "Field with leading whitespace" \
  '  version = "1.0.0"' \
  "version"

assert_detected \
  "Field with tab before equals" \
  "$(printf 'edition\t= \"2021\"')" \
  "edition"

# --- Test: real Cargo.toml has required fields ---
if [ -f Cargo.toml ]; then
  for field in "name" "version" "edition"; do
    assert_detected \
      "Real Cargo.toml contains $field" \
      "$(cat Cargo.toml)" \
      "$field"
  done
fi

# --- Test: old pattern false positives (regression) ---
# The old pattern `grep -q "$field = "` would match these incorrectly
assert_not_detected \
  "Old pattern regression: commented name" \
  '# name = "commented-out"' \
  "name"

assert_not_detected \
  "Old pattern regression: rust-version matching version" \
  'rust-version = "1.70"' \
  "version"

# --- Summary ---
echo ""
echo "Results: $PASS passed, $FAIL failed"

if [ "$FAIL" -gt 0 ]; then
  echo "❌ Some validation tests failed"
  exit 1
fi

echo "✅ All validation tests passed"
