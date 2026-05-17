#!/usr/bin/env bash
# Test that .github/workflows/shellcheck.yml pins third-party actions to
# a 40-character commit SHA rather than a mutable branch like `@master`
# (Issue #1215).
#
# Pinning to a mutable branch lets any push to upstream `master` execute
# in this repository's CI on the next PR, with access to GITHUB_TOKEN.
# The coding guidelines mandate SHA pins for third-party actions.

set -euo pipefail

WORKFLOW_FILE=".github/workflows/shellcheck.yml"
PASS=0
FAIL=0

fail() {
  echo "FAIL: $1"
  FAIL=$((FAIL + 1))
}

pass() {
  PASS=$((PASS + 1))
}

if [ ! -f "$WORKFLOW_FILE" ]; then
  echo "FAIL: $WORKFLOW_FILE not found"
  exit 1
fi

# --- Test: no @master / @main pin for ludeeus/action-shellcheck ---
if grep -qE 'ludeeus/action-shellcheck@(master|main)\b' "$WORKFLOW_FILE"; then
  fail "ludeeus/action-shellcheck is pinned to a mutable branch (@master or @main)"
else
  pass
fi

# --- Test: ludeeus/action-shellcheck is pinned to a 40-char SHA ---
if grep -qE 'ludeeus/action-shellcheck@[0-9a-f]{40}\b' "$WORKFLOW_FILE"; then
  pass
else
  fail "ludeeus/action-shellcheck is not pinned to a 40-character commit SHA"
fi

# --- Test: a trailing comment records the human-readable version ---
# Per the coding guideline: "Add the resolved tag in a trailing comment so
# the version is auditable."
if grep -qE 'ludeeus/action-shellcheck@[0-9a-f]{40}[[:space:]]+#' "$WORKFLOW_FILE"; then
  pass
else
  fail "ludeeus/action-shellcheck SHA pin lacks a trailing version comment"
fi

echo ""
echo "Results: $PASS passed, $FAIL failed"

if [ "$FAIL" -gt 0 ]; then
  echo "❌ Shellcheck workflow pinning validation failed"
  exit 1
fi

echo "✅ Shellcheck workflow pinning validation passed"
