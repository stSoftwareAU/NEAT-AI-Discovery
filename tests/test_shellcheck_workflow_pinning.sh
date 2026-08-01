#!/usr/bin/env bash
# Test that .github/workflows/shellcheck.yml pins third-party actions to
# a 40-character commit SHA rather than a mutable branch like `@master`
# (Issue #1215).
#
# Pinning to a mutable branch lets any push to upstream `master` execute
# in this repository's CI on the next PR, with access to GITHUB_TOKEN.
# The coding guidelines mandate SHA pins for third-party actions.
#
# Issue #1898 removed the unmaintained `ludeeus/action-shellcheck` wrapper, so
# these checks are no longer written against that one action by name: they now
# assert the pinning invariant over *every* action the workflow uses, which is
# what Issue #1215 actually required.

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

# Every `uses:` reference in the workflow, with comments and indentation
# stripped back to `<action>@<ref> # <comment>`.
uses_lines=$(grep -E '^[[:space:]]*-?[[:space:]]*uses:' "$WORKFLOW_FILE" || true)

# --- Test: the workflow declares at least one action to check ---
if [ -n "$uses_lines" ]; then
  pass
else
  fail "$WORKFLOW_FILE declares no actions — expected at least actions/checkout"
fi

# --- Test: the unmaintained ludeeus wrapper has not come back (Issue #1898) ---
if printf '%s\n' "$uses_lines" | grep -q 'ludeeus/action-shellcheck'; then
  fail "the unmaintained ludeeus/action-shellcheck wrapper is back in $WORKFLOW_FILE"
else
  pass
fi

# --- Test: no action is pinned to a mutable branch or tag ---
if printf '%s\n' "$uses_lines" | grep -qvE 'uses:[[:space:]]*[^@[:space:]]+@[0-9a-f]{40}\b'; then
  fail "an action is not pinned to a 40-character commit SHA:
$(printf '%s\n' "$uses_lines" | grep -vE 'uses:[[:space:]]*[^@[:space:]]+@[0-9a-f]{40}\b')"
else
  pass
fi

# --- Test: every SHA pin carries a trailing human-readable version comment ---
# Per the coding guideline: "Add the resolved tag in a trailing comment so
# the version is auditable."
if printf '%s\n' "$uses_lines" | grep -qvE '@[0-9a-f]{40}[[:space:]]+#[[:space:]]*[^[:space:]]'; then
  fail "an action's SHA pin lacks a trailing version comment:
$(printf '%s\n' "$uses_lines" | grep -vE '@[0-9a-f]{40}[[:space:]]+#[[:space:]]*[^[:space:]]')"
else
  pass
fi

echo ""
echo "Results: $PASS passed, $FAIL failed"

if [ "$FAIL" -gt 0 ]; then
  echo "❌ Shellcheck workflow pinning validation failed"
  exit 1
fi

echo "✅ Shellcheck workflow pinning validation passed"
