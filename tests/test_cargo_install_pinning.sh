#!/usr/bin/env bash
# Test script for cargo install pinning in CI workflows (Issue #1223).
#
# Every `cargo install <plugin>` call site in `.github/workflows/` must:
#   1. Pass `--locked` so the plugin's own Cargo.lock is honoured.
#   2. Pass `--version <X.Y.Z>` so the plugin itself is pinned to a
#      reviewed release rather than re-resolving from the registry on
#      each run.
#
# Without both flags, a poisoned new release of the plugin or any of
# its transitive dependencies would execute via build.rs on the CI
# runner, with access to the workflow's GITHUB_TOKEN (and any other
# secrets in the environment).
#
# This test reads the real workflow files and asserts that every
# `cargo install` invocation carries both flags.

set -euo pipefail

WORKFLOW_DIR=".github/workflows"
PASS=0
FAIL=0

assert_pinned() {
  local file="$1"
  local line_no="$2"
  local line="$3"

  # Must include --locked
  if ! echo "$line" | grep -qE -- '--locked'; then
    echo "FAIL: $file:$line_no — 'cargo install' missing --locked: $line"
    FAIL=$((FAIL + 1))
    return
  fi

  # Must include --version <X.Y.Z>
  if ! echo "$line" | grep -qE -- '--version[ =][0-9]+\.[0-9]+\.[0-9]+'; then
    echo "FAIL: $file:$line_no — 'cargo install' missing --version pin: $line"
    FAIL=$((FAIL + 1))
    return
  fi

  PASS=$((PASS + 1))
}

if [ ! -d "$WORKFLOW_DIR" ]; then
  echo "FAIL: workflow directory $WORKFLOW_DIR not found"
  exit 1
fi

# Scan every workflow YAML for `cargo install` invocations.
found_any=0
while IFS= read -r workflow; do
  while IFS=: read -r line_no line; do
    found_any=1
    assert_pinned "$workflow" "$line_no" "$line"
  done < <(grep -nE 'cargo[[:space:]]+install[[:space:]]+' "$workflow" || true)
done < <(find "$WORKFLOW_DIR" -type f \( -name '*.yml' -o -name '*.yaml' \))

if [ "$found_any" -eq 0 ]; then
  echo "OK: no 'cargo install' invocations found in workflows — nothing to pin"
fi

echo ""
echo "Results: $PASS passed, $FAIL failed"

if [ "$FAIL" -gt 0 ]; then
  echo "❌ cargo install pinning validation failed"
  exit 1
fi

echo "✅ All cargo install invocations are pinned with --locked and --version"
