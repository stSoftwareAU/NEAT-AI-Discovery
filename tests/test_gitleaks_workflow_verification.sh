#!/usr/bin/env bash
# Test that .github/workflows/gitleaks.yml verifies the integrity of the
# downloaded gitleaks binary against a committed SHA-256 hash before
# extracting and executing it (Issue #1217).
#
# Without an integrity check, a tampered release artefact would execute
# inside CI with the runner's privileges. The workflow must therefore
# pin both the version AND the expected SHA-256, and run sha256sum -c
# between download and extract.

set -euo pipefail

WORKFLOW_FILE=".github/workflows/gitleaks.yml"
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

# --- Test: workflow declares an EXPECTED_SHA256 variable ---
if grep -qE 'EXPECTED_SHA256=' "$WORKFLOW_FILE"; then
  pass
else
  fail "workflow does not declare an EXPECTED_SHA256 variable"
fi

# --- Test: SHA-256 is a 64-character lowercase hex string ---
if grep -qE 'EXPECTED_SHA256="[0-9a-f]{64}"' "$WORKFLOW_FILE"; then
  pass
else
  fail "EXPECTED_SHA256 is not a 64-character lowercase hex string"
fi

# --- Test: SHA-256 matches the published gitleaks 8.24.3 linux_x64 hash ---
EXPECTED_LINUX_X64_SHA="9991e0b2903da4c8f6122b5c3186448b927a5da4deef1fe45271c3793f4ee29c"
if grep -qE "EXPECTED_SHA256=\"${EXPECTED_LINUX_X64_SHA}\"" "$WORKFLOW_FILE"; then
  pass
else
  fail "EXPECTED_SHA256 does not match the published gitleaks 8.24.3 linux_x64 checksum (${EXPECTED_LINUX_X64_SHA})"
fi

# --- Test: workflow invokes sha256sum -c against the downloaded tarball ---
if grep -qE 'sha256sum[[:space:]]+-c' "$WORKFLOW_FILE"; then
  pass
else
  fail "workflow does not run 'sha256sum -c' to verify the downloaded artefact"
fi

# --- Test: verification happens BEFORE extraction ---
# The `sha256sum -c` line must appear in the file before the `tar -xzf` line,
# otherwise a tampered tarball would be extracted before verification.
SHA_LINE=$(grep -nE 'sha256sum[[:space:]]+-c' "$WORKFLOW_FILE" | head -1 | cut -d: -f1 || true)
TAR_LINE=$(grep -nE 'tar[[:space:]]+-xzf' "$WORKFLOW_FILE" | head -1 | cut -d: -f1 || true)
if [ -n "$SHA_LINE" ] && [ -n "$TAR_LINE" ] && [ "$SHA_LINE" -lt "$TAR_LINE" ]; then
  pass
else
  fail "sha256sum verification must occur before tar extraction (sha line=$SHA_LINE, tar line=$TAR_LINE)"
fi

# --- Test: version remains pinned (defence in depth alongside hash) ---
if grep -qE 'GITLEAKS_VERSION="[0-9]+\.[0-9]+\.[0-9]+"' "$WORKFLOW_FILE"; then
  pass
else
  fail "GITLEAKS_VERSION is not pinned to a specific semver"
fi

echo ""
echo "Results: $PASS passed, $FAIL failed"

if [ "$FAIL" -gt 0 ]; then
  echo "❌ Gitleaks workflow integrity verification tests failed"
  exit 1
fi

echo "✅ Gitleaks workflow integrity verification tests passed"
