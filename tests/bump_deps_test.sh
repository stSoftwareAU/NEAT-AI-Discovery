#!/bin/bash
set -euo pipefail

# Tests for ./bump-deps.sh
#
# Exercises the script's helper functions and CLI surface against real inputs.
# Network-dependent paths are tested via dry-run / no-op modes that do not
# require crates.io access. The TDD contract for this issue (#1156) is:
#
#   1. Script exists, is executable, and supports --help and --version.
#   2. Quarantine hours default to 24 and can be overridden by env var.
#   3. Internal stSoftwareAU/* deps are detected (none here, so no-op message).
#   4. External cargo deps respect VIBE_BUMP_QUARANTINE_HOURS.
#   5. Audit gate (cargo deny check) is invoked; failure ⇒ non-zero exit.
#   6. Lockfile integrity is verified after bumping.
#   7. A one-line summary is printed (or "no bumps").

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUMP_DEPS="$PROJECT_ROOT/bump-deps.sh"

PASS=0
FAIL=0
ERRORS=""

assert_exit_code() {
    local description="$1"
    local expected="$2"
    local actual="$3"
    if [[ "$actual" -eq "$expected" ]]; then
        echo "  PASS: $description"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $description (expected exit $expected, got $actual)"
        FAIL=$((FAIL + 1))
        ERRORS="${ERRORS}  FAIL: ${description}\n"
    fi
}

assert_output_contains() {
    local description="$1"
    local expected_pattern="$2"
    local output="$3"
    if echo "$output" | grep -qE "$expected_pattern"; then
        echo "  PASS: $description"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $description (output did not match '$expected_pattern')"
        FAIL=$((FAIL + 1))
        ERRORS="${ERRORS}  FAIL: ${description}\n"
    fi
}

# Reserved for future negative-match assertions; keep here so adding new
# tests does not require re-introducing the helper.
# shellcheck disable=SC2329
assert_output_not_contains() {
    local description="$1"
    local pattern="$2"
    local output="$3"
    if echo "$output" | grep -qE "$pattern"; then
        echo "  FAIL: $description (output unexpectedly matched '$pattern')"
        FAIL=$((FAIL + 1))
        ERRORS="${ERRORS}  FAIL: ${description}\n"
    else
        echo "  PASS: $description"
        PASS=$((PASS + 1))
    fi
}

echo "bump-deps.sh Tests"
echo "==================="
echo ""

# ── Test 1: Script exists and is executable ───────────────────────────

echo "Test 1: Script exists and is executable"
if [[ -x "$BUMP_DEPS" ]]; then
    echo "  PASS: bump-deps.sh is executable"
    PASS=$((PASS + 1))
else
    echo "  FAIL: bump-deps.sh is missing or not executable"
    FAIL=$((FAIL + 1))
    ERRORS="${ERRORS}  FAIL: not executable\n"
fi
echo ""

# ── Test 2: --help exits 0 with usage text ────────────────────────────

echo "Test 2: --help prints usage and exits 0"
set +e
OUTPUT=$("$BUMP_DEPS" --help 2>&1)
EXIT_CODE=$?
set -e
assert_exit_code "--help exits 0" 0 "$EXIT_CODE"
assert_output_contains "--help shows Usage" "Usage:" "$OUTPUT"
assert_output_contains "--help mentions VIBE_BUMP_QUARANTINE_HOURS" "VIBE_BUMP_QUARANTINE_HOURS" "$OUTPUT"
echo ""

# ── Test 3: -h is an alias ────────────────────────────────────────────

echo "Test 3: -h alias for --help"
set +e
OUTPUT=$("$BUMP_DEPS" -h 2>&1)
EXIT_CODE=$?
set -e
assert_exit_code "-h exits 0" 0 "$EXIT_CODE"
assert_output_contains "-h shows Usage" "Usage:" "$OUTPUT"
echo ""

# ── Test 4: Unknown flag exits non-zero ───────────────────────────────

echo "Test 4: unknown flag exits non-zero"
set +e
OUTPUT=$("$BUMP_DEPS" --not-a-real-flag 2>&1)
EXIT_CODE=$?
set -e
if [[ "$EXIT_CODE" -ne 0 ]]; then
    echo "  PASS: unknown flag exits non-zero ($EXIT_CODE)"
    PASS=$((PASS + 1))
else
    echo "  FAIL: unknown flag should exit non-zero but exited 0"
    FAIL=$((FAIL + 1))
    ERRORS="${ERRORS}  FAIL: unknown flag accepted\n"
fi
assert_output_contains "unknown flag prints error" "[Uu]nknown" "$OUTPUT"
echo ""

# ── Test 5: --print-config reports defaults ───────────────────────────

echo "Test 5: --print-config reports default quarantine hours"
set +e
OUTPUT=$("$BUMP_DEPS" --print-config 2>&1)
EXIT_CODE=$?
set -e
assert_exit_code "--print-config exits 0" 0 "$EXIT_CODE"
assert_output_contains "--print-config shows 24h default" "quarantine_hours[[:space:]]*=[[:space:]]*24" "$OUTPUT"
echo ""

# ── Test 6: VIBE_BUMP_QUARANTINE_HOURS env override ───────────────────

echo "Test 6: env var override is reflected in --print-config"
set +e
OUTPUT=$(VIBE_BUMP_QUARANTINE_HOURS=48 "$BUMP_DEPS" --print-config 2>&1)
EXIT_CODE=$?
set -e
assert_exit_code "--print-config exits 0 with env override" 0 "$EXIT_CODE"
assert_output_contains "env override surfaces 48" "quarantine_hours[[:space:]]*=[[:space:]]*48" "$OUTPUT"
echo ""

# ── Test 7: --print-config detects manifest ───────────────────────────

echo "Test 7: --print-config reports detected manifests"
set +e
OUTPUT=$("$BUMP_DEPS" --print-config 2>&1)
EXIT_CODE=$?
set -e
assert_exit_code "--print-config exits 0" 0 "$EXIT_CODE"
assert_output_contains "Cargo manifest detected" "Cargo\\.toml" "$OUTPUT"
echo ""

# ── Test 8: list-internal-deps reports stSoftwareAU/* deps ────────────

echo "Test 8: --list-internal-deps reports the stSoftwareAU/* dep set"
set +e
OUTPUT=$("$BUMP_DEPS" --list-internal-deps 2>&1)
EXIT_CODE=$?
set -e
assert_exit_code "--list-internal-deps exits 0" 0 "$EXIT_CODE"
# This repo has no stSoftwareAU/* code deps — should report empty set, not crash.
assert_output_contains "no internal deps message" "(no internal deps|0 internal deps)" "$OUTPUT"
echo ""

# ── Test 9: helper sourcing — version comparator ──────────────────────

echo "Test 9: source-mode helpers can be invoked directly"
set +e
# Source the script in helper-only mode (no main run).
# shellcheck disable=SC1090
OUTPUT=$(BUMP_DEPS_SOURCE_ONLY=1 bash -c "source '$BUMP_DEPS' && bump_deps::is_quarantine_expired 100 50 25 && echo OK || echo FAIL" 2>&1)
EXIT_CODE=$?
set -e
assert_exit_code "helper sourcing exits 0" 0 "$EXIT_CODE"
# 100 (now) - 50 (published) = 50 hours elapsed, threshold 25h ⇒ expired ⇒ true ⇒ OK
assert_output_contains "is_quarantine_expired returns true when elapsed > threshold" "OK" "$OUTPUT"

set +e
OUTPUT=$(BUMP_DEPS_SOURCE_ONLY=1 bash -c "source '$BUMP_DEPS' && bump_deps::is_quarantine_expired 100 90 25 && echo OK || echo FAIL" 2>&1)
EXIT_CODE=$?
set -e
# 100 - 90 = 10 hours, threshold 25 ⇒ NOT expired ⇒ false ⇒ FAIL
assert_output_contains "is_quarantine_expired returns false when elapsed < threshold" "FAIL" "$OUTPUT"
echo ""

# ── Test 10: --dry-run reports a plan, makes no changes ───────────────

echo "Test 10: --dry-run leaves Cargo.toml/Cargo.lock unchanged"
BEFORE_CARGO_HASH=$(shasum "$PROJECT_ROOT/Cargo.toml" | awk '{print $1}')
BEFORE_LOCK_HASH=$(shasum "$PROJECT_ROOT/Cargo.lock" | awk '{print $1}')
set +e
OUTPUT=$("$BUMP_DEPS" --dry-run 2>&1)
EXIT_CODE=$?
set -e
AFTER_CARGO_HASH=$(shasum "$PROJECT_ROOT/Cargo.toml" | awk '{print $1}')
AFTER_LOCK_HASH=$(shasum "$PROJECT_ROOT/Cargo.lock" | awk '{print $1}')
assert_exit_code "--dry-run exits 0" 0 "$EXIT_CODE"
if [[ "$BEFORE_CARGO_HASH" == "$AFTER_CARGO_HASH" ]]; then
    echo "  PASS: Cargo.toml unchanged in dry-run"
    PASS=$((PASS + 1))
else
    echo "  FAIL: Cargo.toml mutated in dry-run"
    FAIL=$((FAIL + 1))
    ERRORS="${ERRORS}  FAIL: Cargo.toml mutated\n"
fi
if [[ "$BEFORE_LOCK_HASH" == "$AFTER_LOCK_HASH" ]]; then
    echo "  PASS: Cargo.lock unchanged in dry-run"
    PASS=$((PASS + 1))
else
    echo "  FAIL: Cargo.lock mutated in dry-run"
    FAIL=$((FAIL + 1))
    ERRORS="${ERRORS}  FAIL: Cargo.lock mutated\n"
fi
assert_output_contains "--dry-run prints summary" "(no bumps|bumped|would bump|plan)" "$OUTPUT"
echo ""

# ── Test 11: --no-network --dry-run is offline-safe ───────────────────

echo "Test 11: --no-network --dry-run does not fail when offline"
set +e
OUTPUT=$("$BUMP_DEPS" --dry-run --no-network 2>&1)
EXIT_CODE=$?
set -e
assert_exit_code "--no-network --dry-run exits 0" 0 "$EXIT_CODE"
assert_output_contains "no-network mode reports skipped network" "(skipping network|--no-network|offline)" "$OUTPUT"
echo ""

# ── Test 12: idempotency — re-run after a clean run is a no-op ────────

echo "Test 12: re-running --dry-run twice yields the same exit code"
set +e
"$BUMP_DEPS" --dry-run --no-network >/dev/null 2>&1
EXIT_CODE_A=$?
"$BUMP_DEPS" --dry-run --no-network >/dev/null 2>&1
EXIT_CODE_B=$?
set -e
if [[ "$EXIT_CODE_A" -eq "$EXIT_CODE_B" ]]; then
    echo "  PASS: dry-run is deterministic ($EXIT_CODE_A == $EXIT_CODE_B)"
    PASS=$((PASS + 1))
else
    echo "  FAIL: dry-run not deterministic ($EXIT_CODE_A vs $EXIT_CODE_B)"
    FAIL=$((FAIL + 1))
    ERRORS="${ERRORS}  FAIL: not deterministic\n"
fi
echo ""

# ── Test 13: invalid VIBE_BUMP_QUARANTINE_HOURS rejected ──────────────

echo "Test 13: non-numeric VIBE_BUMP_QUARANTINE_HOURS is rejected"
set +e
OUTPUT=$(VIBE_BUMP_QUARANTINE_HOURS=abc "$BUMP_DEPS" --print-config 2>&1)
EXIT_CODE=$?
set -e
if [[ "$EXIT_CODE" -ne 0 ]]; then
    echo "  PASS: invalid quarantine hours rejected ($EXIT_CODE)"
    PASS=$((PASS + 1))
else
    echo "  FAIL: invalid quarantine hours accepted"
    FAIL=$((FAIL + 1))
    ERRORS="${ERRORS}  FAIL: invalid hours accepted\n"
fi
assert_output_contains "rejection mentions hours" "(numeric|integer|VIBE_BUMP_QUARANTINE_HOURS)" "$OUTPUT"
echo ""

# ── Summary ──────────────────────────────────────────────────────────

echo ""
echo "==================="
echo "Test Summary"
echo "==================="
echo "Passed: $PASS"
echo "Failed: $FAIL"

if [[ $FAIL -gt 0 ]]; then
    echo ""
    echo "Failures:"
    echo -e "$ERRORS"
    exit 1
fi

echo ""
echo "All tests passed."
exit 0
