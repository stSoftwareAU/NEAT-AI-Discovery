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

# ── Test 14: extract_dep_versions parses inline + table form ──────────

echo "Test 14: extract_dep_versions emits name<TAB>version lines"
TMP_TOML=$(mktemp)
cat > "$TMP_TOML" <<'TOML'
[package]
name = "demo"

[dependencies]
serde = "1.0"
parquet = "58.3"  # trailing comment
arrow = { version = "58.3", default-features = false }
tracing-subscriber = { version = "0.3", features = ["env-filter"] }

[dev-dependencies]
tempfile = "3.27"
TOML
set +e
OUTPUT=$(BUMP_DEPS_SOURCE_ONLY=1 bash -c "source '$BUMP_DEPS' && bump_deps::extract_dep_versions '$TMP_TOML'" 2>&1)
EXIT_CODE=$?
set -e
rm -f "$TMP_TOML"
assert_exit_code "extract_dep_versions exits 0" 0 "$EXIT_CODE"
assert_output_contains "serde 1.0 detected" "^serde	1\\.0$" "$OUTPUT"
assert_output_contains "parquet 58.3 detected (comment stripped)" "^parquet	58\\.3$" "$OUTPUT"
assert_output_contains "arrow inline table version" "^arrow	58\\.3$" "$OUTPUT"
assert_output_contains "tracing-subscriber inline table" "^tracing-subscriber	0\\.3$" "$OUTPUT"
assert_output_contains "tempfile dev-dep detected" "^tempfile	3\\.27$" "$OUTPUT"
echo ""

# ── Test 15: compute_changed_deps emits name<TAB>old<TAB>new ──────────

echo "Test 15: compute_changed_deps reports only changed deps"
BEFORE_FILE=$(mktemp)
AFTER_FILE=$(mktemp)
printf 'serde\t1.0.190\nparquet\t58.3.0\narrow\t58.3.0\n' > "$BEFORE_FILE"
printf 'serde\t1.0.225\nparquet\t58.3.0\narrow\t58.4.0\n' > "$AFTER_FILE"
set +e
OUTPUT=$(BUMP_DEPS_SOURCE_ONLY=1 bash -c "source '$BUMP_DEPS' && bump_deps::compute_changed_deps '$BEFORE_FILE' '$AFTER_FILE'" 2>&1)
EXIT_CODE=$?
set -e
rm -f "$BEFORE_FILE" "$AFTER_FILE"
assert_exit_code "compute_changed_deps exits 0" 0 "$EXIT_CODE"
assert_output_contains "serde change reported" "^serde	1\\.0\\.190	1\\.0\\.225$" "$OUTPUT"
assert_output_contains "arrow change reported" "^arrow	58\\.3\\.0	58\\.4\\.0$" "$OUTPUT"
# Parquet did NOT change — must not appear.
if echo "$OUTPUT" | grep -qE '^parquet'; then
    echo "  FAIL: unchanged dep reported as changed"
    FAIL=$((FAIL + 1))
    ERRORS="${ERRORS}  FAIL: unchanged dep reported\n"
else
    echo "  PASS: unchanged dep is not reported"
    PASS=$((PASS + 1))
fi
echo ""

# ── Test 16: revert_dep_line restores a single version ────────────────

echo "Test 16: revert_dep_line restores the old version string"
TMP_TOML=$(mktemp)
cat > "$TMP_TOML" <<'TOML'
[dependencies]
serde = "1.0.225"
arrow = { version = "58.4.0", default-features = false }
parquet = "58.3.0"
TOML
set +e
BUMP_DEPS_SOURCE_ONLY=1 bash -c "source '$BUMP_DEPS' && bump_deps::revert_dep_line '$TMP_TOML' serde '1.0.190'" >/dev/null 2>&1
EXIT_A=$?
BUMP_DEPS_SOURCE_ONLY=1 bash -c "source '$BUMP_DEPS' && bump_deps::revert_dep_line '$TMP_TOML' arrow '58.3.0'" >/dev/null 2>&1
EXIT_B=$?
set -e
OUTPUT=$(cat "$TMP_TOML")
rm -f "$TMP_TOML"
assert_exit_code "revert serde exits 0" 0 "$EXIT_A"
assert_exit_code "revert arrow exits 0" 0 "$EXIT_B"
assert_output_contains "serde reverted to 1.0.190" 'serde[[:space:]]*=[[:space:]]*"1\.0\.190"' "$OUTPUT"
assert_output_contains "arrow reverted to 58.3.0" 'version[[:space:]]*=[[:space:]]*"58\.3\.0"' "$OUTPUT"
assert_output_contains "parquet untouched at 58.3.0" 'parquet[[:space:]]*=[[:space:]]*"58\.3\.0"' "$OUTPUT"
echo ""

# ── Test 17: fetch_publish_epoch uses test fixture seam ───────────────

echo "Test 17: fetch_publish_epoch reads BUMP_DEPS_TEST_FIXTURE when set"
FIX_DIR=$(mktemp -d)
# crates.io response shape: nested 'version' object with created_at.
cat > "$FIX_DIR/serde-1.0.225.json" <<'JSON'
{"version":{"num":"1.0.225","created_at":"2025-05-19T12:00:00.000000+00:00"}}
JSON
set +e
OUTPUT=$(BUMP_DEPS_SOURCE_ONLY=1 BUMP_DEPS_TEST_FIXTURE="$FIX_DIR" \
    bash -c "source '$BUMP_DEPS' && bump_deps::fetch_publish_epoch serde 1.0.225" 2>&1)
EXIT_CODE=$?
set -e
rm -rf "$FIX_DIR"
assert_exit_code "fetch_publish_epoch exits 0 with fixture" 0 "$EXIT_CODE"
# 2025-05-19T12:00:00 UTC = 1747656000 epoch seconds.
assert_output_contains "fetch_publish_epoch returns 2025-05-19T12:00 epoch" "^1747656000$" "$OUTPUT"
echo ""

# ── Test 18: fetch_publish_epoch missing fixture fails non-zero ───────

echo "Test 18: fetch_publish_epoch returns non-zero when fixture missing"
EMPTY_DIR=$(mktemp -d)
set +e
BUMP_DEPS_SOURCE_ONLY=1 BUMP_DEPS_TEST_FIXTURE="$EMPTY_DIR" \
    bash -c "source '$BUMP_DEPS' && bump_deps::fetch_publish_epoch nope 9.9.9" >/dev/null 2>&1
EXIT_CODE=$?
set -e
rmdir "$EMPTY_DIR"
if [[ "$EXIT_CODE" -ne 0 ]]; then
    echo "  PASS: missing fixture yields non-zero exit ($EXIT_CODE)"
    PASS=$((PASS + 1))
else
    echo "  FAIL: missing fixture incorrectly returned 0"
    FAIL=$((FAIL + 1))
    ERRORS="${ERRORS}  FAIL: missing fixture\n"
fi
echo ""

# ── Test 19: current_epoch honours BUMP_DEPS_NOW_EPOCH stub ───────────

echo "Test 19: current_epoch is stubbable via BUMP_DEPS_NOW_EPOCH"
set +e
OUTPUT=$(BUMP_DEPS_SOURCE_ONLY=1 BUMP_DEPS_NOW_EPOCH=1700000000 \
    bash -c "source '$BUMP_DEPS' && bump_deps::current_epoch" 2>&1)
EXIT_CODE=$?
set -e
assert_exit_code "current_epoch exits 0" 0 "$EXIT_CODE"
assert_output_contains "current_epoch returns stub value" "^1700000000$" "$OUTPUT"
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
