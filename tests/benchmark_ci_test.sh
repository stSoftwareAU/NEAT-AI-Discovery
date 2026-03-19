#!/bin/bash
set -euo pipefail

# Tests for scripts/benchmark-ci.sh
#
# Runs the benchmark CI script in various modes and verifies correct behaviour.
# These tests exercise the real script with real arguments and check exit codes
# and output.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BENCHMARK_CI="$PROJECT_ROOT/scripts/benchmark-ci.sh"

PASS=0
FAIL=0
ERRORS=""

# ── Test helpers ──────────────────────────────────────────────────────

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
    if echo "$output" | grep -q "$expected_pattern"; then
        echo "  PASS: $description"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $description (output did not contain '$expected_pattern')"
        FAIL=$((FAIL + 1))
        ERRORS="${ERRORS}  FAIL: ${description}\n"
    fi
}

echo "Benchmark CI Script Tests"
echo "========================="
echo ""

# ── Test 1: Script exists and is executable ───────────────────────────

echo "Test 1: Script exists and is executable"
if [[ -x "$BENCHMARK_CI" ]]; then
    echo "  PASS: Script is executable"
    PASS=$((PASS + 1))
else
    echo "  FAIL: Script is not executable"
    FAIL=$((FAIL + 1))
    ERRORS="${ERRORS}  FAIL: Script is not executable\n"
fi
echo ""

# ── Test 2: --help flag shows usage and exits 0 ──────────────────────

echo "Test 2: --help shows usage"
set +e
OUTPUT=$("$BENCHMARK_CI" --help 2>&1)
EXIT_CODE=$?
set -e
assert_exit_code "--help exits with 0" 0 "$EXIT_CODE"
assert_output_contains "--help shows Usage" "Usage:" "$OUTPUT"
assert_output_contains "--help mentions --compile-only" "compile-only" "$OUTPUT"
assert_output_contains "--help mentions --compare" "compare" "$OUTPUT"
assert_output_contains "--help mentions --threshold" "threshold" "$OUTPUT"
assert_output_contains "--help mentions --list" "list" "$OUTPUT"
echo ""

# ── Test 3: Unknown option exits with error ──────────────────────────

echo "Test 3: Unknown option exits with error"
set +e
OUTPUT=$("$BENCHMARK_CI" --invalid-flag 2>&1)
EXIT_CODE=$?
set -e
assert_exit_code "Unknown option exits non-zero" 1 "$EXIT_CODE"
assert_output_contains "Unknown option shows error" "Unknown option" "$OUTPUT"
echo ""

# ── Test 4: --threshold without value exits with error ───────────────

echo "Test 4: --threshold without value exits with error"
set +e
OUTPUT=$("$BENCHMARK_CI" --threshold 2>&1)
EXIT_CODE=$?
set -e
assert_exit_code "--threshold without value exits 1" 1 "$EXIT_CODE"
assert_output_contains "Shows threshold error" "requires a value" "$OUTPUT"
echo ""

# ── Test 5: Invalid threshold (non-numeric) exits with error ─────────

echo "Test 5: Invalid threshold exits with error"
set +e
OUTPUT=$("$BENCHMARK_CI" --threshold abc 2>&1)
EXIT_CODE=$?
set -e
assert_exit_code "Non-numeric threshold exits 1" 1 "$EXIT_CODE"
assert_output_contains "Shows threshold validation error" "must be a positive integer" "$OUTPUT"
echo ""

# ── Test 6: --list discovers benchmarks from Cargo.toml ──────────────

echo "Test 6: --list discovers benchmarks from Cargo.toml"
set +e
OUTPUT=$("$BENCHMARK_CI" --list 2>&1)
EXIT_CODE=$?
set -e
assert_exit_code "--list exits with 0" 0 "$EXIT_CODE"
assert_output_contains "Shows discovered count" "Discovered benchmark suites" "$OUTPUT"
assert_output_contains "Lists synapse_counts" "synapse_counts" "$OUTPUT"
assert_output_contains "Lists neuron_interning" "neuron_interning" "$OUTPUT"
assert_output_contains "Lists gpu_buffer_transfers" "gpu_buffer_transfers" "$OUTPUT"
echo ""

# ── Test 7: Default mode is compile ──────────────────────────────────

echo "Test 7: BENCHMARK_CI_MODE env var is respected"
set +e
OUTPUT=$(BENCHMARK_CI_MODE="invalid_mode" "$BENCHMARK_CI" 2>&1)
EXIT_CODE=$?
set -e
assert_exit_code "Invalid BENCHMARK_CI_MODE exits 1" 1 "$EXIT_CODE"
assert_output_contains "Shows unknown mode error" "Unknown mode" "$OUTPUT"
echo ""

# ── Test 8: --compare mode shows threshold in header ──────────────────
# Note: We cannot fully test --compare mode as it requires benchmark compilation.
# Instead, we verify the script accepts the flags without error by checking
# that it at least prints the mode header before attempting any compilation.

echo "Test 8: --compile-only flag is accepted"
set +e
# Use timeout to avoid blocking on actual compilation — just verify the script starts correctly
OUTPUT=$(timeout 5 "$BENCHMARK_CI" --compile-only 2>&1) || true
set -e
assert_output_contains "Compile mode shown in header" "Mode: compile" "$OUTPUT"
assert_output_contains "Shows benchmark suite count" "Benchmark suites:" "$OUTPUT"
echo ""

# ── Test 9: --list discovers expected number of benchmarks ────────────

echo "Test 9: --list finds multiple benchmark suites"
set +e
OUTPUT=$("$BENCHMARK_CI" --list 2>&1)
EXIT_CODE=$?
set -e
# Count the number of benchmark entries (lines starting with "  - ")
BENCH_COUNT=$(echo "$OUTPUT" | grep -c "^  - " || true)
if [[ "$BENCH_COUNT" -ge 10 ]]; then
    echo "  PASS: Found $BENCH_COUNT benchmark suites (expected >= 10)"
    PASS=$((PASS + 1))
else
    echo "  FAIL: Found only $BENCH_COUNT benchmark suites (expected >= 10)"
    FAIL=$((FAIL + 1))
    ERRORS="${ERRORS}  FAIL: Found only ${BENCH_COUNT} benchmark suites\n"
fi
echo ""

# ── Test 10: -h is an alias for --help ────────────────────────────────

echo "Test 10: -h is an alias for --help"
set +e
OUTPUT=$("$BENCHMARK_CI" -h 2>&1)
EXIT_CODE=$?
set -e
assert_exit_code "-h exits with 0" 0 "$EXIT_CODE"
assert_output_contains "-h shows Usage" "Usage:" "$OUTPUT"
echo ""

# ── Summary ──────────────────────────────────────────────────────────

echo ""
echo "========================="
echo "Test Summary"
echo "========================="
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
