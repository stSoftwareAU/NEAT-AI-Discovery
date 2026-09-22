#!/bin/bash
# GPU Performance Benchmark Script
# Compares baseline vs current implementation
#
# Usage:
#   ./benchmark.sh                    # Run test benchmarks only
#   ./benchmark.sh /path/to/file.parquet  # Also run parquet analysis benchmark
#
# Note: Large parquet files should be stored locally and NOT committed to git.
#       Add them to .gitignore: echo "*.parquet" >> .gitignore
#
# Portable across macOS (bash 3.2, BSD userland), Ubuntu and AWS Linux — the
# clock is read by now_seconds(), never by the GNU-only `date +%s.%N`.

set -euo pipefail

PARQUET_FILE="${1:-}"
# Baseline: v0.1.146 (11-Dec-2025) - before GPU processing optimisations
# Recent GPU changes: v0.1.151-153 (12-Dec) introduced GPU timeouts, memory adaptation
BASELINE_COMMIT="bed746a"  # v0.1.146 - last stable before GPU optimisation work

# Print the current time as epoch seconds, sub-second where the host can offer
# it (Issue #2141). `date +%s.%N` is a GNU extension: BSD `date` — what macOS
# ships — emits the two characters verbatim, so every downstream `bc` sum saw
# `1769040000.N` and returned nothing. Sources are tried best-resolution first
# and the last resort is POSIX; with none of them the function fails loud rather
# than handing an empty string to the arithmetic.
now_seconds() {
    local whole

    # bash >= 5: microseconds, no subprocess. Some locales render the decimal
    # separator as a comma, which `bc`/`printf` reject.
    if [ -n "${EPOCHREALTIME:-}" ]; then
        printf '%s\n' "${EPOCHREALTIME/,/.}"
        return 0
    fi

    # bash 3.2 (macOS) has no EPOCHREALTIME; perl ships with both platforms.
    if command -v perl > /dev/null 2>&1 &&
        perl -MTime::HiRes=time -e 'printf "%.6f\n", time' 2> /dev/null; then
        return 0
    fi

    # POSIX `date +%s` — whole seconds, but a benchmark that runs for minutes
    # still reads meaningfully.
    if whole=$(date +%s 2> /dev/null); then
        case "$whole" in
            '' | *[!0-9]*) ;;
            *)
                printf '%s\n' "$whole"
                return 0
                ;;
        esac
    fi

    echo "❌ benchmark.sh: no usable clock source — need bash 5 (EPOCHREALTIME), perl (Time::HiRes), or a POSIX date (+%s)" >&2
    return 1
}

# Function to run benchmark and capture timing
run_benchmark() {
    local label="$1"
    local cmd="$2"
    local start end duration

    echo "⏱️  Running: $label" >&2
    start=$(now_seconds)
    eval "$cmd" > /dev/null 2>&1 || true
    end=$(now_seconds)
    duration=$(echo "$end - $start" | bc)
    # An empty or malformed subtraction must not reach the summary's `printf`
    # as a silent 0.00s — say what was read instead (Issue #2141).
    if [[ ! "$duration" =~ ^-?[0-9]+(\.[0-9]+)?$ ]]; then
        echo "❌ benchmark.sh: no numeric duration for '$label' — bc returned '${duration}' from '${end} - ${start}' (is bc installed?)" >&2
        return 1
    fi
    echo "   Duration: ${duration}s" >&2
    echo "$duration"
}

# Helper-only mode: the regression tests source this script to exercise
# now_seconds() directly, and must not trigger the benchmark itself — it checks
# out git refs and stashes uncommitted work (Issue #2141). Only honoured when
# the script really is sourced; `return` outside a sourced file is a bash error,
# so an exported variable must not be able to break a normal ./benchmark.sh run.
if [ -n "${BENCHMARK_SOURCE_ONLY:-}" ] && [ "${BASH_SOURCE[0]}" != "$0" ]; then
    return 0
fi

echo "🏁 GPU Performance Benchmark"
echo "=============================="
echo "Date: $(date '+%d-%b-%Y %H:%M:%S')"
echo ""

# Store current branch/commit - handle detached HEAD state
CURRENT_REF=$(git rev-parse --short HEAD)
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

# If in detached HEAD state, abbrev-ref returns "HEAD" - use commit ref instead
if [ "$CURRENT_BRANCH" = "HEAD" ]; then
    CURRENT_BRANCH=""
    RESTORE_REF="$CURRENT_REF"
    echo "⚠️  Running in detached HEAD state - will restore to $CURRENT_REF"
else
    RESTORE_REF="$CURRENT_BRANCH"
fi

HAS_CHANGES=$(git status --porcelain | wc -l | tr -d ' ')

echo "📍 Current: ${CURRENT_BRANCH:-detached} ($CURRENT_REF)"
echo "📍 Baseline: $BASELINE_COMMIT"
echo ""

# Stash changes if any
STASHED=0
if [ "$HAS_CHANGES" != "0" ]; then
    echo "📦 Stashing local changes..."
    git stash push -q -m "benchmark-temp"
    STASHED=1
fi

cleanup() {
    local exit_code=$?
    
    # Always try to restore to original ref
    echo ""
    echo "🔄 Restoring to $RESTORE_REF..."
    git checkout -q "$RESTORE_REF" 2>/dev/null || {
        echo "⚠️  Failed to checkout $RESTORE_REF, trying commit ref $CURRENT_REF..."
        git checkout -q "$CURRENT_REF" || true
    }
    
    # Restore stashed changes
    if [ "$STASHED" = "1" ]; then
        echo "📦 Restoring stashed changes..."
        git stash pop -q || {
            echo "❌ Failed to restore stash! Your changes are in 'git stash list'"
            echo "   Run 'git stash pop' manually to recover them."
        }
    fi
    
    exit $exit_code
}
trap cleanup EXIT

# Build and benchmark baseline
echo ""
echo "═══════════════════════════════════════"
echo "📊 BASELINE ($BASELINE_COMMIT)"
echo "═══════════════════════════════════════"
git checkout -q "$BASELINE_COMMIT"
cargo build --release -q 2>/dev/null

BASELINE_UNIT=$(run_benchmark "Unit tests" "cargo test --lib -- --test-threads=2")
BASELINE_FULL=$(run_benchmark "Full test suite" "cargo test --all-targets --all-features -- --test-threads=2")

if [ -n "$PARQUET_FILE" ] && [ -f "$PARQUET_FILE" ]; then
    echo "⏱️  Running: Parquet analysis ($PARQUET_FILE)"
    echo "   (Large file benchmark - may take several minutes)"
    # TODO: Add actual parquet analysis command when available
    BASELINE_PARQUET="N/A"
else
    BASELINE_PARQUET="N/A"
fi

# Return to current branch/commit (cleanup trap handles errors)
git checkout -q "$RESTORE_REF"

# Restore stash to test current changes
if [ "$STASHED" = "1" ]; then
    git stash pop -q
    STASHED=0  # Mark as unstashed so cleanup doesn't try again
fi

# Build and benchmark current version
echo ""
echo "═══════════════════════════════════════"
echo "📊 CURRENT (${CURRENT_BRANCH:-$CURRENT_REF})"
echo "═══════════════════════════════════════"
cargo build --release -q 2>/dev/null

CURRENT_UNIT=$(run_benchmark "Unit tests" "cargo test --lib -- --test-threads=2")
CURRENT_FULL=$(run_benchmark "Full test suite" "cargo test --all-targets --all-features -- --test-threads=2")

if [ -n "$PARQUET_FILE" ] && [ -f "$PARQUET_FILE" ]; then
    echo "⏱️  Running: Parquet analysis ($PARQUET_FILE)"
    CURRENT_PARQUET="N/A"
else
    CURRENT_PARQUET="N/A"
fi

# Calculate improvements
calc_improvement() {
    local baseline="$1"
    local current="$2"
    if [ "$baseline" = "N/A" ] || [ "$current" = "N/A" ]; then
        echo "N/A"
    else
        echo "scale=1; ($baseline - $current) / $baseline * 100" | bc
    fi
}

UNIT_IMPROVEMENT=$(calc_improvement "$BASELINE_UNIT" "$CURRENT_UNIT")
FULL_IMPROVEMENT=$(calc_improvement "$BASELINE_FULL" "$CURRENT_FULL")

# Print summary
echo ""
echo "═══════════════════════════════════════"
echo "📈 PERFORMANCE SUMMARY"
echo "═══════════════════════════════════════"
printf "%-25s %12s %12s %12s\n" "Test" "Baseline" "Current" "Improvement"
printf "%-25s %12s %12s %12s\n" "-------------------------" "------------" "------------" "------------"
printf "%-25s %11.2fs %11.2fs %11s%%\n" "Unit tests" "$BASELINE_UNIT" "$CURRENT_UNIT" "$UNIT_IMPROVEMENT"
printf "%-25s %11.2fs %11.2fs %11s%%\n" "Full test suite" "$BASELINE_FULL" "$CURRENT_FULL" "$FULL_IMPROVEMENT"

if [ -n "$PARQUET_FILE" ]; then
    printf "%-25s %12s %12s %12s\n" "Parquet analysis" "$BASELINE_PARQUET" "$CURRENT_PARQUET" "N/A"
fi

echo ""
echo "✅ Benchmark complete"
