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

set -euo pipefail

PARQUET_FILE="${1:-}"
# Baseline: v0.1.146 (11-Dec-2025) - before GPU processing optimisations
# Recent GPU changes: v0.1.151-153 (12-Dec) introduced GPU timeouts, memory adaptation
BASELINE_COMMIT="bed746a"  # v0.1.146 - last stable before GPU optimisation work

echo "🏁 GPU Performance Benchmark"
echo "=============================="
echo "Date: $(date '+%d-%b-%Y %H:%M:%S')"
echo ""

# Function to run benchmark and capture timing
run_benchmark() {
    local label="$1"
    local cmd="$2"
    local start end duration
    
    echo "⏱️  Running: $label"
    start=$(date +%s.%N)
    eval "$cmd" > /dev/null 2>&1 || true
    end=$(date +%s.%N)
    duration=$(echo "$end - $start" | bc)
    echo "   Duration: ${duration}s"
    echo "$duration"
}

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

BASELINE_UNIT=$(run_benchmark "Unit tests" "cargo test --lib -- --test-threads=1")
BASELINE_FULL=$(run_benchmark "Full test suite" "cargo test --all-targets --all-features -- --test-threads=1")

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

CURRENT_UNIT=$(run_benchmark "Unit tests" "cargo test --lib -- --test-threads=1")
CURRENT_FULL=$(run_benchmark "Full test suite" "cargo test --all-targets --all-features -- --test-threads=1")

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
