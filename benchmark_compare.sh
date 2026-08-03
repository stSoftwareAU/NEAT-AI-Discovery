#!/bin/bash
# Benchmark Regression Tracking with Criterion Comparison
#
# Compares current benchmark results against a saved baseline using
# Criterion's built-in comparison features.
#
# Usage:
#   ./benchmark_compare.sh                          # Compare all benchmarks against baseline
#   ./benchmark_compare.sh --save-baseline          # Save current results as baseline
#   ./benchmark_compare.sh --threshold 10           # Set regression threshold to 10%
#   ./benchmark_compare.sh --bench synapse_counts   # Compare a single benchmark suite
#   ./benchmark_compare.sh --list                   # List available benchmark suites
#   ./benchmark_compare.sh --help                   # Show this help
#
# Environment variables:
#   BENCHMARK_THRESHOLD  Regression threshold percentage (default: 5). Must be a
#                        non-negative number such as 5 or 2.5 — anything else
#                        (including a bc expression like 10^9) is rejected.
#   BENCHMARK_BASELINE   Baseline name (default: "saved")
#
# The baseline is stored in target/criterion/ and is machine-specific.
# Each developer/CI runner must save their own baseline.
#
# Workflow:
#   1. Save a baseline:    ./benchmark_compare.sh --save-baseline
#   2. Make code changes
#   3. Compare:            ./benchmark_compare.sh
#   4. After intentional performance changes: re-save the baseline

set -euo pipefail

# ── Shared helpers ────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THRESHOLD_LIB="$SCRIPT_DIR/scripts/benchmark_threshold.sh"
if [[ ! -r "$THRESHOLD_LIB" ]]; then
    echo "Error: required helper not found: $THRESHOLD_LIB" >&2
    exit 1
fi
# shellcheck source=scripts/benchmark_threshold.sh
# shellcheck disable=SC1091
source "$THRESHOLD_LIB"

# ── Configuration ─────────────────────────────────────────────────────

THRESHOLD="${BENCHMARK_THRESHOLD:-5}"
BASELINE_NAME="${BENCHMARK_BASELINE:-saved}"
SAVE_BASELINE=false
SINGLE_BENCH=""
LIST_ONLY=false

# ── Argument parsing ──────────────────────────────────────────────────

print_usage() {
    # Extract usage from header comments (POSIX-compatible sed)
    sed -n '2,/^$/p' "$0" | sed 's/^# //' | sed 's/^#$//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --save-baseline)
            SAVE_BASELINE=true
            shift
            ;;
        --threshold)
            if [[ $# -lt 2 ]]; then
                echo "Error: --threshold requires a value" >&2
                exit 1
            fi
            THRESHOLD="$2"
            shift 2
            ;;
        --bench)
            SINGLE_BENCH="$2"
            shift 2
            ;;
        --list)
            LIST_ONLY=true
            shift
            ;;
        --help|-h)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# ── Validate threshold ────────────────────────────────────────────────

# Runs after argument parsing so `--help` still works, and before any
# comparison so a bogus value can never reach `bc` (Issue #1918).
benchmark_threshold::require_valid "$THRESHOLD"

# ── Discover benchmark suites from Cargo.toml ────────────────────────

discover_benchmarks() {
    local benchmarks=()
    local in_bench_block=false
    while IFS= read -r line; do
        if [[ "$line" =~ ^\[\[bench\]\] ]]; then
            in_bench_block=true
        elif [[ "$in_bench_block" == true ]] && [[ "$line" =~ ^name[[:space:]]*=[[:space:]]*\"([^\"]+)\" ]]; then
            benchmarks+=("${BASH_REMATCH[1]}")
            in_bench_block=false
        elif [[ "$line" =~ ^\[ ]]; then
            in_bench_block=false
        fi
    done < Cargo.toml
    if [[ ${#benchmarks[@]} -gt 0 ]]; then
        printf '%s\n' "${benchmarks[@]}"
    fi
}

BENCHMARKS=()
while IFS= read -r line; do
    BENCHMARKS+=("$line")
done < <(discover_benchmarks)

if [[ "$LIST_ONLY" == true ]]; then
    echo "Available benchmark suites (${#BENCHMARKS[@]}):"
    for bench in "${BENCHMARKS[@]}"; do
        echo "  - $bench"
    done
    exit 0
fi

# Filter to single benchmark if requested
if [[ -n "$SINGLE_BENCH" ]]; then
    found=false
    for bench in "${BENCHMARKS[@]}"; do
        if [[ "$bench" == "$SINGLE_BENCH" ]]; then
            found=true
            break
        fi
    done
    if [[ "$found" != true ]]; then
        echo "Error: Unknown benchmark '$SINGLE_BENCH'"
        echo "Available benchmarks: ${BENCHMARKS[*]}"
        exit 1
    fi
    BENCHMARKS=("$SINGLE_BENCH")
fi

# ── Save baseline mode ───────────────────────────────────────────────

if [[ "$SAVE_BASELINE" == true ]]; then
    echo "Saving benchmark baseline as '$BASELINE_NAME'"
    echo "========================================="
    echo ""
    echo "This will run all ${#BENCHMARKS[@]} benchmark suites and save results."
    echo "Benchmarks that require a GPU will be skipped if no GPU is available."
    echo ""

    for bench in "${BENCHMARKS[@]}"; do
        echo "Running: $bench"
        cargo bench --bench "$bench" -- --save-baseline "$BASELINE_NAME" 2>&1 || {
            echo "  Warning: $bench failed (may require GPU) — skipping"
        }
        echo ""
    done

    echo "Baseline '$BASELINE_NAME' saved to target/criterion/"
    echo ""
    echo "To compare against this baseline after making changes:"
    echo "  ./benchmark_compare.sh"
    exit 0
fi

# ── Comparison mode ──────────────────────────────────────────────────

echo "Benchmark Regression Comparison"
echo "==============================="
echo "Baseline:  $BASELINE_NAME"
echo "Threshold: ${THRESHOLD}%"
echo "Suites:    ${#BENCHMARKS[@]}"
echo ""

# Check baseline exists
CRITERION_DIR="target/criterion"
if [[ ! -d "$CRITERION_DIR" ]]; then
    echo "Error: No baseline found at $CRITERION_DIR/"
    echo ""
    echo "Save a baseline first:"
    echo "  ./benchmark_compare.sh --save-baseline"
    exit 1
fi

# Track overall results
TOTAL_BENCHMARKS=0
REGRESSIONS=0
IMPROVEMENTS=0
UNCHANGED=0
SKIPPED=0
REGRESSION_DETAILS=""

# Run each benchmark suite and compare against baseline
for bench in "${BENCHMARKS[@]}"; do
    echo "Comparing: $bench"

    # Run benchmark with baseline comparison, capturing output
    OUTPUT=$(cargo bench --bench "$bench" -- --baseline "$BASELINE_NAME" 2>&1) || {
        echo "  Skipped (may require GPU or baseline missing)"
        SKIPPED=$((SKIPPED + 1))
        echo ""
        continue
    }

    # Parse Criterion output for performance changes
    # Criterion outputs lines like:
    #   <benchmark_name>  time:   [1.2345 µs 1.2500 µs 1.2678 µs]
    #                     change: [-2.3456% -1.2345% +0.1234%] (p = 0.12 < 0.05)
    #                     Performance has regressed.
    while IFS= read -r line; do
        if [[ "$line" =~ change:[[:space:]]*\[.*[[:space:]]([+-]?[0-9]+\.[0-9]+)% ]]; then
            change="${BASH_REMATCH[1]}"
            TOTAL_BENCHMARKS=$((TOTAL_BENCHMARKS + 1))

            # Remove leading + for comparison
            change_num="${change#+}"

            # A malformed measurement would make `bc` error and leave `(( ))`
            # with an empty operand, so fail loud rather than miscompare.
            if ! benchmark_threshold::is_measurement "$change_num"; then
                echo "Error: unparsable benchmark change value '$change_num' in suite '$bench'" >&2
                exit 1
            fi

            # Check if it's a regression (positive change means slower)
            if (( $(echo "$change_num > $THRESHOLD" | bc -l) )); then
                REGRESSIONS=$((REGRESSIONS + 1))
                # Find the benchmark name from the previous lines
                bench_name=$(echo "$OUTPUT" | grep -B5 "$line" | grep "time:" | tail -1 | sed 's/[[:space:]]*time:.*//' | xargs)
                REGRESSION_DETAILS="${REGRESSION_DETAILS}  REGRESSION: ${bench_name:-$bench} changed by ${change}% (threshold: ${THRESHOLD}%)\n"
                echo "  REGRESSION: ${bench_name:-$bench} (${change}%)"
            elif (( $(echo "$change_num < -$THRESHOLD" | bc -l) )); then
                IMPROVEMENTS=$((IMPROVEMENTS + 1))
                echo "  Improved (${change}%)"
            else
                UNCHANGED=$((UNCHANGED + 1))
            fi
        fi
    done <<< "$OUTPUT"

    echo ""
done

# ── Summary ──────────────────────────────────────────────────────────

echo "==============================="
echo "Summary"
echo "==============================="
echo "Total comparisons: $TOTAL_BENCHMARKS"
echo "Regressions:       $REGRESSIONS (exceeding ${THRESHOLD}% threshold)"
echo "Improvements:      $IMPROVEMENTS"
echo "Within threshold:  $UNCHANGED"
echo "Skipped suites:    $SKIPPED"
echo ""

if [[ $REGRESSIONS -gt 0 ]]; then
    echo "REGRESSIONS DETECTED:"
    echo -e "$REGRESSION_DETAILS"
    echo ""
    echo "If these regressions are intentional, update the baseline:"
    echo "  ./benchmark_compare.sh --save-baseline"
    exit 1
fi

if [[ $TOTAL_BENCHMARKS -eq 0 ]] && [[ $SKIPPED -eq ${#BENCHMARKS[@]} ]]; then
    echo "Warning: All benchmark suites were skipped."
    echo "Ensure a GPU is available and baseline is saved."
    exit 0
fi

echo "No regressions detected."
exit 0
