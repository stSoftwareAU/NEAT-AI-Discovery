#!/bin/bash
set -euo pipefail

# Benchmark CI script — verifies benchmarks compile and optionally detects regressions.
#
# Usage:
#   ./scripts/benchmark-ci.sh                    # Verify benchmarks compile (default)
#   ./scripts/benchmark-ci.sh --compile-only     # Only check compilation
#   ./scripts/benchmark-ci.sh --compare           # Compare against baseline (requires saved baseline)
#   ./scripts/benchmark-ci.sh --threshold 10      # Set regression threshold to 10% (default: 10)
#   ./scripts/benchmark-ci.sh --list               # List discovered benchmark suites
#   ./scripts/benchmark-ci.sh --help               # Show this help
#
# This script is intended to be called from CI (e.g., a GitHub Actions benchmark
# job) or locally to verify benchmarks before merge.
#
# Modes:
#   compile-only (default): Runs `cargo bench --no-run` to verify all benchmarks
#     compile without actually executing them. Fast and suitable for every PR.
#
#   compare: Runs benchmarks and compares against a saved baseline using the
#     existing benchmark_compare.sh script. Requires a GPU and a previously
#     saved baseline. Suitable for self-hosted runners with consistent hardware.
#
# Environment variables:
#   BENCHMARK_THRESHOLD   Regression threshold percentage (default: 10)
#   BENCHMARK_CI_MODE     Override mode: "compile" or "compare" (default: "compile")

# ── Configuration ─────────────────────────────────────────────────────

THRESHOLD="${BENCHMARK_THRESHOLD:-10}"
MODE="${BENCHMARK_CI_MODE:-compile}"
LIST_ONLY=false
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Argument parsing ──────────────────────────────────────────────────

print_usage() {
    echo "Usage: ./scripts/benchmark-ci.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --compile-only    Only verify benchmarks compile (default)"
    echo "  --compare         Run benchmarks and compare against baseline"
    echo "  --threshold N     Regression threshold percentage (default: 10)"
    echo "  --list            List all discovered benchmark suites"
    echo "  --help, -h        Show this help"
    echo ""
    echo "Environment variables:"
    echo "  BENCHMARK_THRESHOLD   Regression threshold percentage"
    echo "  BENCHMARK_CI_MODE     Override mode: 'compile' or 'compare'"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --compile-only)
            MODE="compile"
            shift
            ;;
        --compare)
            MODE="compare"
            shift
            ;;
        --threshold)
            if [[ $# -lt 2 ]]; then
                echo "Error: --threshold requires a value"
                exit 1
            fi
            THRESHOLD="$2"
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
            echo "Error: Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# ── Validate threshold ────────────────────────────────────────────────

if ! [[ "$THRESHOLD" =~ ^[0-9]+$ ]]; then
    echo "Error: Threshold must be a positive integer, got '$THRESHOLD'"
    exit 1
fi

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
    done < "$PROJECT_ROOT/Cargo.toml"
    if [[ ${#benchmarks[@]} -gt 0 ]]; then
        printf '%s\n' "${benchmarks[@]}"
    fi
}

BENCHMARKS=()
while IFS= read -r line; do
    BENCHMARKS+=("$line")
done < <(discover_benchmarks)

if [[ ${#BENCHMARKS[@]} -eq 0 ]]; then
    echo "Error: No benchmark targets found in Cargo.toml"
    exit 1
fi

if [[ "$LIST_ONLY" == true ]]; then
    echo "Discovered benchmark suites (${#BENCHMARKS[@]}):"
    for bench in "${BENCHMARKS[@]}"; do
        echo "  - $bench"
    done
    exit 0
fi

echo "Benchmark CI — Mode: $MODE"
echo "=============================="
echo "Benchmark suites: ${#BENCHMARKS[@]}"
echo ""

# ── Compile-only mode ────────────────────────────────────────────────

if [[ "$MODE" == "compile" ]]; then
    echo "Verifying all benchmarks compile..."
    echo ""

    COMPILED=0
    FAILED=0
    FAILED_NAMES=""

    for bench in "${BENCHMARKS[@]}"; do
        echo -n "  Compiling: $bench ... "
        if cargo bench --bench "$bench" --no-run 2>&1; then
            echo "OK"
            COMPILED=$((COMPILED + 1))
        else
            echo "FAILED"
            FAILED=$((FAILED + 1))
            FAILED_NAMES="${FAILED_NAMES}  - ${bench}\n"
        fi
    done

    echo ""
    echo "=============================="
    echo "Compilation Summary"
    echo "=============================="
    echo "Compiled: $COMPILED / ${#BENCHMARKS[@]}"
    echo "Failed:   $FAILED"

    if [[ $FAILED -gt 0 ]]; then
        echo ""
        echo "Failed benchmarks:"
        echo -e "$FAILED_NAMES"
        exit 1
    fi

    echo ""
    echo "All benchmarks compile successfully."
    exit 0
fi

# ── Compare mode ─────────────────────────────────────────────────────

if [[ "$MODE" == "compare" ]]; then
    COMPARE_SCRIPT="$PROJECT_ROOT/benchmark_compare.sh"

    if [[ ! -x "$COMPARE_SCRIPT" ]]; then
        echo "Error: benchmark_compare.sh not found or not executable at $COMPARE_SCRIPT"
        exit 1
    fi

    # Check if baseline exists
    CRITERION_DIR="$PROJECT_ROOT/target/criterion"
    if [[ ! -d "$CRITERION_DIR" ]]; then
        echo "No baseline found. Saving initial baseline..."
        echo ""
        "$COMPARE_SCRIPT" --save-baseline
        echo ""
        echo "Baseline saved. No comparison possible on first run."
        exit 0
    fi

    echo "Comparing against baseline with ${THRESHOLD}% threshold..."
    echo ""

    # Delegate to the existing comparison script
    "$COMPARE_SCRIPT" --threshold "$THRESHOLD"
    exit $?
fi

echo "Error: Unknown mode '$MODE'. Use 'compile' or 'compare'."
exit 1
