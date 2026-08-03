#!/bin/bash
# Shared threshold validation for the benchmark regression gate (Issue #1918).
#
# Sourced by `benchmark_compare.sh` and `scripts/benchmark-ci.sh`; not executed
# directly. Both scripts interpolate the threshold into a `bc` expression, so an
# unvalidated value defeats the gate: `10^9` is valid `bc` and puts every
# regression under threshold, while `abc` makes `bc` error and leaves the
# comparison operating on an empty operand. Keeping the check here stops the two
# scripts drifting apart again.

# Returns 0 when $1 is a plain non-negative decimal — no sign, exponent, or `bc`
# expression. This is the only shape a threshold may take.
benchmark_threshold::is_valid() {
    [[ "${1:-}" =~ ^[0-9]+(\.[0-9]+)?$ ]]
}

# Returns 0 when $1 is a signed decimal, the shape Criterion reports percentage
# changes in. Used to guard measured values before they reach `bc`.
benchmark_threshold::is_measurement() {
    [[ "${1:-}" =~ ^-?[0-9]+(\.[0-9]+)?$ ]]
}

# Exits 1 with a clear message unless $1 is a valid threshold.
benchmark_threshold::require_valid() {
    if ! benchmark_threshold::is_valid "${1:-}"; then
        echo "Error: Threshold must be a positive integer or decimal, got '${1:-}'" >&2
        exit 1
    fi
}
