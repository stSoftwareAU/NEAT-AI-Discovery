#!/usr/bin/env bash
#
# ShellCheck lint gate (Issue #1898).
#
# Runs ShellCheck (upstream: koalaman/shellcheck) over every shell script under
# the given roots. This script is the repository's own committed gate — the
# single source of truth shared by `./quality.sh` (local gate) and
# `.github/workflows/shellcheck.yml` (pull-request gate), so both enforce
# exactly the same rules. It replaces the unmaintained `ludeeus/action-shellcheck`
# wrapper action; CI installs the koalaman binary directly.
#
# Usage: quality/shellcheck.sh [root ...]      (default root: the current directory)
#
# Exits non-zero — loudly — when any script fails ShellCheck, when ShellCheck is
# not installed, when a root does not exist, or when no scripts were found at
# all (a gate that scans nothing must not report success).

set -euo pipefail

roots=("$@")
if [[ ${#roots[@]} -eq 0 ]]; then
    roots=(".")
fi

if ! command -v shellcheck > /dev/null 2>&1; then
    echo "shellcheck: not installed — install it: https://github.com/koalaman/shellcheck#installing" >&2
    exit 1
fi

for root in "${roots[@]}"; do
    if [[ ! -d "$root" ]]; then
        echo "shellcheck: scan root does not exist: $root" >&2
        exit 1
    fi
done

scanned=0
failed=0

# Process substitution keeps the loop in this shell, so the counters survive
# (a `find | while` pipeline would increment them inside a subshell).
# NUL-delimited so paths containing spaces are handled correctly.
while IFS= read -r -d '' script; do
    scanned=$((scanned + 1))
    # Every script is linted; a failure does not stop the scan, so one run
    # reports every offending file.
    if ! shellcheck -s bash "$script"; then
        echo "shellcheck: FAILED $script" >&2
        failed=$((failed + 1))
    fi
done < <(find "${roots[@]}" \
    \( -path '*/target' -o -path '*/.git' -o -path '*/node_modules' \) -prune -o \
    -name '*.sh' -type f -print0)

if [[ "$scanned" -eq 0 ]]; then
    echo "shellcheck: no shell scripts found under: ${roots[*]}" >&2
    echo "shellcheck: the gate scanned nothing — refusing to report success" >&2
    exit 1
fi

if [[ "$failed" -ne 0 ]]; then
    echo "shellcheck: $failed of $scanned script(s) failed ShellCheck" >&2
    exit 1
fi

echo "shellcheck: OK — $scanned script(s) passed ShellCheck"
