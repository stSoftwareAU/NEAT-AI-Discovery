#!/usr/bin/env bash
#
# Bash syntax gate (Issue #1755).
#
# Runs `bash -n` over every shell script under the given roots so a syntax
# error fails the build instead of landing on the default branch — bash has no
# compile step, so nothing else catches it. This script is the repository's own
# committed gate: `./quality.sh` runs it locally and
# `.github/workflows/shellcheck.yml` runs it on every pull request.
#
# Usage: quality/bash_syntax.sh [root ...]     (default root: the current directory)
#
# Exits non-zero — loudly — when any script fails `bash -n`, when a root does
# not exist, or when no scripts were found at all (a gate that scans nothing
# must not report success).

set -euo pipefail

roots=("$@")
if [[ ${#roots[@]} -eq 0 ]]; then
    roots=(".")
fi

for root in "${roots[@]}"; do
    if [[ ! -d "$root" ]]; then
        echo "bash-syntax: scan root does not exist: $root" >&2
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
    if ! bash -n "$script"; then
        echo "bash-syntax: FAILED $script" >&2
        failed=$((failed + 1))
    fi
done < <(find "${roots[@]}" \
    \( -path '*/target' -o -path '*/.git' -o -path '*/node_modules' \) -prune -o \
    -name '*.sh' -type f -print0)

if [[ "$scanned" -eq 0 ]]; then
    echo "bash-syntax: no shell scripts found under: ${roots[*]}" >&2
    echo "bash-syntax: the gate scanned nothing — refusing to report success" >&2
    exit 1
fi

if [[ "$failed" -ne 0 ]]; then
    echo "bash-syntax: $failed of $scanned script(s) failed 'bash -n'" >&2
    exit 1
fi

echo "bash-syntax: OK — $scanned script(s) passed 'bash -n'"
