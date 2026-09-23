#!/bin/bash
# Guard: every pr-summary-*.md must live in docs/archive/pr-summaries/ (Issue #1613).
#
# The archive is the canonical, documented home for PR summaries
# (docs/archive/pr-summaries/README.md, CONTRIBUTING.md). A summary left loose
# in docs/ (or anywhere else) contradicts that convention and splits the
# learnings across two locations. This check fails loud if any stray file is
# found so the layout cannot silently regress.
set -euo pipefail

# Resolve repo root so the check works from any working directory.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CANONICAL_DIR="docs/archive/pr-summaries"

# Find every pr-summary-*.md tracked in the tree, excluding the canonical dir.
# Use find over a git-tracked list so the guard also catches unstaged strays.
#
# The scan lands in a temporary file rather than a process substitution: `set -e`
# never checks a process substitution's exit status, and `sort` succeeds on empty
# input, so a find that could not scan (missing docs/, unreadable subtree) used
# to be indistinguishable from one that found nothing — and the guard reported ✅
# for a check that never ran (Issue #2139). find's own diagnostics stay on stderr
# so the operator sees why the scan failed.
scan_results="$(mktemp)"
trap 'rm -f "$scan_results"' EXIT

if ! find docs -type f -name 'pr-summary-*.md' -not -path "./${CANONICAL_DIR}/*" -not -path "${CANONICAL_DIR}/*" -print0 >"$scan_results"; then
    echo "❌ find could not scan docs/ — the PR summary layout was NOT checked." >&2
    echo "   Fix the error reported by find above, then re-run this guard." >&2
    exit 1
fi
if ! sort -z "$scan_results" -o "$scan_results"; then
    echo "❌ sort could not order the find results — the PR summary layout was NOT checked." >&2
    exit 1
fi

# NUL-delimited, so paths containing spaces (or newlines) survive intact.
stray_files=()
while IFS= read -r -d '' file; do
    stray_files+=("$file")
done <"$scan_results"

if [[ ${#stray_files[@]} -gt 0 ]]; then
    echo "❌ Found ${#stray_files[@]} pr-summary-*.md file(s) outside ${CANONICAL_DIR}/:"
    printf '   %s\n' "${stray_files[@]}"
    echo ""
    echo "Move them into ${CANONICAL_DIR}/ with 'git mv'."
    echo "Retention rule: never delete an unfolded summary — fold its durable"
    echo "learnings into the live docs first, then delete it."
    echo "See docs/archive/pr-summaries/README.md (Issue #1682)."
    exit 1
fi

echo "✅ All pr-summary-*.md files live in ${CANONICAL_DIR}/"
