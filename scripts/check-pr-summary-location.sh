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
stray_files=()
while IFS= read -r file; do
    stray_files+=("$file")
done < <(find docs -type f -name 'pr-summary-*.md' -not -path "./${CANONICAL_DIR}/*" -not -path "${CANONICAL_DIR}/*" 2>/dev/null | sort)

if [[ ${#stray_files[@]} -gt 0 ]]; then
    echo "❌ Found ${#stray_files[@]} pr-summary-*.md file(s) outside ${CANONICAL_DIR}/:"
    printf '   %s\n' "${stray_files[@]}"
    echo ""
    echo "Move them into ${CANONICAL_DIR}/ with 'git mv' (never delete — the"
    echo "learnings must be preserved). See docs/archive/pr-summaries/README.md."
    exit 1
fi

echo "✅ All pr-summary-*.md files live in ${CANONICAL_DIR}/"
