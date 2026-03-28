#!/bin/bash
# =============================================================================
# Auto-increment Cargo.toml patch version when src/ has changed.
#
# Called by quality.sh so that the version is always incremented BEFORE
# committing. This prevents the CI version-increment job from pushing a
# new commit with GITHUB_TOKEN (which does not re-trigger workflows),
# eliminating the need for manual follow-up pushes.
#
# The script is idempotent — it only increments once per branch by
# comparing the local version against the base branch.
#
# Issue #955
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve the repository root (where Cargo.toml lives)
# ---------------------------------------------------------------------------
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || echo "")"
if [ -z "$REPO_ROOT" ]; then
    echo "⏭️  Not a git repository — skipping version auto-increment"
    exit 0
fi

CARGO_TOML="$REPO_ROOT/Cargo.toml"
if [ ! -f "$CARGO_TOML" ]; then
    echo "⏭️  No Cargo.toml found — skipping version auto-increment"
    exit 0
fi

# ---------------------------------------------------------------------------
# Determine the base branch (Develop or develop)
# ---------------------------------------------------------------------------
BASE_BRANCH=""
for candidate in Develop develop main; do
    if git show-ref --verify --quiet "refs/remotes/origin/$candidate" 2>/dev/null; then
        BASE_BRANCH="$candidate"
        break
    fi
done

if [ -z "$BASE_BRANCH" ]; then
    echo "⏭️  No base branch found — skipping version auto-increment"
    exit 0
fi

# ---------------------------------------------------------------------------
# Skip if we ARE on the base branch (no increment needed for direct pushes)
# ---------------------------------------------------------------------------
CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")"
if [ "$CURRENT_BRANCH" = "$BASE_BRANCH" ]; then
    echo "⏭️  On base branch ($BASE_BRANCH) — skipping version auto-increment"
    exit 0
fi

# ---------------------------------------------------------------------------
# Check whether src/ has changed compared to the base branch
# ---------------------------------------------------------------------------
if git diff --quiet "origin/$BASE_BRANCH"...HEAD -- src/ 2>/dev/null; then
    # Also check for staged but uncommitted src/ changes
    if git diff --quiet --cached -- src/ 2>/dev/null; then
        # Also check for unstaged src/ changes
        if git diff --quiet -- src/ 2>/dev/null; then
            echo "⏭️  No src/ changes — skipping version auto-increment"
            exit 0
        fi
    fi
fi

# ---------------------------------------------------------------------------
# Read the current version and the base-branch version
# ---------------------------------------------------------------------------
CURRENT_VERSION="$(grep '^version = ' "$CARGO_TOML" | sed 's/version = "\(.*\)"/\1/')"
BASE_VERSION="$(git show "origin/$BASE_BRANCH:Cargo.toml" 2>/dev/null \
    | grep '^version = ' \
    | sed 's/version = "\(.*\)"/\1/' || echo "")"

if [ -z "$CURRENT_VERSION" ]; then
    echo "⚠️  Could not read version from Cargo.toml — skipping"
    exit 0
fi

# ---------------------------------------------------------------------------
# Skip if the version has already been incremented for this branch
# ---------------------------------------------------------------------------
if [ -n "$BASE_VERSION" ] && [ "$CURRENT_VERSION" != "$BASE_VERSION" ]; then
    echo "✅ Version already incremented ($BASE_VERSION → $CURRENT_VERSION)"
    exit 0
fi

# ---------------------------------------------------------------------------
# Increment the patch version
# ---------------------------------------------------------------------------
IFS='.' read -r major minor patch <<< "$CURRENT_VERSION"
major=${major:-0}
minor=${minor:-0}
patch=${patch:-0}
NEW_VERSION="$major.$minor.$((patch + 1))"

# Use a portable sed invocation (works on macOS and Linux)
if sed --version >/dev/null 2>&1; then
    # GNU sed
    sed -i "s/^version = \"$CURRENT_VERSION\"/version = \"$NEW_VERSION\"/" "$CARGO_TOML"
else
    # BSD sed (macOS)
    sed -i '' "s/^version = \"$CURRENT_VERSION\"/version = \"$NEW_VERSION\"/" "$CARGO_TOML"
fi

echo "📦 Version auto-incremented: $CURRENT_VERSION → $NEW_VERSION"
