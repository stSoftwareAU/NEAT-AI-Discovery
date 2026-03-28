#!/bin/bash
# =============================================================================
# Integration test for scripts/auto-version.sh
#
# Creates a temporary git repository, simulates the branch/change patterns
# that occur during normal PR development, and verifies that auto-version.sh
# increments the version correctly (and skips when it should).
#
# Exit code: 0 on success, 1 on any assertion failure.
# Issue #955
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
AUTO_VERSION="$REPO_ROOT/scripts/auto-version.sh"

PASS=0
FAIL=0
TMPDIR_BASE=""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
cleanup() {
    if [ -n "$TMPDIR_BASE" ] && [ -d "$TMPDIR_BASE" ]; then
        rm -rf "$TMPDIR_BASE"
    fi
}
trap cleanup EXIT

assert_version() {
    local cargo_toml="$1"
    local expected="$2"
    local label="$3"
    local actual
    actual="$(grep '^version = ' "$cargo_toml" | sed 's/version = "\(.*\)"/\1/')"
    if [ "$actual" = "$expected" ]; then
        echo "  ✅ PASS: $label (version=$actual)"
        PASS=$((PASS + 1))
    else
        echo "  ❌ FAIL: $label — expected $expected, got $actual"
        FAIL=$((FAIL + 1))
    fi
}

# Create a fresh temporary git repo with a Cargo.toml and a src/ directory.
# Sets up an "origin/Develop" base branch and a feature branch.
setup_repo() {
    local version="${1:-1.0.0}"
    local dir
    dir="$(mktemp -d)"
    TMPDIR_BASE="$dir"

    cd "$dir"
    git init -b Develop --quiet
    git config user.email "test@test.com"
    git config user.name "Test"

    mkdir -p src
    cat > Cargo.toml <<EOF
[package]
name = "test_crate"
version = "$version"
edition = "2024"
EOF
    echo "// placeholder" > src/lib.rs

    git add -A
    git commit -m "initial" --quiet

    # Create a bare remote so we have origin/Develop
    local bare
    bare="$(mktemp -d)"
    git clone --bare "$dir" "$bare" --quiet 2>/dev/null
    git remote remove origin 2>/dev/null || true
    git remote add origin "$bare"
    git fetch origin --quiet 2>/dev/null

    # Create a feature branch
    git checkout -b feature/test --quiet
}

# ---------------------------------------------------------------------------
# Test 1: Increment when branch has changes and version matches base
# ---------------------------------------------------------------------------
echo "Test 1: Increment when branch has changes"
setup_repo "1.0.0"
echo "// new code" >> src/lib.rs
git add src/lib.rs
git commit -m "add code" --quiet
bash "$AUTO_VERSION"
assert_version "$TMPDIR_BASE/Cargo.toml" "1.0.1" "Version incremented from 1.0.0 to 1.0.1"
cleanup
TMPDIR_BASE=""

# ---------------------------------------------------------------------------
# Test 2: Skip when version already differs from base
# ---------------------------------------------------------------------------
echo "Test 2: Skip when version already incremented"
setup_repo "1.0.0"
# Manually change version before running script
sed -i.bak 's/version = "1.0.0"/version = "1.0.5"/' Cargo.toml && rm -f Cargo.toml.bak
echo "// new code" >> src/lib.rs
git add -A
git commit -m "manual version bump" --quiet
bash "$AUTO_VERSION"
assert_version "$TMPDIR_BASE/Cargo.toml" "1.0.5" "Version unchanged (already differs)"
cleanup
TMPDIR_BASE=""

# ---------------------------------------------------------------------------
# Test 3: Increment for non-src changes (e.g. docs, scripts)
# ---------------------------------------------------------------------------
echo "Test 3: Increment for non-src changes"
setup_repo "2.0.0"
echo "# docs change" >> README.md
git add -A
git commit -m "docs only" --quiet
bash "$AUTO_VERSION"
assert_version "$TMPDIR_BASE/Cargo.toml" "2.0.1" "Version incremented for non-src changes"
cleanup
TMPDIR_BASE=""

# ---------------------------------------------------------------------------
# Test 4: Skip when on the base branch
# ---------------------------------------------------------------------------
echo "Test 4: Skip when on base branch"
setup_repo "3.0.0"
git checkout Develop --quiet
echo "// new code" >> src/lib.rs
git add src/lib.rs
git commit -m "direct commit" --quiet
bash "$AUTO_VERSION"
assert_version "$TMPDIR_BASE/Cargo.toml" "3.0.0" "Version unchanged (on base branch)"
cleanup
TMPDIR_BASE=""

# ---------------------------------------------------------------------------
# Test 5: Idempotent — running twice does not double-increment
# ---------------------------------------------------------------------------
echo "Test 5: Idempotent (no double-increment)"
setup_repo "1.0.0"
echo "// new code" >> src/lib.rs
git add src/lib.rs
git commit -m "add code" --quiet
bash "$AUTO_VERSION"
assert_version "$TMPDIR_BASE/Cargo.toml" "1.0.1" "First run increments"
# Commit the version change so the second run sees the diff
git add Cargo.toml
git commit -m "version bump" --quiet
bash "$AUTO_VERSION"
assert_version "$TMPDIR_BASE/Cargo.toml" "1.0.1" "Second run is a no-op"
cleanup
TMPDIR_BASE=""

# ---------------------------------------------------------------------------
# Test 6: Handles unstaged changes
# ---------------------------------------------------------------------------
echo "Test 6: Handles unstaged changes"
setup_repo "4.0.0"
echo "// uncommitted change" >> src/lib.rs
bash "$AUTO_VERSION"
assert_version "$TMPDIR_BASE/Cargo.toml" "4.0.1" "Incremented for unstaged changes"
cleanup
TMPDIR_BASE=""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "================================"
echo "Results: $PASS passed, $FAIL failed"
echo "================================"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
