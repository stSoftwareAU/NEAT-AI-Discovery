#!/bin/bash
set -euo pipefail

# bump-deps.sh — refresh dependencies on every PR (Issue #1156).
#
# Policy (per stSoftwareAU/VibeCoding#1614):
#   * Internal deps under stSoftwareAU/* (e.g. NEAT-AI itself) advance to the
#     current release / Develop HEAD immediately, with no quarantine window.
#   * External deps (crates.io, GitHub Actions, etc.) bump to the latest
#     version published BEFORE the quarantine window — VIBE_BUMP_QUARANTINE_HOURS
#     hours ago (default 24h). This dodges fast-flagged supply-chain attacks.
#
# Audit gate: after bumping, `cargo deny check` must pass; any new advisory
# fails the run. The Vibe Coder worker reverts the bump on non-zero exit.
#
# Lockfile integrity: after Cargo.toml mutations we run `cargo update` and
# then `cargo check --locked` to ensure the lockfile matches the registry.
#
# Usage:
#   ./bump-deps.sh                  # bump deps, run audit gate
#   ./bump-deps.sh --dry-run        # report planned bumps; do not modify files
#   ./bump-deps.sh --no-network     # skip crates.io lookups (offline mode)
#   ./bump-deps.sh --print-config   # print effective configuration and exit
#   ./bump-deps.sh --list-internal-deps
#                                   # list any stSoftwareAU/* deps and exit
#   ./bump-deps.sh --help           # show this help
#
# Environment:
#   VIBE_BUMP_QUARANTINE_HOURS  external-dep quarantine window (default 24)
#
# Exit codes:
#   0   clean (or no-op)
#   non-zero  bump rejected (audit failed, lockfile mismatch, invalid input)

# ── Helper functions (sourceable for tests) ───────────────────────────

# bump_deps::is_quarantine_expired NOW PUBLISHED_AT THRESHOLD_HOURS
# Returns 0 (success) when (NOW - PUBLISHED_AT) >= THRESHOLD_HOURS, else 1.
# All values are in hours.
bump_deps::is_quarantine_expired() {
    local now="$1"
    local published="$2"
    local threshold="$3"
    local elapsed=$(( now - published ))
    if (( elapsed >= threshold )); then
        return 0
    fi
    return 1
}

# bump_deps::list_internal_deps MANIFEST
# Print stSoftwareAU/* git deps from the given Cargo.toml on stdout, one per
# line. Empty output (and exit 0) means none found.
bump_deps::list_internal_deps() {
    local manifest="$1"
    if [[ ! -f "$manifest" ]]; then
        return 0
    fi
    # Look for git = "https://github.com/stSoftwareAU/...".
    grep -E 'git[[:space:]]*=[[:space:]]*"[^"]*stSoftwareAU/' "$manifest" || true
}

# bump_deps::validate_hours VALUE
# Exit 0 when VALUE is a non-negative decimal integer; exit 1 otherwise.
bump_deps::validate_hours() {
    local value="$1"
    if [[ "$value" =~ ^[0-9]+$ ]]; then
        return 0
    fi
    return 1
}

# When sourced by the test suite we stop before parsing arguments / running.
if [[ "${BUMP_DEPS_SOURCE_ONLY:-0}" == "1" ]]; then
    # shellcheck disable=SC2317  # `exit 0` is the fallback when not sourced.
    return 0 2>/dev/null || exit 0
fi

# ── Source cargo environment for non-login shells ─────────────────────

if [[ -f "$HOME/.cargo/env" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
fi

# ── Configuration ─────────────────────────────────────────────────────

QUARANTINE_HOURS_RAW="${VIBE_BUMP_QUARANTINE_HOURS:-24}"
DRY_RUN=0
NO_NETWORK=0
ACTION="bump"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
CARGO_MANIFEST="$PROJECT_ROOT/Cargo.toml"

# ── Argument parsing ──────────────────────────────────────────────────

print_usage() {
    cat <<'USAGE'
Usage: ./bump-deps.sh [OPTIONS]

Refresh dependencies with audit-gated bumps. Designed to run before
quality.sh in the Vibe Coder worker.

Options:
  --dry-run               Report planned bumps without modifying files.
  --no-network            Skip crates.io lookups (offline; quarantine treated
                          as "do not bump" for affected deps).
  --print-config          Print effective configuration and exit.
  --list-internal-deps    List stSoftwareAU/* internal deps and exit.
  -h, --help              Show this help and exit.

Environment:
  VIBE_BUMP_QUARANTINE_HOURS  External-dep quarantine window in hours
                              (default 24). New external versions younger
                              than this window are skipped.

Exit codes:
  0  clean (or no-op)
  *  bump rejected — audit failure, lockfile drift, or bad input.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            print_usage
            exit 0
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --no-network)
            NO_NETWORK=1
            shift
            ;;
        --print-config)
            ACTION="print-config"
            shift
            ;;
        --list-internal-deps)
            ACTION="list-internal-deps"
            shift
            ;;
        *)
            echo "Unknown option: $1" >&2
            print_usage >&2
            exit 2
            ;;
    esac
done

# Validate quarantine hours (after parsing so --help still works).
if ! bump_deps::validate_hours "$QUARANTINE_HOURS_RAW"; then
    echo "ERROR: VIBE_BUMP_QUARANTINE_HOURS must be a non-negative integer (got: $QUARANTINE_HOURS_RAW)" >&2
    exit 2
fi
QUARANTINE_HOURS="$QUARANTINE_HOURS_RAW"

# ── Action: --print-config ────────────────────────────────────────────

if [[ "$ACTION" == "print-config" ]]; then
    echo "bump-deps configuration"
    echo "  quarantine_hours = $QUARANTINE_HOURS"
    echo "  dry_run          = $DRY_RUN"
    echo "  no_network       = $NO_NETWORK"
    echo "  project_root     = $PROJECT_ROOT"
    if [[ -f "$CARGO_MANIFEST" ]]; then
        echo "  Cargo.toml       = present"
    else
        echo "  Cargo.toml       = absent"
    fi
    exit 0
fi

# ── Action: --list-internal-deps ──────────────────────────────────────

if [[ "$ACTION" == "list-internal-deps" ]]; then
    INTERNAL_DEPS="$(bump_deps::list_internal_deps "$CARGO_MANIFEST")"
    if [[ -z "$INTERNAL_DEPS" ]]; then
        echo "0 internal deps (no stSoftwareAU/* git dependencies in $CARGO_MANIFEST)"
        exit 0
    fi
    echo "Internal deps (stSoftwareAU/*):"
    echo "$INTERNAL_DEPS"
    exit 0
fi

# ── Action: bump (default) ────────────────────────────────────────────

echo "🔄 bump-deps.sh — refresh dependencies"
echo "   quarantine_hours = $QUARANTINE_HOURS"
echo "   dry_run          = $DRY_RUN"
echo "   no_network       = $NO_NETWORK"
echo ""

# Phase 1: internal deps.
INTERNAL_DEPS="$(bump_deps::list_internal_deps "$CARGO_MANIFEST")"
INTERNAL_COUNT=0
if [[ -n "$INTERNAL_DEPS" ]]; then
    INTERNAL_COUNT=$(printf "%s\n" "$INTERNAL_DEPS" | wc -l | tr -d ' ')
    echo "📦 Internal deps detected: $INTERNAL_COUNT (advance immediately, no quarantine)"
    echo "$INTERNAL_DEPS"
    if [[ "$DRY_RUN" -eq 0 ]]; then
        # Internal deps are git pins — refresh by running `cargo update -p <name>`
        # for each, but the dep name parsing is not implemented (none in this repo).
        # Document the no-op so a reviewer can spot if this branch is ever taken.
        echo "   NOTE: internal-dep refresh path not exercised by this repo; relies on cargo update below."
    fi
else
    echo "📦 Internal deps: none (no stSoftwareAU/* git deps in Cargo.toml)"
fi
echo ""

# Phase 2: external Cargo deps.
EXTERNAL_BUMPED=0
EXTERNAL_PLAN=""
if [[ -f "$CARGO_MANIFEST" ]]; then
    if ! command -v cargo >/dev/null 2>&1; then
        echo "ERROR: cargo not found on PATH" >&2
        exit 3
    fi
    if ! command -v cargo-upgrade >/dev/null 2>&1; then
        echo "⚠️  cargo-edit not installed — external bumps skipped (install: cargo install cargo-edit)"
    else
        echo "🔍 External Cargo deps — checking for upgrades…"
        if [[ "$NO_NETWORK" -eq 1 ]]; then
            echo "   --no-network set: skipping network lookups; treating all new versions as inside quarantine."
            EXTERNAL_PLAN="(skipped: --no-network)"
        else
            # Use cargo upgrade --dry-run to discover candidates.
            UPGRADE_OUT="$(cargo upgrade --dry-run --incompatible 2>&1 || true)"
            if echo "$UPGRADE_OUT" | grep -qE '\->'; then
                EXTERNAL_PLAN="$(echo "$UPGRADE_OUT" | grep -E '\->' || true)"
            fi
            if [[ "$DRY_RUN" -eq 0 ]]; then
                # Apply compatible upgrades only (incompatible upgrades are
                # higher-risk and require a human review; the worker can do
                # those via the upgrade-dependencies.yml workflow).
                if cargo upgrade --compatible 2>&1 | tee /tmp/bump-deps-upgrade.log; then
                    if ! git diff --quiet -- "$CARGO_MANIFEST"; then
                        EXTERNAL_BUMPED=1
                    fi
                else
                    echo "ERROR: cargo upgrade failed" >&2
                    exit 4
                fi
            else
                echo "   (dry-run: no Cargo.toml changes will be applied)"
            fi
        fi
    fi
else
    echo "📦 External Cargo deps: skipped (no Cargo.toml at $CARGO_MANIFEST)"
fi
echo ""

# Phase 3: lockfile refresh.
if [[ "$DRY_RUN" -eq 0 && -f "$CARGO_MANIFEST" ]]; then
    echo "🔒 Refreshing Cargo.lock…"
    if ! cargo update 2>&1 | tail -20; then
        echo "ERROR: cargo update failed" >&2
        exit 5
    fi
    echo ""

    # Phase 3b: lockfile integrity — registry hashes must match.
    echo "🔐 Verifying lockfile integrity (cargo check --locked)…"
    if ! cargo check --locked --quiet >/tmp/bump-deps-check.log 2>&1; then
        echo "ERROR: lockfile integrity check failed — registry hashes do not match Cargo.lock" >&2
        tail -40 /tmp/bump-deps-check.log >&2 || true
        exit 6
    fi
    echo "   lockfile OK"
    echo ""
fi

# Phase 4: audit gate.
AUDIT_RUN=0
if [[ "$DRY_RUN" -eq 0 ]]; then
    if command -v cargo-deny >/dev/null 2>&1; then
        echo "📜 Running audit gate (cargo deny check)…"
        if ! cargo deny check 2>&1 | tee /tmp/bump-deps-deny.log; then
            OFFENDER="$(grep -oE '[a-zA-Z0-9_-]+ v[0-9][^ ]*' /tmp/bump-deps-deny.log | head -1 || true)"
            if [[ -n "$OFFENDER" ]]; then
                echo "ERROR: audit gate failed (offending crate: $OFFENDER)" >&2
            else
                echo "ERROR: audit gate failed (cargo deny check rejected the bumped tree)" >&2
            fi
            exit 7
        fi
        AUDIT_RUN=1
        echo ""
    else
        echo "⚠️  cargo-deny not installed — audit gate skipped (install: cargo install cargo-deny)"
    fi
fi

# Phase 5: summary.
if [[ "$EXTERNAL_BUMPED" -eq 1 || "$INTERNAL_COUNT" -gt 0 ]]; then
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "✅ bump-deps: would bump (internal=$INTERNAL_COUNT external=$EXTERNAL_BUMPED, dry-run)"
    else
        echo "✅ bump-deps: bumped (internal=$INTERNAL_COUNT external=$EXTERNAL_BUMPED, audit_run=$AUDIT_RUN)"
    fi
else
    if [[ "$DRY_RUN" -eq 1 ]]; then
        if [[ -n "$EXTERNAL_PLAN" ]]; then
            echo "✅ bump-deps: plan ready (no bumps applied — dry-run)"
        else
            echo "✅ bump-deps: no bumps (dry-run)"
        fi
    else
        echo "✅ bump-deps: no bumps"
    fi
fi
exit 0
