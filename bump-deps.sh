#!/bin/bash
set -euo pipefail

# bump-deps.sh — refresh dependencies on every PR (Issue #1156).
#
# Supply-chain quarantine policy:
#   * Internal deps under stSoftwareAU/* (e.g. NEAT-AI itself) advance to the
#     current release / Develop HEAD immediately, with no quarantine window.
#   * External deps (crates.io, GitHub Actions, etc.) bump to the latest
#     version published BEFORE the quarantine window — VIBE_BUMP_QUARANTINE_HOURS
#     hours ago (default 24h). This dodges fast-flagged supply-chain attacks.
#   * The window is enforced against the RESOLVED graph, not just the manifest
#     requirement strings: after `cargo update` the lockfile is diffed against
#     its pre-bump state and every in-quarantine change — transitive included —
#     is pinned back with `cargo update --precise` (Issue #1865). A newly-pulled
#     package inside the window has no earlier version to pin back to, so the
#     run fails loud instead.
#   * This script — not ./quality.sh — is the only path that bumps dependencies.
#     The pre-commit quality gate verifies the tree and never mutates it.
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

# bump_deps::extract_dep_versions MANIFEST
# Print `name<TAB>version` for every top-level inline dependency declared
# in MANIFEST under [dependencies] or [dev-dependencies]. Handles both
#     foo = "1.2.3"
#     foo = { version = "1.2.3", … }
# Used to diff before/after states across a `cargo upgrade` run so we can
# revert any bump that lands inside the quarantine window (Issue #1234).
bump_deps::extract_dep_versions() {
    local manifest="$1"
    if [[ ! -f "$manifest" ]]; then
        return 0
    fi
    awk '
        /^\[/ {
            in_deps = ($0 == "[dependencies]" || $0 == "[dev-dependencies]") ? 1 : 0
            next
        }
        in_deps && /^[a-zA-Z0-9_-]+[[:space:]]*=/ {
            # Strip trailing comment.
            line = $0
            sub(/[[:space:]]*#.*/, "", line)
            # Capture name (before =).
            name = line
            sub(/[[:space:]]*=.*/, "", name)
            # Capture the first quoted string after "version" if present,
            # otherwise the first quoted string after =.
            rest = line
            sub(/^[^=]*=[[:space:]]*/, "", rest)
            version = ""
            if (match(rest, /version[[:space:]]*=[[:space:]]*"[^"]+"/)) {
                seg = substr(rest, RSTART, RLENGTH)
                match(seg, /"[^"]+"/)
                version = substr(seg, RSTART + 1, RLENGTH - 2)
            } else if (match(rest, /"[^"]+"/)) {
                version = substr(rest, RSTART + 1, RLENGTH - 2)
            }
            if (version != "") {
                printf "%s\t%s\n", name, version
            }
        }
    ' "$manifest"
}

# bump_deps::compute_changed_deps BEFORE AFTER
# Given two `name<TAB>version` listings (BEFORE and AFTER, both file
# paths), print `name<TAB>old<TAB>new` for every dep whose version
# differs between the two files. Skips entries that disappear or appear
# (those are not version bumps).
bump_deps::compute_changed_deps() {
    local before="$1"
    local after="$2"
    awk -F '\t' '
        NR == FNR { old[$1] = $2; next }
        ($1 in old) && (old[$1] != $2) { printf "%s\t%s\t%s\n", $1, old[$1], $2 }
    ' "$before" "$after"
}

# bump_deps::extract_lock_versions LOCKFILE
# Print `name<TAB>version` for every [[package]] entry in LOCKFILE — direct
# and transitive alike. The manifest gate only sees top-level requirement
# strings, so this is the only view that covers the resolved graph an
# attacker actually pivots through (Issue #1865).
bump_deps::extract_lock_versions() {
    local lockfile="$1"
    if [[ ! -f "$lockfile" ]]; then
        return 0
    fi
    awk '
        /^\[\[package\]\]/ { in_pkg = 1; name = ""; next }
        /^\[/ { in_pkg = 0; name = ""; next }
        in_pkg && /^name[[:space:]]*=/ {
            if (match($0, /"[^"]+"/)) {
                name = substr($0, RSTART + 1, RLENGTH - 2)
            }
            next
        }
        in_pkg && /^version[[:space:]]*=/ {
            if (name != "" && match($0, /"[^"]+"/)) {
                printf "%s\t%s\n", name, substr($0, RSTART + 1, RLENGTH - 2)
            }
            next
        }
    ' "$lockfile"
}

# bump_deps::compute_new_deps BEFORE AFTER
# Given two `name<TAB>version` listings, print `name<TAB>version` for every
# package present in AFTER but absent from BEFORE. A newly-pulled transitive
# package cannot be "reverted" to an earlier version, so it is age-checked
# separately (Issue #1865).
bump_deps::compute_new_deps() {
    local before="$1"
    local after="$2"
    awk -F '\t' '
        NR == FNR { old[$1] = $2; next }
        !($1 in old) { printf "%s\t%s\n", $1, $2 }
    ' "$before" "$after"
}

# bump_deps::plan_lock_quarantine BEFORE AFTER NOW_HOURS WINDOW_HOURS
# Age-check every lockfile change between the BEFORE and AFTER listings
# (both `name<TAB>version` files) and print one verdict line per package
# that must not be accepted:
#
#   revert<TAB>name<TAB>old<TAB>new<TAB>age    pin back to the old version
#   block<TAB>name<TAB>-<TAB>new<TAB>age       new package, cannot pin back
#
# `age` is either "<N>h" or "unknown". Packages published outside the
# window produce no output. Unknown publish times fail closed: a version we
# cannot date is never confirmed safe, so it is treated as in-quarantine.
bump_deps::plan_lock_quarantine() {
    local before="$1"
    local after="$2"
    local now_hours="$3"
    local window="$4"
    local name old new pub_epoch pub_hours

    while IFS=$'\t' read -r name old new; do
        [[ -z "$name" ]] && continue
        pub_epoch="$(bump_deps::fetch_publish_epoch "$name" "$new" || true)"
        if [[ -z "$pub_epoch" ]]; then
            printf 'revert\t%s\t%s\t%s\tunknown\n' "$name" "$old" "$new"
            continue
        fi
        pub_hours=$(( pub_epoch / 3600 ))
        if ! bump_deps::is_quarantine_expired "$now_hours" "$pub_hours" "$window"; then
            printf 'revert\t%s\t%s\t%s\t%sh\n' "$name" "$old" "$new" "$(( now_hours - pub_hours ))"
        fi
    done < <(bump_deps::compute_changed_deps "$before" "$after")

    while IFS=$'\t' read -r name new; do
        [[ -z "$name" ]] && continue
        pub_epoch="$(bump_deps::fetch_publish_epoch "$name" "$new" || true)"
        if [[ -z "$pub_epoch" ]]; then
            printf 'block\t%s\t-\t%s\tunknown\n' "$name" "$new"
            continue
        fi
        pub_hours=$(( pub_epoch / 3600 ))
        if ! bump_deps::is_quarantine_expired "$now_hours" "$pub_hours" "$window"; then
            printf 'block\t%s\t-\t%s\t%sh\n' "$name" "$new" "$(( now_hours - pub_hours ))"
        fi
    done < <(bump_deps::compute_new_deps "$before" "$after")
}

# bump_deps::fetch_publish_epoch NAME VERSION
# Query crates.io for the publish time of NAME@VERSION. Print the epoch
# seconds on stdout on success; print nothing and return non-zero on
# failure. The result is cached per-invocation in $BUMP_DEPS_CACHE_DIR
# (if exported) to keep retries cheap.
bump_deps::fetch_publish_epoch() {
    local name="$1"
    local version="$2"
    local url="https://crates.io/api/v1/crates/${name}/${version}"
    # Test seam: when BUMP_DEPS_TEST_FIXTURE is set, read the canned
    # response from $BUMP_DEPS_TEST_FIXTURE/<name>-<version>.json
    # instead of hitting the network. Keeps unit tests hermetic.
    local body
    if [[ -n "${BUMP_DEPS_TEST_FIXTURE:-}" ]]; then
        local fixture="$BUMP_DEPS_TEST_FIXTURE/${name}-${version}.json"
        if [[ ! -f "$fixture" ]]; then
            return 1
        fi
        body="$(cat "$fixture")"
    else
        if ! command -v curl >/dev/null 2>&1; then
            return 1
        fi
        # User-Agent is mandatory for crates.io; identify the tool.
        body="$(curl --silent --fail \
            --max-time 15 \
            -H 'User-Agent: bump-deps.sh (stSoftwareAU/NEAT-AI-Discovery; Issue #1234)' \
            "$url" 2>/dev/null || true)"
    fi
    if [[ -z "$body" ]]; then
        return 1
    fi
    # Extract `"created_at":"2025-01-15T03:45:12.345678+00:00"` without
    # depending on jq.
    local iso
    # Use POSIX BRE only — `\+` is not portable across GNU and BSD sed.
    iso="$(printf '%s' "$body" \
        | sed -n 's/.*"version"[[:space:]]*:[[:space:]]*{[^}]*"created_at"[[:space:]]*:[[:space:]]*"\([^"][^"]*\)".*/\1/p' \
        | head -n 1)"
    if [[ -z "$iso" ]]; then
        # Fall back to the first created_at anywhere in the body.
        iso="$(printf '%s' "$body" \
            | sed -n 's/.*"created_at"[[:space:]]*:[[:space:]]*"\([^"][^"]*\)".*/\1/p' \
            | head -n 1)"
    fi
    if [[ -z "$iso" ]]; then
        return 1
    fi
    # Drop sub-second + zone-offset suffix for portability; keep "YYYY-MM-DDTHH:MM:SS".
    local trimmed
    trimmed="$(printf '%s' "$iso" | sed -E 's/(\.[0-9]+)?([+-][0-9:]+|Z)?$//')"
    local epoch=""
    # GNU date: -d ISO works directly. BSD/macOS date: needs explicit format.
    if epoch="$(date -u -d "$trimmed" +%s 2>/dev/null)"; then
        :
    elif epoch="$(date -u -j -f "%Y-%m-%dT%H:%M:%S" "$trimmed" +%s 2>/dev/null)"; then
        :
    else
        return 1
    fi
    printf '%s' "$epoch"
}

# bump_deps::revert_dep_line MANIFEST NAME OLD_VERSION
# Restore the version string of NAME in MANIFEST to OLD_VERSION. Touches
# only the top-level inline declaration; nested [dependencies.<name>]
# tables are left alone (the manifest in this repo uses inline form).
bump_deps::revert_dep_line() {
    local manifest="$1"
    local name="$2"
    local old="$3"
    local tmp
    tmp="$(mktemp)"
    # Match either `name = "X.Y.Z"` or `name = { … version = "X.Y.Z" … }`
    # and rewrite only the first quoted version string on that line.
    awk -v target="$name" -v new_v="$old" '
        BEGIN { done = 0 }
        {
            if (!done) {
                # Anchor at column 0; allow optional whitespace before =.
                pattern = "^" target "[[:space:]]*="
                if ($0 ~ pattern) {
                    # Prefer rewriting `version = "…"` if present.
                    if (match($0, /version[[:space:]]*=[[:space:]]*"[^"]+"/)) {
                        seg = substr($0, RSTART, RLENGTH)
                        new_seg = seg
                        sub(/"[^"]+"/, "\"" new_v "\"", new_seg)
                        $0 = substr($0, 1, RSTART - 1) new_seg substr($0, RSTART + RLENGTH)
                    } else if (match($0, /"[^"]+"/)) {
                        seg = substr($0, RSTART, RLENGTH)
                        new_seg = "\"" new_v "\""
                        $0 = substr($0, 1, RSTART - 1) new_seg substr($0, RSTART + RLENGTH)
                    }
                    done = 1
                }
            }
            print
        }
    ' "$manifest" > "$tmp"
    mv "$tmp" "$manifest"
}

# bump_deps::current_epoch
# Print current epoch seconds. Exists so tests can stub via the
# BUMP_DEPS_NOW_EPOCH override.
bump_deps::current_epoch() {
    if [[ -n "${BUMP_DEPS_NOW_EPOCH:-}" ]]; then
        printf '%s' "$BUMP_DEPS_NOW_EPOCH"
    else
        date -u +%s
    fi
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
                          as "do not bump" for affected deps). The lockfile
                          refresh is skipped too — a re-resolve cannot be
                          age-checked without the registry (Issue #1865).
  --print-config          Print effective configuration and exit.
  --list-internal-deps    List stSoftwareAU/* internal deps and exit.
  -h, --help              Show this help and exit.

Environment:
  VIBE_BUMP_QUARANTINE_HOURS  External-dep quarantine window in hours
                              (default 24). New external versions younger
                              than this window are skipped.

Exit codes:
  0  clean (or no-op)
  8  bump rejected — a package inside the quarantine window could not be
     pinned back (new transitive package, or --precise failed).
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

# Snapshot the resolved graph BEFORE any mutation so phase 3 can age-check
# every lockfile change — including transitive packages the manifest gate
# never sees (Issue #1865).
CARGO_LOCK="$PROJECT_ROOT/Cargo.lock"
LOCK_BEFORE="$(mktemp)"
bump_deps::extract_lock_versions "$CARGO_LOCK" > "$LOCK_BEFORE"

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
EXTERNAL_REVERTED=0
EXTERNAL_PLAN=""
QUARANTINED_DEPS=""
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
                # Snapshot the manifest so we can identify which deps the
                # upgrade actually changed (Issue #1234).
                BEFORE_VERSIONS="$(mktemp)"
                AFTER_VERSIONS="$(mktemp)"
                bump_deps::extract_dep_versions "$CARGO_MANIFEST" > "$BEFORE_VERSIONS"

                # Apply compatible upgrades only (incompatible upgrades are
                # higher-risk and require a human review; raise a manual PR
                # for those — the weekly upgrade-dependencies.yml workflow
                # was removed in Issue #1282).
                if cargo upgrade --compatible 2>&1 | tee /tmp/bump-deps-upgrade.log; then
                    bump_deps::extract_dep_versions "$CARGO_MANIFEST" > "$AFTER_VERSIONS"

                    # Quarantine gate (Issue #1234): for each newly-bumped
                    # version, query crates.io for its publish time and
                    # revert any bump that is younger than the configured
                    # window. The header policy promised this — the helper
                    # `bump_deps::is_quarantine_expired` was wired up but
                    # never reached the bump path until now.
                    CHANGED="$(bump_deps::compute_changed_deps "$BEFORE_VERSIONS" "$AFTER_VERSIONS" || true)"
                    if [[ -n "$CHANGED" ]]; then
                        NOW_EPOCH="$(bump_deps::current_epoch)"
                        NOW_HOURS=$(( NOW_EPOCH / 3600 ))
                        echo "🛡️  Quarantine gate (window=${QUARANTINE_HOURS}h): checking publish times…"
                        while IFS=$'\t' read -r DEP_NAME DEP_OLD DEP_NEW; do
                            [[ -z "$DEP_NAME" ]] && continue
                            PUB_EPOCH="$(bump_deps::fetch_publish_epoch "$DEP_NAME" "$DEP_NEW" || true)"
                            if [[ -z "$PUB_EPOCH" ]]; then
                                echo "   ⚠️  $DEP_NAME@$DEP_NEW — publish time unknown; reverting to $DEP_OLD"
                                bump_deps::revert_dep_line "$CARGO_MANIFEST" "$DEP_NAME" "$DEP_OLD"
                                EXTERNAL_REVERTED=$(( EXTERNAL_REVERTED + 1 ))
                                QUARANTINED_DEPS="${QUARANTINED_DEPS} ${DEP_NAME}@${DEP_NEW}(unknown)"
                                continue
                            fi
                            PUB_HOURS=$(( PUB_EPOCH / 3600 ))
                            if bump_deps::is_quarantine_expired "$NOW_HOURS" "$PUB_HOURS" "$QUARANTINE_HOURS"; then
                                echo "   ✅ $DEP_NAME $DEP_OLD → $DEP_NEW (publish age ≥ ${QUARANTINE_HOURS}h, kept)"
                            else
                                AGE_HOURS=$(( NOW_HOURS - PUB_HOURS ))
                                echo "   🚧 $DEP_NAME $DEP_OLD → $DEP_NEW (publish age ${AGE_HOURS}h < ${QUARANTINE_HOURS}h, reverting)"
                                bump_deps::revert_dep_line "$CARGO_MANIFEST" "$DEP_NAME" "$DEP_OLD"
                                EXTERNAL_REVERTED=$(( EXTERNAL_REVERTED + 1 ))
                                QUARANTINED_DEPS="${QUARANTINED_DEPS} ${DEP_NAME}@${DEP_NEW}(${AGE_HOURS}h)"
                            fi
                        done <<< "$CHANGED"
                    fi
                    rm -f "$BEFORE_VERSIONS" "$AFTER_VERSIONS"
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
LOCK_REVERTED=0
if [[ "$DRY_RUN" -eq 0 && "$NO_NETWORK" -eq 1 ]]; then
    echo "🔒 Lockfile refresh skipped (--no-network) — a re-resolve cannot be age-checked offline."
    echo ""
elif [[ "$DRY_RUN" -eq 0 && -f "$CARGO_MANIFEST" ]]; then
    echo "🔒 Refreshing Cargo.lock…"
    if ! cargo update 2>&1 | tail -20; then
        echo "ERROR: cargo update failed" >&2
        exit 5
    fi
    echo ""

    # Phase 3a: quarantine gate over the RESOLVED graph (Issue #1865).
    # `cargo update` re-resolves transitive dependencies that the manifest
    # gate in phase 2 never inspects — the exact surface recent registry
    # compromises pivoted through. Diff the lockfile against the pre-bump
    # snapshot and pin every in-quarantine change back with --precise.
    echo "🛡️  Lockfile quarantine gate (window=${QUARANTINE_HOURS}h)…"
    LOCK_AFTER="$(mktemp)"
    bump_deps::extract_lock_versions "$CARGO_LOCK" > "$LOCK_AFTER"
    LOCK_NOW_HOURS=$(( $(bump_deps::current_epoch) / 3600 ))
    LOCK_PLAN="$(bump_deps::plan_lock_quarantine "$LOCK_BEFORE" "$LOCK_AFTER" "$LOCK_NOW_HOURS" "$QUARANTINE_HOURS")"
    LOCK_BLOCKED=""
    if [[ -n "$LOCK_PLAN" ]]; then
        while IFS=$'\t' read -r VERDICT PKG_NAME PKG_OLD PKG_NEW PKG_AGE; do
            [[ -z "$VERDICT" ]] && continue
            case "$VERDICT" in
                revert)
                    echo "   🚧 $PKG_NAME $PKG_OLD → $PKG_NEW (publish age $PKG_AGE < ${QUARANTINE_HOURS}h) — pinning back"
                    if ! cargo update -p "${PKG_NAME}@${PKG_NEW}" --precise "$PKG_OLD" >/dev/null 2>&1; then
                        echo "ERROR: could not pin $PKG_NAME back to $PKG_OLD — refusing to accept an in-quarantine dependency" >&2
                        exit 8
                    fi
                    LOCK_REVERTED=$(( LOCK_REVERTED + 1 ))
                    QUARANTINED_DEPS="${QUARANTINED_DEPS} ${PKG_NAME}@${PKG_NEW}(${PKG_AGE},lock)"
                    ;;
                block)
                    echo "   ⛔ $PKG_NAME@$PKG_NEW is a NEW package published $PKG_AGE ago — inside the quarantine window"
                    LOCK_BLOCKED="${LOCK_BLOCKED} ${PKG_NAME}@${PKG_NEW}(${PKG_AGE})"
                    ;;
                *)
                    echo "ERROR: unrecognised quarantine verdict '$VERDICT' for $PKG_NAME" >&2
                    exit 8
                    ;;
            esac
        done <<< "$LOCK_PLAN"
    fi
    if [[ -n "$LOCK_BLOCKED" ]]; then
        echo "ERROR: new dependencies published inside the ${QUARANTINE_HOURS}h quarantine window:${LOCK_BLOCKED}" >&2
        echo "       They are newly pulled, so there is no earlier version to pin back to." >&2
        echo "       Re-run ./bump-deps.sh once the window has elapsed." >&2
        exit 8
    fi

    # Confirm success positively: re-diff after pinning and fail loud if any
    # in-quarantine package survived (absence of an error is not a pass).
    bump_deps::extract_lock_versions "$CARGO_LOCK" > "$LOCK_AFTER"
    LOCK_RECHECK="$(bump_deps::plan_lock_quarantine "$LOCK_BEFORE" "$LOCK_AFTER" "$LOCK_NOW_HOURS" "$QUARANTINE_HOURS")"
    rm -f "$LOCK_AFTER"
    if [[ -n "$LOCK_RECHECK" ]]; then
        echo "ERROR: lockfile still contains in-quarantine packages after pinning:" >&2
        echo "$LOCK_RECHECK" >&2
        exit 8
    fi
    echo "   lockfile quarantine gate OK (pinned back=$LOCK_REVERTED)"
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
rm -f "$LOCK_BEFORE"
if [[ "$EXTERNAL_BUMPED" -eq 1 || "$INTERNAL_COUNT" -gt 0 ]]; then
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "✅ bump-deps: would bump (internal=$INTERNAL_COUNT external=$EXTERNAL_BUMPED, dry-run)"
    else
        echo "✅ bump-deps: bumped (internal=$INTERNAL_COUNT external=$EXTERNAL_BUMPED, quarantined=$EXTERNAL_REVERTED, lock_pinned_back=$LOCK_REVERTED, audit_run=$AUDIT_RUN)"
    fi
else
    if [[ "$DRY_RUN" -eq 1 ]]; then
        if [[ -n "$EXTERNAL_PLAN" ]]; then
            echo "✅ bump-deps: plan ready (no bumps applied — dry-run)"
        else
            echo "✅ bump-deps: no bumps (dry-run)"
        fi
    else
        echo "✅ bump-deps: no bumps (quarantined=$EXTERNAL_REVERTED, lock_pinned_back=$LOCK_REVERTED)"
    fi
fi
if [[ -n "$QUARANTINED_DEPS" ]]; then
    echo "   quarantined:$QUARANTINED_DEPS"
fi
exit 0
