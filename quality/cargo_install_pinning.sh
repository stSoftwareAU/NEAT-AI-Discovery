#!/usr/bin/env bash
#
# `cargo install` pinning gate (Issue #1912 — enforcing Issue #1223's invariant).
#
# Every `cargo install` invocation in a committed shell script or workflow must
# pass both:
#   1. `--locked`          — honour the tool's published `Cargo.lock` instead of
#                            re-resolving the whole transitive graph, so a
#                            freshly published dependency cannot slip its
#                            `build.rs` onto the runner.
#   2. `--version <X.Y.Z>` — pin the tool itself to a reviewed release.
#
# Issue #1223 pinned the workflow call sites but left the rule unenforced, so
# `scripts/fuzz-ci.sh` was missed and ran an unpinned `cargo install cargo-fuzz`
# for months. This gate scans the whole repository, not just `.github/workflows`.
#
# Scope: `cargo install` only. The nightly **rustup toolchain** channel is
# deliberately left floating and must NOT be pinned to `nightly-YYYY-MM-DD` — a
# dated nightly goes stale against the `-Z sanitizer` / `libfuzzer-sys` support
# the fuzz targets need, and rustup's signed channel carries no third-party
# `build.rs`. See README.md § Fuzz Testing (Issue #1912).
#
# Usage: quality/cargo_install_pinning.sh [root ...]    (default root: .)
#
# Exits non-zero — loudly — on any unpinned invocation, when a root does not
# exist, or when no scannable file was found at all (a gate that scans nothing
# must not report success).

set -euo pipefail

roots=("$@")
if [[ ${#roots[@]} -eq 0 ]]; then
    roots=(".")
fi

install_re='cargo[[:space:]]+(\+[^[:space:]]+[[:space:]]+)?install[[:space:]]'

# True when `cargo install` sits in command position rather than inside a
# message, comment, or usage string. `$1` is the line text preceding the match.
is_invocation() {
    local prefix="$1"
    local quotes trimmed

    # An odd number of quotes before the match means it is inside a string.
    quotes="${prefix//[^\"]/}"
    if (( ${#quotes} % 2 == 1 )); then
        return 1
    fi
    quotes="${prefix//[^\']/}"
    if (( ${#quotes} % 2 == 1 )); then
        return 1
    fi

    # Anything after a `#` on the line is a comment.
    if [[ "$prefix" == *"#"* ]]; then
        return 1
    fi

    trimmed="${prefix#"${prefix%%[![:space:]]*}"}"
    trimmed="${trimmed%"${trimmed##*[![:space:]]}"}"

    if [[ -z "$trimmed" ]]; then
        return 0
    fi
    # Shell separators / YAML list and `run:` scaffolding.
    if [[ "$trimmed" =~ (\&\&|\|\||\;|\||\(|\{|!|-)$ ]]; then
        return 0
    fi
    if [[ "$trimmed" =~ (^|[[:space:]])(then|else|do|run:|sudo|exec|time|env)$ ]]; then
        return 0
    fi
    return 1
}

pass=0
fail=0
files=0

for root in "${roots[@]}"; do
    if [[ ! -e "$root" ]]; then
        echo "cargo_install_pinning: root not found: $root" >&2
        exit 1
    fi

    while IFS= read -r -d '' file; do
        files=$((files + 1))
        while IFS= read -r match; do
            line_no="${match%%:*}"
            line="${match#*:}"

            [[ "$line" =~ $install_re ]] || continue
            if ! is_invocation "${line%%"${BASH_REMATCH[0]}"*}"; then
                continue
            fi

            if ! grep -qE -- '--locked' <<< "$line"; then
                echo "FAIL: $file:$line_no — 'cargo install' missing --locked:$line"
                fail=$((fail + 1))
                continue
            fi
            if ! grep -qE -- '--version[ =][0-9]+\.[0-9]+\.[0-9]+' <<< "$line"; then
                echo "FAIL: $file:$line_no — 'cargo install' missing --version pin:$line"
                fail=$((fail + 1))
                continue
            fi
            pass=$((pass + 1))
        done < <(grep -nE "$install_re" "$file" || true)
    done < <(find "$root" \
        \( -name target -o -name .git -o -name node_modules \) -prune -o \
        -type f \( -name '*.sh' -o -name '*.yml' -o -name '*.yaml' \) -print0)
done

if (( files == 0 )); then
    echo "cargo_install_pinning: no shell or workflow files found under: ${roots[*]}" >&2
    exit 1
fi

if (( fail > 0 )); then
    echo "❌ cargo install pinning gate failed: $fail unpinned invocation(s), $pass pinned" >&2
    exit 1
fi

echo "✅ cargo install pinning gate passed: $pass pinned invocation(s) across $files file(s)"
