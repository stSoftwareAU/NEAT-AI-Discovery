#!/usr/bin/env bash
#
# fuzz/Cargo.lock version-stamp gate (Issue #2506).
#
# `fuzz/Cargo.lock` records the root crate as a path package, and that entry
# carries the crate's version. The line goes stale on every version bump in
# Cargo.toml, so `cargo fuzz run --locked` fails with "lock file needs to be
# updated but --locked was passed". This script keeps that one line in step.
#
# Modes:
#   sync   Restamp the `version = "..."` line of the path package (the
#          [[package]] block named after the crate that has no `source = `
#          line) to the version in Cargo.toml. Only that line changes. Nothing
#          is re-resolved: no `cargo update`, no `cargo upgrade`, no network.
#          The dependency quarantine (Issue #1865) is therefore not breached.
#   check  Confirm `cargo metadata --locked` accepts fuzz/Cargo.lock. Success
#          is the positive exit status of cargo, not the absence of output.
#
# Usage: quality/fuzz_lockfile.sh <sync|check> [root]
#        (default root: the parent of this script's directory)
#
# Exits 0 on success. Exits 1, loudly, when a file is missing, when the
# [package] name or version cannot be read, when the path package is not found
# exactly once, or when cargo rejects the lockfile. Exits 2 on a usage error.

set -euo pipefail

scratch=""
cargo_err=""
# Invoked indirectly by the EXIT trap below, which ShellCheck cannot see.
# shellcheck disable=SC2317,SC2329
cleanup() {
    if [[ -n "$scratch" ]]; then
        rm -f -- "$scratch"
    fi
    if [[ -n "$cargo_err" ]]; then
        rm -f -- "$cargo_err"
    fi
    return 0
}
trap cleanup EXIT

usage() {
    echo "usage: quality/fuzz_lockfile.sh <sync|check> [root]" >&2
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage
    exit 2
fi

mode="$1"
case "$mode" in
    sync | check) ;;
    *)
        echo "fuzz-lockfile: unknown mode '$mode'" >&2
        usage
        exit 2
        ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -eq 2 ]]; then
    root="$2"
else
    root="$(cd "$script_dir/.." && pwd)"
fi

cargo_manifest="$root/Cargo.toml"
fuzz_manifest="$root/fuzz/Cargo.toml"
fuzz_lock="$root/fuzz/Cargo.lock"

for required in "$cargo_manifest" "$fuzz_manifest" "$fuzz_lock"; do
    if [[ ! -f "$required" ]]; then
        echo "fuzz-lockfile: missing file: $required" >&2
        exit 1
    fi
done

# package_field MANIFEST FIELD
# Print the quoted string value of FIELD in the [package] table of MANIFEST.
# Every other table ([lib], [package.metadata.*], ...) is ignored.
package_field() {
    awk -v field="$2" '
        /^[[:space:]]*\[/ {
            in_pkg = ($0 ~ /^[[:space:]]*\[package\][[:space:]]*(#.*)?$/)
            next
        }
        in_pkg && $0 ~ ("^[[:space:]]*" field "[[:space:]]*=") {
            line = $0
            sub(/^[^=]*=[[:space:]]*/, "", line)
            if (match(line, /^"[^"]*"/)) {
                print substr(line, 2, RLENGTH - 2)
            }
            exit
        }
    ' "$1"
}

# locate_path_package LOCKFILE CRATE
# Print `hits<TAB>line<TAB>version` for every [[package]] block named CRATE
# that carries no `source = ` line (the path package). `line` is the line
# number of that block's `version = ` line. A well-formed lockfile has hits=1.
locate_path_package() {
    awk -v crate="$2" '
        function quoted(s) {
            if (match(s, /"[^"]*"/)) {
                return substr(s, RSTART + 1, RLENGTH - 2)
            }
            return ""
        }
        function close_block() {
            if (in_block && name == crate && !has_source) {
                hits++
                hit_line = ver_line
                hit_ver = ver
            }
            in_block = 0
        }
        BEGIN { hits = 0; hit_line = 0; hit_ver = ""; in_block = 0 }
        /^\[\[package\]\]/ {
            close_block()
            in_block = 1
            name = ""
            ver = ""
            ver_line = 0
            has_source = 0
            next
        }
        /^\[/ { close_block(); next }
        in_block && /^name[[:space:]]*=/ { name = quoted($0); next }
        in_block && /^version[[:space:]]*=/ && ver_line == 0 {
            ver = quoted($0)
            ver_line = NR
            next
        }
        in_block && /^source[[:space:]]*=/ { has_source = 1; next }
        END {
            close_block()
            printf "%d\t%d\t%s\n", hits, hit_line, hit_ver
        }
    ' "$1"
}

crate="$(package_field "$cargo_manifest" name || true)"
root_ver="$(package_field "$cargo_manifest" version || true)"

re_name='^[A-Za-z0-9_-]+$'
re_version='^[0-9]+\.[0-9]+\.[0-9]+([-+][0-9A-Za-z.+-]+)?$'
if [[ ! "$crate" =~ $re_name ]]; then
    echo "fuzz-lockfile: cannot read the [package] name from $cargo_manifest" >&2
    exit 1
fi
if [[ ! "$root_ver" =~ $re_version ]]; then
    echo "fuzz-lockfile: cannot read a valid [package] version from $cargo_manifest (got '$root_ver')" >&2
    exit 1
fi

if ! found="$(locate_path_package "$fuzz_lock" "$crate")"; then
    echo "fuzz-lockfile: cannot read $fuzz_lock" >&2
    exit 1
fi
IFS=$'\t' read -r hits lock_line lock_ver <<< "$found"

if [[ "$mode" == "sync" ]]; then
    if [[ "$hits" -ne 1 ]]; then
        echo "fuzz-lockfile: $fuzz_lock must have exactly one [[package]] block named '$crate' with no source line (the path package); found $hits — refusing to guess" >&2
        exit 1
    fi
    if [[ "$lock_line" -eq 0 ]]; then
        echo "fuzz-lockfile: the path package '$crate' in $fuzz_lock has no version line" >&2
        exit 1
    fi
    if [[ "$lock_ver" == "$root_ver" ]]; then
        echo "fuzz-lockfile: $fuzz_lock already records $crate $root_ver — nothing to do"
        exit 0
    fi

    # Rewrite into a scratch file beside the lockfile, check that exactly one
    # line differs, and only then copy it over. `cat >` keeps the file's mode.
    scratch="$(mktemp "$root/fuzz/Cargo.lock.XXXXXX")"
    sed "${lock_line}s/^version[[:space:]]*=[[:space:]]*\"[^\"]*\"/version = \"$root_ver\"/" "$fuzz_lock" > "$scratch"
    changed="$(diff "$fuzz_lock" "$scratch" | grep -c '^[<>]' || true)"
    if [[ "$changed" -ne 2 ]]; then
        echo "fuzz-lockfile: restamp changed $changed line(s) of $fuzz_lock (expected 1) — leaving it untouched" >&2
        exit 1
    fi
    cat "$scratch" > "$fuzz_lock"
    echo "fuzz-lockfile: $fuzz_lock: $crate $lock_ver -> $root_ver"
    exit 0
fi

# check mode
if ! command -v cargo >/dev/null 2>&1; then
    echo "fuzz-lockfile: cargo is not on PATH — cannot run 'cargo metadata --locked'" >&2
    exit 1
fi
cargo_err="$(mktemp)"
if cargo metadata --locked --format-version 1 --manifest-path "$fuzz_manifest" >/dev/null 2>"$cargo_err"; then
    echo "fuzz-lockfile: OK — cargo metadata --locked accepts fuzz/Cargo.lock ($crate $root_ver)"
    exit 0
fi

echo "fuzz-lockfile: cargo metadata --locked failed for $fuzz_manifest:" >&2
sed 's/^/  cargo: /' "$cargo_err" >&2
if [[ "$hits" -ne 1 ]]; then
    echo "fuzz-lockfile: fuzz/Cargo.lock has $hits path package block(s) named $crate (expected 1)" >&2
elif [[ "$lock_ver" != "$root_ver" ]]; then
    echo "fuzz-lockfile: fuzz/Cargo.lock records $crate $lock_ver but Cargo.toml says $root_ver — run quality/fuzz_lockfile.sh sync" >&2
fi
exit 1
