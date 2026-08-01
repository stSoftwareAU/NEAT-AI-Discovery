#!/bin/bash
set -euo pipefail

# Install a Rust toolchain using the runner's preinstalled rustup (Issue #1891).
#
# This replaces `dtolnay/rust-toolchain` at every CI call site. That action's
# tarball is fetched from codeload.github.com during the runner's "Prepare all
# required actions" phase, which has a fixed 100 s HttpClient timeout and a
# 3-attempt retry policy — neither tunable from a workflow file. A codeload
# stall therefore fails the job before any repository code runs, as it did on
# PR #1890 (run 30665006027) while every sibling job using the same pinned SHA
# passed.
#
# Supply-chain posture (Issue #1216): the action was SHA-pinned so the executed
# code was immutable. This script keeps that property by removing the
# third-party code entirely — the only executable inputs are the runner image's
# rustup and the toolchain rustup fetches from static.rust-lang.org, which the
# action fetched anyway. Toolchain and component names are validated against a
# strict allowlist before reaching rustup, so nothing from the caller is
# interpreted as shell.
#
# Retries here are ours, so unlike the action download they are tunable.
#
# Usage: scripts/install-rust-toolchain.sh [TOOLCHAIN] [COMPONENT...]

usage() {
    cat <<'EOF'
Usage: install-rust-toolchain.sh [TOOLCHAIN] [COMPONENT...]

Install a Rust toolchain with the preinstalled rustup and make it the default.

Arguments:
  TOOLCHAIN    Toolchain to install (default: stable).
  COMPONENT... Extra rustup components. Accepted either as separate arguments
               or as one comma-separated argument, e.g. "rustfmt, clippy".

Environment:
  RUST_TOOLCHAIN_MAX_ATTEMPTS  Install attempts before failing (default: 3).
  RUST_TOOLCHAIN_RETRY_DELAY   Seconds between attempts (default: 15).

Examples:
  scripts/install-rust-toolchain.sh
  scripts/install-rust-toolchain.sh stable rustfmt clippy
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

MAX_ATTEMPTS="${RUST_TOOLCHAIN_MAX_ATTEMPTS:-3}"
RETRY_DELAY="${RUST_TOOLCHAIN_RETRY_DELAY:-15}"

# Allowlist: rustup toolchain and component names are plain identifiers.
# Anything else is caller error or an injection attempt — reject before use.
NAME_PATTERN='^[A-Za-z0-9][A-Za-z0-9._+-]*$'

validate_name() {
    local kind="$1" value="$2"
    if [[ ! "$value" =~ $NAME_PATTERN ]]; then
        echo "::error::install-rust-toolchain.sh: invalid $kind name '$value'" >&2
        exit 2
    fi
}

TOOLCHAIN="${1:-stable}"
if [[ $# -gt 0 ]]; then
    shift
fi
validate_name toolchain "$TOOLCHAIN"

COMPONENT_ARGS=()
for argument in "$@"; do
    # Tolerate the workflow-friendly "rustfmt, clippy" single-argument form.
    IFS=',' read -r -a parts <<< "$argument"
    for part in ${parts[@]+"${parts[@]}"}; do
        # Trim surrounding whitespace without relying on GNU-only tools.
        part="${part#"${part%%[![:space:]]*}"}"
        part="${part%"${part##*[![:space:]]}"}"
        [[ -z "$part" ]] && continue
        validate_name component "$part"
        COMPONENT_ARGS+=(--component "$part")
    done
done

if ! command -v rustup > /dev/null 2>&1; then
    echo "::error::install-rust-toolchain.sh: rustup was not found on PATH." >&2
    echo "The GitHub-hosted runner images ship rustup; a missing rustup means" >&2
    echo "the runner image changed and this step needs revisiting." >&2
    exit 1
fi

echo "Installing Rust toolchain '$TOOLCHAIN' (attempts: $MAX_ATTEMPTS)"

attempt=1
while true; do
    if rustup toolchain install "$TOOLCHAIN" \
        --profile minimal \
        --no-self-update \
        ${COMPONENT_ARGS[@]+"${COMPONENT_ARGS[@]}"}; then
        break
    fi
    if [[ "$attempt" -ge "$MAX_ATTEMPTS" ]]; then
        echo "::error::install-rust-toolchain.sh: failed to install toolchain" \
            "'$TOOLCHAIN' after $MAX_ATTEMPTS attempts." >&2
        exit 1
    fi
    echo "Attempt $attempt of $MAX_ATTEMPTS failed; retrying in ${RETRY_DELAY}s..." >&2
    sleep "$RETRY_DELAY"
    attempt=$((attempt + 1))
done

rustup default "$TOOLCHAIN"

# Positively confirm the toolchain runs. A silent no-op install must not be
# reported as success.
rustup run "$TOOLCHAIN" rustc --version
rustup run "$TOOLCHAIN" cargo --version

# Later steps in the job get cargo's shims on PATH, matching what the action did.
if [[ -n "${GITHUB_PATH:-}" ]]; then
    echo "$HOME/.cargo/bin" >> "$GITHUB_PATH"
fi

echo "Rust toolchain '$TOOLCHAIN' ready."
