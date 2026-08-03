#!/usr/bin/env bash
#
# Bootstrap rustup from a digest-verified installer (Issue #1911).
#
# The previous bootstrap piped https://sh.rustup.rs straight into `sh`. Pinning
# the transport (`--proto "=https" --tlsv1.2`) proves only that the bytes came
# from that host, not that they are the bytes anyone reviewed: a hijacked
# distribution point, or a proxy holding a CA the machine trusts, would have run
# arbitrary code as the invoking user — and that code installs the compiler that
# subsequently runs `build.rs` for every dependency.
#
# This script instead downloads the pinned `rustup-init` binary for the detected
# host target and refuses to execute it unless its SHA-256 matches the digest
# committed in scripts/rustup-init.sha256. Same posture as the gitleaks download
# in .github/workflows/gitleaks.yml (Issue #1217), applied to a far more
# privileged artefact.
#
# Fails closed: an unknown host target, a missing digest, a download failure, or
# a digest mismatch all exit non-zero without executing anything.
#
# Usage: scripts/install-rustup.sh [rustup-init-arg...]   (default: -y)

set -euo pipefail

# Bump this together with every digest in scripts/rustup-init.sha256.
RUSTUP_VERSION="1.29.0"

RUSTUP_ARCHIVE_BASE_URL="https://static.rust-lang.org/rustup/archive"

_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIGEST_MANIFEST="${_script_dir}/rustup-init.sha256"

# Echo the SHA-256 of "$1" as lower-case hex, using whichever standard tool the
# host provides. No digest tool means no verification, which means no install.
_sha256_of() {
  local file="$1"
  if command -v sha256sum > /dev/null 2>&1; then
    sha256sum "$file" | cut -d' ' -f1
  elif command -v shasum > /dev/null 2>&1; then
    shasum -a 256 "$file" | cut -d' ' -f1
  else
    echo "ERROR: no SHA-256 tool found (need sha256sum or shasum)." >&2
    echo "Refusing to run an unverified rustup installer." >&2
    return 1
  fi
}

# Echo the rustup target triple for this host.
_host_target() {
  local os arch libc
  case "$(uname -s)" in
    Linux) os="unknown-linux" ;;
    Darwin) os="apple-darwin" ;;
    *)
      echo "ERROR: unsupported operating system '$(uname -s)' — no pinned rustup-init available." >&2
      return 1
      ;;
  esac

  case "$(uname -m)" in
    x86_64 | amd64) arch="x86_64" ;;
    arm64 | aarch64) arch="aarch64" ;;
    *)
      echo "ERROR: unsupported architecture '$(uname -m)' — no pinned rustup-init available." >&2
      return 1
      ;;
  esac

  if [[ "$os" == "unknown-linux" ]]; then
    libc="gnu"
    if command -v ldd > /dev/null 2>&1 && ldd --version 2>&1 | grep -qi musl; then
      libc="musl"
    fi
    echo "${arch}-${os}-${libc}"
  else
    echo "${arch}-${os}"
  fi
}

# Echo the pinned digest for target "$1" from the committed manifest.
_pinned_digest() {
  local target="$1" digest=""

  if [[ ! -f "$DIGEST_MANIFEST" ]]; then
    echo "ERROR: digest manifest not found: ${DIGEST_MANIFEST}" >&2
    return 1
  fi

  local line sha name
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    [[ -z "${line// /}" ]] && continue
    read -r sha name <<< "$line"
    if [[ "$name" == "$target" ]]; then
      digest="$sha"
      break
    fi
  done < "$DIGEST_MANIFEST"

  if [[ -z "$digest" ]]; then
    echo "ERROR: no pinned rustup-init digest for target '${target}' in ${DIGEST_MANIFEST}." >&2
    echo "Add the published digest for that target before installing." >&2
    return 1
  fi

  echo "$digest"
}

install_rustup() {
  local target url tmp_dir installer expected actual
  target="$(_host_target)"
  expected="$(_pinned_digest "$target")"
  url="${RUSTUP_ARCHIVE_BASE_URL}/${RUSTUP_VERSION}/${target}/rustup-init"

  tmp_dir="$(mktemp -d)"
  # shellcheck disable=SC2064  # expand tmp_dir now, so cleanup runs on any exit
  trap "rm -rf '$tmp_dir'" EXIT
  installer="${tmp_dir}/rustup-init"

  echo "Downloading rustup-init ${RUSTUP_VERSION} for ${target}..." >&2
  if ! curl --proto "=https" --tlsv1.2 -sSfL \
    --retry 3 --retry-delay 2 --connect-timeout 30 \
    -o "$installer" "$url"; then
    echo "ERROR: failed to download rustup-init from ${url}" >&2
    return 1
  fi

  actual="$(_sha256_of "$installer")"
  if [[ "$actual" != "$expected" ]]; then
    echo "ERROR: rustup-init digest mismatch — refusing to execute the installer." >&2
    echo "  url:      ${url}" >&2
    echo "  expected: ${expected}" >&2
    echo "  actual:   ${actual}" >&2
    echo "Either the pinned digest in ${DIGEST_MANIFEST} is stale, or the" >&2
    echo "download was tampered with. Do not run the downloaded file." >&2
    return 1
  fi
  echo "rustup-init digest verified (${expected})." >&2

  chmod +x "$installer"
  "$installer" "$@"
}

# Only run when executed directly; sourcing exposes the helpers for testing.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  if [[ $# -eq 0 ]]; then
    install_rustup -y
  else
    install_rustup "$@"
  fi
fi
