#!/usr/bin/env bash
# Shared logic for library build/versioning.
set -euo pipefail

_require_tools() {
  export PATH="$HOME/.cargo/bin:$PATH"
  for cmd in cargo rustup jq; do
    command -v "$cmd" >/dev/null 2>&1 || {
      echo "Missing required tool: $cmd. Install prerequisites (Rust + rustup + jq). See README 'Prerequisites'." >&2
      exit 1
    }
  done
  rustup show >/dev/null
}

# Must be run from the crate dir (where Cargo.toml lives).
_meta_for_here() {
  local manifest="$PWD/Cargo.toml"
  [[ -f "$manifest" ]] || { echo "Cargo.toml not found in $PWD" >&2; exit 1; }

  local meta pkg_name desired_ver
  meta="$(cargo metadata --no-deps --format-version 1)"
  pkg_name="$(echo "$meta" | jq -r --arg MP "$manifest" '.packages[] | select(.manifest_path==$MP) | .name')"
  desired_ver="$(echo "$meta" | jq -r --arg MP "$manifest" '.packages[] | select(.manifest_path==$MP) | .version')"

  [[ -n "$pkg_name" && "$pkg_name" != "null" ]] || { echo "Cannot identify package" >&2; exit 1; }
  [[ -n "$desired_ver" && "$desired_ver" != "null" ]] || { echo "Cannot read version for '$pkg_name'" >&2; exit 1; }

  echo "$pkg_name" "$desired_ver"
}

# Build library if missing or version changed. Echoes the final path.
# This works for libraries by using version files instead of --version.
ensure_lib_built() {
  _require_tools
  read -r PKG DESIRED < <(_meta_for_here)

  # Find the project root (where the main Cargo.toml is located)
  local project_root
  project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  
  # Determine library file based on platform
  local lib_name="lib${PKG}"
  local lib_file
  case "$(uname -s)" in
    Darwin)
      lib_file="${lib_name}.dylib"
      ;;
    Linux)
      lib_file="${lib_name}.so"
      ;;
    *)
      lib_file="${lib_name}.rlib"
      ;;
  esac
  
  local lib_path="$HOME/.cargo/lib/${lib_file}"
  local version_marker="$HOME/.cargo/lib/.${PKG}.version"
  local target_lib="$project_root/target/release/${lib_file}"

  # Check if library needs rebuilding based on version only
  local needs_rebuild=false
  local rebuild_reason=""

  if [[ ! -f "$lib_path" ]]; then
    needs_rebuild=true
    rebuild_reason="Library ${lib_file} not found"
  elif [[ ! -f "$version_marker" ]]; then
    needs_rebuild=true
    rebuild_reason="Version marker missing for ${PKG}"
  else
    # Check version from marker file
    local installed_version
    installed_version="$(cat "$version_marker" 2>/dev/null || echo "")"
    if [[ "$installed_version" != "$DESIRED" ]]; then
      needs_rebuild=true
      rebuild_reason="Version mismatch: installed v${installed_version}, need v${DESIRED}"
    fi
  fi

  if [[ "$needs_rebuild" == "false" ]]; then
    # Silent when up-to-date - just return the path
    echo "$lib_path"
    return 0
  fi

  # Show rebuild reason
  >&2 echo "Library crate: $PKG"
  >&2 echo "Library: $lib_file"
  >&2 echo "Version: $DESIRED"
  >&2 echo "$rebuild_reason, building"

  # Build the library (Cargo will handle incremental compilation)
  >&2 echo "Building ${PKG} v${DESIRED}"
  cargo build --release --lib --locked >&2

  # Copy to cargo lib directory
  >&2 echo "Installing ${lib_file} → ${lib_path}"
  mkdir -p "$HOME/.cargo/lib" >&2
  cp "$target_lib" "$lib_path" >&2

  # Store version marker for future checks
  echo "$DESIRED" > "$version_marker"

  [[ -f "$lib_path" ]] || { >&2 echo "Expected library not found at $lib_path"; exit 1; }

  # IMPORTANT: stdout must contain ONLY the path (no extra text)
  echo "$lib_path"
}

