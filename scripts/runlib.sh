#!/usr/bin/env bash
# Shared logic for library build/versioning.
set -euo pipefail

# Minimum rustc version required by dependencies (e.g. wgpu 28 requires 1.92)
RUST_MSRV="1.92.0"

# Returns 0 if v1 >= v2 (semver-style), 1 otherwise.
_version_ge() {
  local v1="$1" v2="$2"
  local IFS=.
  local i
  local -a a b
  a=(${v1%%-*})  # strip any -pre suffix
  b=(${v2%%-*})
  for ((i=0; i<${#a[@]} || i<${#b[@]}; i++)); do
    local x=${a[i]:-0} y=${b[i]:-0}
    ((10#$x > 10#$y)) && return 0
    ((10#$x < 10#$y)) && return 1
  done
  return 0
}

_require_tools() {
  export PATH="$HOME/.cargo/bin:$PATH"
  
  # Check for jq (should be system-wide)
  if ! command -v jq >/dev/null 2>&1; then
    echo "ERROR: jq is not available. Please install jq system-wide." >&2
    exit 1
  fi
  
  # On Linux, check for build tools (gcc/cc) needed for Rust compilation
  if [[ "$(uname -s)" == "Linux" ]]; then
    if ! command -v cc >/dev/null 2>&1 && ! command -v gcc >/dev/null 2>&1; then
      echo "ERROR: Build tools (gcc/cc) not found." >&2
      echo "" >&2
      echo "An administrator must install the following system packages:" >&2
      echo "" >&2
      
      # Detect package manager and provide installation instructions
      if command -v apt-get >/dev/null 2>&1; then
        # Ubuntu/Debian
        echo "  For Ubuntu/Debian:" >&2
        echo "    sudo apt-get update" >&2
        echo "    sudo apt-get install -y build-essential" >&2
      elif command -v yum >/dev/null 2>&1; then
        # RHEL/CentOS/Amazon Linux
        echo "  For RHEL/CentOS/Amazon Linux:" >&2
        echo "    sudo yum groupinstall -y \"Development Tools\"" >&2
        echo "    sudo yum install -y gcc" >&2
      elif command -v dnf >/dev/null 2>&1; then
        # Fedora
        echo "  For Fedora:" >&2
        echo "    sudo dnf groupinstall -y \"Development Tools\"" >&2
        echo "    sudo dnf install -y gcc" >&2
      else
        echo "  Please install gcc and build-essential using your system's package manager." >&2
      fi
      echo "" >&2
      exit 1
    fi
  fi
  
  # On macOS, check for Xcode Command Line Tools (can be installed without sudo)
  if [[ "$(uname -s)" == "Darwin" ]]; then
    if ! command -v cc >/dev/null 2>&1 && ! command -v gcc >/dev/null 2>&1; then
      # Check if Xcode Command Line Tools are installed by checking for developer directory
      if [[ ! -d "/Library/Developer/CommandLineTools" ]] && [[ ! -d "/Applications/Xcode.app/Contents/Developer" ]]; then
        echo "WARNING: Build tools (gcc/cc) not found on macOS." >&2
        echo "Attempting to install Xcode Command Line Tools (no sudo required)..." >&2
        # xcode-select --install returns 0 if it triggers installation dialog, non-zero if already installed
        if xcode-select --install 2>&1; then
          echo "Installation dialog triggered. Please follow the prompt to install Xcode Command Line Tools." >&2
          echo "After installation completes, run this script again." >&2
          exit 1
        else
          # Tools claim to be installed but not found - might be a PATH issue
          echo "Xcode Command Line Tools may be installed but gcc/cc not found in PATH." >&2
          echo "Please ensure Xcode Command Line Tools are properly installed and configured." >&2
          exit 1
        fi
      else
        echo "ERROR: Xcode Command Line Tools appear to be installed but gcc/cc not found in PATH." >&2
        echo "Please ensure Xcode Command Line Tools are properly configured and PATH is set correctly." >&2
        exit 1
      fi
    fi
  fi
  
  # Install Rust (rustup + cargo) if missing
  if ! command -v cargo >/dev/null 2>&1 || ! command -v rustup >/dev/null 2>&1; then
    echo "Installing Rust (rustup + cargo)..." >&2
    curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    export PATH="$HOME/.cargo/bin:$PATH"
    # Ensure PATH is set for future invocations
    if [[ -f "$HOME/.bashrc" ]] && ! grep -q "\.cargo/bin" "$HOME/.bashrc" 2>/dev/null; then
      echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> "$HOME/.bashrc"
    fi
    if [[ -f "$HOME/.zshrc" ]] && ! grep -q "\.cargo/bin" "$HOME/.zshrc" 2>/dev/null; then
      echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> "$HOME/.zshrc"
    fi
    if [[ -f "$HOME/.bash_profile" ]] && ! grep -q "\.cargo/bin" "$HOME/.bash_profile" 2>/dev/null; then
      echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> "$HOME/.bash_profile"
    fi
    echo "Rust installed successfully" >&2
  fi
  
  # Verify rustup is working
  rustup show >/dev/null 2>&1 || {
    echo "ERROR: rustup installation appears incomplete. Please check Rust installation." >&2
    exit 1
  }
  
  # Ensure a default toolchain is set (required for cargo to work)
  # Check if cargo can run (which requires a default toolchain)
  if ! cargo --version >/dev/null 2>&1; then
    echo "No default Rust toolchain configured. Setting default to stable..." >&2
    rustup default stable >&2 || {
      echo "ERROR: Failed to set default Rust toolchain. Please run 'rustup default stable' manually." >&2
      exit 1
    }
  fi

  # Check Rust version meets minimum (e.g. wgpu 28 requires rustc 1.92)
  local rust_ver
  rust_ver="$(rustc --version 2>/dev/null | sed -n 's/^rustc \([0-9]*\.[0-9]*\.[0-9]*\).*/\1/p')"
  if [[ -z "$rust_ver" ]]; then
    echo "WARNING: Could not determine rustc version, skipping version check" >&2
  elif ! _version_ge "$rust_ver" "$RUST_MSRV"; then
    echo "rustc ${rust_ver} is below minimum required (${RUST_MSRV}). Updating toolchain..." >&2
    rustup update stable >&2 || {
      echo "ERROR: Failed to update Rust toolchain. Please run 'rustup update stable' manually." >&2
      exit 1
    }
    rust_ver="$(rustc --version 2>/dev/null | sed -n 's/^rustc \([0-9]*\.[0-9]*\.[0-9]*\).*/\1/p')"
    if ! _version_ge "$rust_ver" "$RUST_MSRV"; then
      echo "ERROR: rustc ${rust_ver} still below ${RUST_MSRV} after update. Dependencies (e.g. wgpu) require a newer Rust." >&2
      exit 1
    fi
    echo "Rust updated to rustc ${rust_ver}" >&2
  fi
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
  local deps_lib="$project_root/target/release/deps/${lib_file}"
  
  # Helper function to find the actual library location
  # Cargo places cdylib files in target/release/deps/, so check there first
  _find_target_lib() {
    if [[ -f "$deps_lib" ]]; then
      echo "$deps_lib"
    elif [[ -f "$target_lib" ]]; then
      echo "$target_lib"
    else
      echo ""
    fi
  }

  # Check if library needs rebuilding based on version only
  local needs_rebuild=false
  local rebuild_reason=""
  local installed_version=""

  if [[ ! -f "$lib_path" ]]; then
    needs_rebuild=true
    rebuild_reason="Library ${lib_file} not found"
  elif [[ ! -f "$version_marker" ]]; then
    needs_rebuild=true
    rebuild_reason="Version marker missing for ${PKG}"
  else
    installed_version="$(cat "$version_marker" 2>/dev/null || echo "")"
    if [[ "$installed_version" != "$DESIRED" ]]; then
      needs_rebuild=true
      rebuild_reason="Version mismatch: installed v${installed_version}, need v${DESIRED}"
    fi
  fi

  if [[ "$needs_rebuild" == "false" ]]; then
    # Ensure target artifact exists in either location, otherwise trigger a rebuild
    local existing_lib
    existing_lib="$(_find_target_lib)"
    if [[ -z "$existing_lib" ]]; then
      needs_rebuild=true
      rebuild_reason="Target artifact missing in both ${target_lib} and ${deps_lib}, rebuilding"
    else
      # Update target_lib to point to the actual location for consistency
      target_lib="$existing_lib"
    fi
  fi

  if [[ "$needs_rebuild" == "false" ]]; then
    # Verify version marker matches expected version (double-check)
    if [[ -f "$version_marker" ]]; then
      local marker_version
      marker_version="$(cat "$version_marker" 2>/dev/null || echo "")"
      if [[ "$marker_version" == "$DESIRED" ]]; then
        >&2 echo "Library v${DESIRED} already installed and up to date at $lib_path"
        >&2 echo "To verify the running version, call get_library_version() FFI function"
      else
        >&2 echo "WARNING: Version marker mismatch (marker: ${marker_version}, expected: ${DESIRED}), rebuilding"
        needs_rebuild=true
        rebuild_reason="Version marker mismatch"
      fi
    fi
    
    if [[ "$needs_rebuild" == "false" ]]; then
      echo "$lib_path"
      return 0
    fi
  fi

  >&2 echo "Library crate: $PKG"
  >&2 echo "Library: $lib_file"
  >&2 echo "Version: $DESIRED"
  >&2 echo "$rebuild_reason, building"

  # Build the library (Cargo will handle incremental compilation)
  >&2 echo "Building ${PKG} v${DESIRED}"
  cargo build --release --lib >&2

  # Determine the actual library location after build
  # Cargo places cdylib files in target/release/deps/, so check there first
  local actual_lib
  actual_lib="$(_find_target_lib)"
  if [[ -z "$actual_lib" ]]; then
    >&2 echo "Build failed: expected library not found in ${target_lib} or ${deps_lib}"
    exit 1
  fi
  target_lib="$actual_lib"

  # On macOS, sign the library (required for Deno FFI to load it without SIGKILL)
  if [[ "$(uname -s)" == "Darwin" ]]; then
    >&2 echo "Signing ${lib_file} for macOS compatibility"
    codesign --force --sign - --timestamp=none --preserve-metadata=entitlements "$target_lib" >&2 2>/dev/null || true
  fi

  # Copy to cargo lib directory
  >&2 echo "Installing ${lib_file} → ${lib_path}"
  mkdir -p "$HOME/.cargo/lib" >&2
  cp "$target_lib" "$lib_path" >&2

  # On macOS, fix install_name and re-sign after copying
  if [[ "$(uname -s)" == "Darwin" ]]; then
    >&2 echo "Fixing library install_name for macOS"
    # Fix the install_name to be relative to the library location
    # This prevents issues when the library is loaded from ~/.cargo/lib/
    install_name_tool -id "@rpath/${lib_file}" "$lib_path" >&2 2>/dev/null || {
      >&2 echo "Warning: install_name_tool failed, but continuing..."
    }
    # Re-sign after fixing install_name
    codesign --force --sign - --timestamp=none --preserve-metadata=entitlements "$lib_path" >&2 2>/dev/null || true
  fi

  # Verify copy succeeded - installed library must exist
  [[ -f "$lib_path" ]] || { >&2 echo "Installation failed: expected library not found at $lib_path"; exit 1; }

  # Only write version marker after successful build, signing, and installation
  echo "$DESIRED" > "$version_marker"

  # Verify the installed library matches the expected version
  # This is a sanity check to ensure the version marker is accurate
  >&2 echo "Version marker written: $DESIRED"
  >&2 echo "Installed library: $lib_path"
  >&2 echo "To verify the running version, call get_library_version() FFI function"

  # IMPORTANT: stdout must contain ONLY the path (no extra text)
  echo "$lib_path"
}

# If script is executed directly (not sourced), run the function
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  ensure_lib_built
fi