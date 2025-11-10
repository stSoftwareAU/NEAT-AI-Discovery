#!/usr/bin/env bash
# Shared logic for library build/versioning.
set -euo pipefail

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
  cargo build --release --lib >&2

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

# If script is executed directly (not sourced), run the function
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  ensure_lib_built
fi