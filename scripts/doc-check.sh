#!/bin/bash
set -euo pipefail

# Documentation build check — verifies that cargo doc builds without warnings.
#
# Usage:
#   ./scripts/doc-check.sh
#
# This treats all documentation warnings as errors, catching broken doc links,
# malformed doc comments, and missing documentation for public items.

echo "📖 Building documentation (warnings as errors)..."
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features

echo "✅ Documentation build passed — no warnings"
