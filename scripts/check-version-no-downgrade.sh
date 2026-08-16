#!/usr/bin/env bash
# check-version-no-downgrade.sh — refuse a Cargo.toml version going backwards.
#
# Usage: check-version-no-downgrade.sh <base_version> <head_version>
#
# Ensures the PR (head) crate version is not strictly older than the base
# branch (typically origin/Develop). A merge conflict that silently takes
# Develop's older version must fail CI rather than look like a deliberate
# bump to the version-increment job's `CURRENT != BASE` skip logic
# (Issue #2015).
#
# Policy (mirrors NEAT-AI / NEAT-AI-core fleet guards):
#   * head < base  → fail (downgrade)
#   * head == base → ok (CI may still auto-patch-bump)
#   * head > base  → ok (already ahead; no forced second bump)
#
# Exits 0 when the versions satisfy the policy, non-zero otherwise.
# Uses Australian English in messages (behaviour, organisation).

set -euo pipefail

die() {
  echo "check-version-no-downgrade.sh: $1" >&2
  exit 1
}

[ "$#" -eq 2 ] || die "usage: check-version-no-downgrade.sh <base_version> <head_version>"

base="${1#v}"
head="${2#v}"

for v in "$base" "$head"; do
  if ! printf '%s' "$v" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+$'; then
    die "malformed version (expected x.y.z): $v"
  fi
done

IFS='.' read -r b_major b_minor b_patch <<<"$base"
IFS='.' read -r h_major h_minor h_patch <<<"$head"

# Numeric comparison: echoes -1, 0, or 1 for head <=> base on (major, minor, patch).
cmp_triple() {
  local am=$1 an=$2 ap=$3 bm=$4 bn=$5 bp=$6
  if [ "$am" -ne "$bm" ]; then [ "$am" -gt "$bm" ] && echo 1 || echo -1; return; fi
  if [ "$an" -ne "$bn" ]; then [ "$an" -gt "$bn" ] && echo 1 || echo -1; return; fi
  if [ "$ap" -ne "$bp" ]; then [ "$ap" -gt "$bp" ] && echo 1 || echo -1; return; fi
  echo 0
}

order=$(cmp_triple "$h_major" "$h_minor" "$h_patch" "$b_major" "$b_minor" "$b_patch")

if [ "$order" -lt 0 ]; then
  die "version downgraded: ${base} -> ${head} (Cargo.toml must never go backwards vs Develop — Issue #2015)"
fi

echo "check-version-no-downgrade.sh: OK (${base} -> ${head})"
