#!/usr/bin/env bash
#
# Lint scope gate (Issue #2560).
#
# Decides whether a specialised lint workflow (actionlint, markdown lint or
# ShellCheck) has anything to check in the current change set, so the workflow
# can skip its job when a pull request touches none of the files it lints. Each
# workflow runs this script first and gates the rest of its job on the `run`
# value it prints.
#
# Usage: quality/lint_scope.sh <actionlint|markdown|shellcheck> [base [head]]
#        (default base: HEAD^1, default head: HEAD)
#
# On a pull_request run HEAD is GitHub's merge commit and HEAD^1 is the base
# branch tip, so the defaults are exactly the pull request's change set.
#
# Prints exactly one line on stdout, `run=true` or `run=false`. When
# $GITHUB_OUTPUT is set and non-empty the same line is appended to that file.
# A one-line reason goes to stderr.
#
# Fails closed: if the change set cannot be read (a shallow checkout without
# HEAD^1, an unknown ref) it warns on stderr and reports `run=true`. It never
# reports `run=false` because something could not be read. A usage error exits
# 2 and prints no `run=` line; every other outcome exits 0.
#
# Matching uses a bash `case` on each changed path, not regular expressions.
# The first changed path that matches decides the scope. The markdown rule
# mirrors the `ignores` list in .markdownlint-cli2.jsonc, which is the source
# of truth for what markdownlint lints.
#
# Bash 3.2 compatible: no mapfile, no ${var,,}, no grep -P, no GNU-only flags.

set -euo pipefail

usage() {
    echo "usage: quality/lint_scope.sh <actionlint|markdown|shellcheck> [base [head]]" >&2
}

if [[ $# -lt 1 || $# -gt 3 ]]; then
    usage
    exit 2
fi

scope="$1"
base="${2:-HEAD^1}"
head="${3:-HEAD}"

case "$scope" in
    actionlint | markdown | shellcheck) ;;
    *)
        echo "lint_scope: unknown scope '$scope'" >&2
        usage
        exit 2
        ;;
esac

# Prints the decision on stdout and appends it to $GITHUB_OUTPUT when set.
emit() {
    local line="run=$1"
    echo "$line"
    if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
        echo "$line" >> "$GITHUB_OUTPUT"
    fi
}

# Fail closed: an unreadable change set always means "run".
fail_closed() {
    echo "lint_scope: could not read the change set $base..$head; running the $scope lint (fail closed)" >&2
    emit true
    exit 0
}

if ! changed_file="$(mktemp)"; then
    fail_closed
fi
trap 'rm -f "$changed_file"' EXIT

# Written to a file rather than a pipeline so that git's exit status is kept.
# --no-renames lists BOTH the old and the new path of a rename, and deleted
# files appear too. -z keeps paths with spaces or newlines intact.
if ! git diff --name-only --no-renames -z "$base" "$head" > "$changed_file"; then
    fail_closed
fi

# Sets `reason` and returns 0 when the changed path $1 is relevant to $scope.
# Returns 1 otherwise.
reason=""
match_path() {
    local path="$1"

    # The gate script decides every gate, so a change to it re-runs them all.
    case "$path" in
        quality/lint_scope.sh)
            reason="$path changed (the lint scope gate decides every gate)"
            return 0
            ;;
    esac

    case "$scope" in
        actionlint)
            case "$path" in
                .github/workflows/* | .github/actions/* | .github/actionlint.yml | .github/actionlint.yaml)
                    reason="$path is an actionlint input"
                    return 0
                    ;;
            esac
            ;;
        markdown)
            case "$path" in
                .markdownlint* | */.markdownlint* | .github/workflows/markdown-lint.yml)
                    reason="$path is a markdown lint input"
                    return 0
                    ;;
                # Paths markdownlint never lints. This mirrors the docs/archive
                # and CHANGELOG entries of the `ignores` list in
                # .markdownlint-cli2.jsonc. The config's other entries
                # (target/**, node_modules/**, graft/**) are not mirrored, so a
                # Markdown file under them only causes an extra run, never a
                # skipped lint.
                docs/archive/* | CHANGELOG.md | */CHANGELOG.md)
                    return 1
                    ;;
                *.md)
                    reason="$path is a Markdown file"
                    return 0
                    ;;
            esac
            ;;
        shellcheck)
            case "$path" in
                *.sh | .shellcheckrc | */.shellcheckrc | .github/workflows/shellcheck.yml)
                    reason="$path is a ShellCheck input"
                    return 0
                    ;;
            esac
            ;;
    esac

    return 1
}

run=false
reason="no relevant changes between $base and $head"

while IFS= read -r -d '' path; do
    if match_path "$path"; then
        run=true
        break
    fi
done < "$changed_file"

if [[ "$run" == true ]]; then
    echo "lint_scope: $scope lint runs: $reason" >&2
else
    echo "lint_scope: $scope lint skipped: $reason" >&2
fi
emit "$run"
