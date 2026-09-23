# Security sweep — chunk `16`: build/install shell scripts (`scripts/*.sh`, `quality.sh`, `benchmark.sh`)

Ledger rules: [`README.md`](README.md). Index entry:
[`lib-sweep-coverage.json`](lib-sweep-coverage.json).

## Record

- **Chunk id:** `16` — matches the `id` in the index.
- **Human name:** Build/install shell scripts — `scripts/*.sh`, `quality.sh`,
  `benchmark.sh`.
- **Sweep date:** `2026-09-22`
- **Baseline commit:** `4f269d6b604eb480b9efe5267628d10484ea25b1` — the commit
  actually read, line for line. The tracker issue named `b85a551` as its
  baseline (`b85a551ed2521ed327469b20eb88aeda828357d2`, 2,066 lines across the
  eleven files); `git diff b85a551..4f269d6 -- scripts/ quality.sh benchmark.sh`
  is **not** empty — `scripts/runlib.sh` grew from 1,130 to 1,164 lines and
  `scripts/doc-check.sh` from 15 to 21 — so both SHAs are recorded and the
  2,106 lines at `4f269d6` are what this record describes.
- **Exposure:** `local` — none of these scripts is reachable over a network.
  They run on a developer's machine, a fleet build host, or a GitHub Actions
  runner, as the invoking user.
- **Swept by:** Issue #2128 (the chunk-16 audit sub-issue of #2097).
- **Tracker issue:** `#2083` (overflow tracker) via `#2097` (chunk 16).

## Citation convention — a deliberate departure

`CONTRIBUTING.md` → "Cite Code by Symbol, Never by Line Number" (Issue #1942)
says documentation cites `<file>::<symbol>`, because line numbers rot silently.
**This record cites `file:line` anyway**, and does so deliberately:

- The sub-issue driving this sweep makes it an acceptance criterion — "the
  complete script table with `file:line` for every strict-mode and download
  cell" — because a `set -euo pipefail` line and a bare `curl` invocation have
  no enclosing symbol to name in several of these scripts.
- Unlike a live code document, a sweep record is **pinned to a baseline commit**
  and carries the `git diff` command that falsifies it (see *Verify this
  record*). A citation here is a claim about one commit, not about HEAD, so it
  cannot rot unnoticed — the diff goes non-empty and the record is stale by its
  own rule.
- Where a symbol exists, it is named **alongside** the line, so the citation
  survives a shift even before the diff is run.

The rot risk is real for `scripts/runlib.sh`, which is byte-synced from
NEAT-AI-core on every pull request and can therefore move with no edit in this
repository. Its citations are the ones to re-derive from the named symbol first.

## Files swept

2,106 lines, read in full. Line counts as at the baseline commit.

| Path | Lines | Outcome |
| --- | --- | --- |
| `scripts/runlib.sh` | 1164 | accepted: staging names under `$CARGO_HOME` are predictable, but planting the symlink already needs write access to the destination directory — no privilege is crossed (see [runlib.sh](#scriptsrunlibsh--1164-lines)) |
| `scripts/benchmark-ci.sh` | 217 | clean — the only externally-set value, `BENCHMARK_THRESHOLD`/`--threshold`, is validated by `benchmark_threshold::require_valid`, called at `benchmark-ci.sh:101` before it reaches `bc` |
| `benchmark.sh` | 165 | finding #2140 — `eval "$cmd" … \|\| true` (`:32`) times a suite that never completed and prints it as a speed-up |
| `scripts/install-rustup.sh` | 151 | finding #2127 — `$tmp_dir` interpolated into the `EXIT` trap string (`:117`); the #1911 digest path itself re-verified clean |
| `scripts/install-rust-toolchain.sh` | 126 | finding #2126 — `RUST_TOOLCHAIN_MAX_ATTEMPTS` reaches the `[[ -ge ]]` arithmetic unvalidated (`:104`); the argument allowlist (`:57-65`) is clean |
| `scripts/fuzz-ci.sh` | 73 | accepted: `MAX_TIME` (`:15`) reaches libFuzzer as one quoted argv element (`:59`), never an arithmetic or `eval` context, and a bad value fails loud |
| `scripts/check-version-no-downgrade.sh` | 56 | clean — both versions are regex-validated (`:33`) before any arithmetic test |
| `scripts/check-pr-summary-location.sh` | 35 | finding #2139 — a `find` that never ran is reported as ✅ with exit `0` (`:22`, `:35`) |
| `scripts/benchmark_threshold.sh` | 29 | clean — three pure `[[ =~ ]]` predicates; no `set` line by design (sourced library, see the script table) |
| `scripts/doc-check.sh` | 21 | clean — two fixed `cargo` invocations, no inputs |
| `quality.sh` | 69 | accepted: sources `$HOME/.cargo/env` (`:7`) and mutates the tree with `cargo fmt --all` (`:47`) — both are the documented contract of a local pre-commit gate |

## Defect classes probed

All nine classes named in #2097, applied to every line of every file:

- **Command injection** — a caller value reaching `eval`, a command position, or
  a re-parsed string.
- **Arithmetic-context injection** — an unvalidated value inside `(( ))`,
  `$(( ))` or a `[[ … -eq … ]]` test, where bash evaluates an array subscript
  and so a command substitution.
- **Unvalidated environment input** — `$CARGO_HOME`, `$HOME`, `$TMPDIR`,
  `$GITHUB_PATH`, `$BENCHMARK_*`, `$RUST_TOOLCHAIN_*`, `$RUSTFLAGS`.
- **Unverified downloads and installs** — every `curl`, `rustup`, `cargo
  install` and `pip install`: transport, pin, and whether the artefact is
  checked before it is executed.
- **Destructive filesystem operations** — every `rm -rf`, `mv -f`, `cp` and
  `mkdir -p`: what bounds the path, and whether a symlink can redirect it.
- **Temporary-file handling** — predictable names, fixed `/tmp` paths, races
  between creation and use, and whether a trap reaps what it created.
- **Fail-silent** — `2>/dev/null`, `|| true`, an unchecked pipeline or process
  substitution, or any path where the absence of an explicit failure is read as
  success.
- **Secret exposure** — `set -x`, tokens or credentials in arguments,
  environment echoes, or error strings.
- **Strict-mode and lint coverage** — whether each script actually runs under
  `set -euo pipefail` and is actually reached by the committed gates.

**Not covered by this sweep:** `bump-deps.sh` and `benchmark_compare.sh` (swept
under #1908/#1909/#1910/#1918); the workflow YAML under `.github/workflows/`
except where a chunk-16 script is called from it; and the Rust code every one of
these scripts eventually invokes — that is chunks 2, 4, 7, 8a, 8b, 9, 11 and 13.

## Script table

Every cell verified by reading the file at `4f269d6`.

| script | shebang | strict mode (line) | `bash -n` gated | shellcheck gated | downloads (URL, curl flags) | checksum verification |
| --- | --- | --- | --- | --- | --- | --- |
| `scripts/runlib.sh` | `#!/usr/bin/env bash` (`:1`) | `set -euo pipefail` (`:86`) | yes | yes | `_runlib_bootstrap_rustup`, `:543` `curl --proto "=https" --tlsv1.2 -sSfL --retry 3 --retry-delay 2 --connect-timeout 30 -o` → `https://static.rust-lang.org/rustup/archive/1.29.0/<target>/rustup-init` (`:422`, `:421`, `:532`). Indirect: `rustup toolchain install` (`:662`, `:787`), `rustup default stable` (`:666`), `rustup update` (`:729`) | yes — SHA-256 computed by `_runlib_sha256_of` (`:547`), compared with the inlined pin from `_runlib_pinned_rustup_digest` (`:428-445`) at `:548-549`, and the file is `chmod +x`'d `:551` and executed `:553` only after the match. Indirect rustup fetches rely on rustup's own signed channel manifests |
| `scripts/install-rustup.sh` | `#!/usr/bin/env bash` (`:1`) | `set -euo pipefail` (`:23`) | yes | yes | `install_rustup`, `:121-123` `curl --proto "=https" --tlsv1.2 -sSfL --retry 3 --retry-delay 2 --connect-timeout 30 -o` → `https://static.rust-lang.org/rustup/archive/1.29.0/<target>/rustup-init` (`:28`, `:26`, `:113`) | yes — SHA-256 computed by `_sha256_of` (`:128`, defined `:35-46`), compared with the pin read from `scripts/rustup-init.sha256` by `_pinned_digest` (`:81-107`, digests at `rustup-init.sha256:19-24`) at `:129-137`; `chmod +x` `:140` and execute `:141` are after the comparison |
| `scripts/install-rust-toolchain.sh` | `#!/bin/bash` (`:1`) | `set -euo pipefail` (`:2`) | yes | yes | indirect only — `rustup toolchain install` (`:98-101`), `rustup default` (`:114`), `rustup run` (`:118-119`) | n/a for this script — rustup verifies its own signed channel manifests; nothing is downloaded and executed by this file |
| `scripts/benchmark-ci.sh` | `#!/bin/bash` (`:1`) | `set -euo pipefail` (`:2`) | yes | yes | none | n/a |
| `scripts/benchmark_threshold.sh` | `#!/bin/bash` (`:1`) | **none — by design.** A sourced library, never executed: `benchmark-ci.sh` sets `set -euo pipefail` at `:2` and sources it at `:45`, so the functions always run under the caller's strict mode. `benchmark_compare.sh` (out of scope) is the other caller. A `set` line here would instead impose strict mode on any future caller that had not opted in — recorded as a deliberate omission, not a gap | yes | yes | none | n/a |
| `scripts/fuzz-ci.sh` | `#!/bin/bash` (`:1`) | `set -euo pipefail` (`:2`) | yes | yes | indirect — `rustup install nightly` (`:29`) and `cargo +nightly install --locked --version 0.13.2 cargo-fuzz` (`:42`) | pinned rather than digest-checked: `--locked` + `--version 0.13.2` (`:42`) fix the resolved graph, enforced by `quality/cargo_install_pinning.sh` (`quality.sh:24`); crates.io checksums are verified by cargo against `Cargo.lock`. `rustup install nightly` (`:29`) floats by design — see the re-verified #1912 row |
| `scripts/check-version-no-downgrade.sh` | `#!/usr/bin/env bash` (`:1`) | `set -euo pipefail` (`:20`) | yes | yes | none | n/a |
| `scripts/check-pr-summary-location.sh` | `#!/bin/bash` (`:1`) | `set -euo pipefail` (`:9`) | yes | yes | none | n/a |
| `scripts/doc-check.sh` | `#!/bin/bash` (`:1`) | `set -euo pipefail` (`:2`) | yes | yes | none | n/a |
| `quality.sh` | `#!/bin/bash` (`:1`) | `set -euo pipefail` (`:2`) | yes | yes | none directly — `cargo deny check` (`:40`) and the `cargo build`/`test` steps fetch through the committed `Cargo.lock` | n/a — cargo verifies registry checksums against `Cargo.lock` |
| `benchmark.sh` | `#!/bin/bash` (`:1`) | `set -euo pipefail` (`:12`) | yes | yes | none | n/a |

## Lint coverage

All eleven scripts are reached by both committed gates, and the coverage is
verified rather than assumed:

- **`bash -n`** — `quality/bash_syntax.sh:43-45`:
  `find "${roots[@]}" \( -path '*/target' -o -path '*/.git' -o -path
  '*/node_modules' \) -prune -o -name '*.sh' -type f -print0`. The only pruned
  paths are `target`, `.git` and `node_modules`; none of the eleven lives under
  one. The gate refuses to report success when it scanned nothing
  (`:47-51`), so a broken root cannot pass silently.
- **ShellCheck** — `quality/shellcheck.sh:51-53` uses the identical `find`
  predicate and the same scanned-nothing refusal (`:55-59`), plus a hard failure
  when `shellcheck` is not installed (`:25-28`).
- **Call sites, both gates, both places:** `./quality/bash_syntax.sh .` and
  `./quality/shellcheck.sh .` from `quality.sh:16,20` (the local pre-commit
  gate) and from `.github/workflows/shellcheck.yml:48,61` (the pull-request
  gate). The root argument is `.`, so the scan covers the whole checkout — both
  `scripts/*.sh` and the two root-level scripts.
- **Negative check — no blind spot in the `-name '*.sh'` filter.** A shebang
  grep over the tree excluding `target/`, `.git/` and `node_modules/`
  (`grep -rIl '^#!.*\(bash\|sh\)$'`) returns **no** shell script without a `.sh`
  suffix. Every shell script in the repository today is therefore matched by
  both gates' filter. This is a property of the tree as at `4f269d6`, not a
  guarantee: a future extensionless script would be silently unlinted.
- **`cargo install` pinning** — `quality/cargo_install_pinning.sh`, run from
  `quality.sh:24`, is the third committed gate over this chunk; it is what keeps
  `fuzz-ci.sh:42` carrying both `--locked` and `--version`.

## Negative checks

Recorded so a later reader knows these were looked for and not found, across all
eleven files:

- **No `set -x` / `set +x`** anywhere — nothing traces a command line that could
  carry a credential.
- **No token, secret, password or API-key reference** — a case-insensitive grep
  for `token|secret|password|api[-_]?key` matches nothing.
- **No fixed `/tmp` path** — the only temporaries come from `mktemp -d`
  (`install-rustup.sh:115`, `runlib.sh:536`); every other staged file lives under
  `$CARGO_HOME`.

## Re-verified remediations

Prior fixes on this chunk, re-read at `4f269d6` against the current code rather
than trusted from their issue text.

| issue | guard | citing `file:line` | test | on live path? |
| --- | --- | --- | --- | --- |
| #1911 | rustup-init is executed only when its SHA-256 matches a committed pin | `install-rustup.sh::install_rustup` (`:128-137` then `:140-141`); `runlib.sh::_runlib_bootstrap_rustup` (`:547-549` then `:551,553`); pins at `rustup-init.sha256:19-24` and `runlib.sh::_runlib_pinned_rustup_digest` (`:428-445`) | `tests/issue_1911_rustup_digest_verification.rs::rejects_a_tampered_download_without_executing_it`, `::fails_loud_when_the_download_fails`, `::fails_closed_when_no_digest_is_pinned_for_the_host_target`, `::fails_closed_when_the_digest_manifest_is_missing`; `tests/issue_2072_canonical_runlib.rs::a_tampered_rustup_init_is_refused_without_being_executed`; **new** `tests/issue_2097_rustup_pin_parity.rs` | yes — `runlib.sh:514-562` is the bootstrap every fleet host takes when it has no `rustc` |
| #1912 | `cargo-fuzz` installed with `--locked` and an explicit `--version` | `fuzz-ci.sh:42`, gated by `quality/cargo_install_pinning.sh` (`quality.sh:24`) | `tests/issue_1912_fuzz_ci_pinned_install.rs::cargo_fuzz_is_installed_with_locked_and_a_version_pin` | yes — the only `cargo install` in the chunk |
| #1913 | codespell installed from a hash-pinned requirements file | `.github/workflows/ci.yml:543` `pip install --user --require-hashes -r .github/requirements/codespell-requirements.txt`; the file pins `codespell==2.4.3` with both wheel and sdist SHA-256 | `tests/issue_1913_codespell_pin.rs::codespell_install_uses_the_hash_pinned_requirements_file` | yes — runs on every PR |

### #1911, in detail

- **Enforced, not advisory.** `install-rustup.sh::install_rustup` (`:129`) compares the computed
  digest with the pin and `return 1`s at `:136` *before* `chmod +x` (`:140`) and
  the execution (`:141`). There is no `else` branch, no override flag and no
  environment variable that skips the comparison. `runlib.sh::_runlib_bootstrap_rustup` (`:548-549`) is the
  same shape via `_runlib_die`.
- **Every failure path exits non-zero.** Download failure
  `install-rustup.sh:121-126`; missing manifest `:84-87`; unknown host target
  `:54-66`; no digest for the host target `:100-104`; no SHA-256 tool `:41-45`.
  `runlib.sh` additionally checks the digest tool (`:526-527`) and `curl`
  (`:530-531`) *before* fetching anything, so an unverifiable install is refused
  rather than dressed up as a network error.
- **Pin provenance is documentary, and must stay that way.**
  `scripts/rustup-init.sha256:8-11` records that each digest is the contents of
  the `.sha256` file the Rust project publishes beside the artefact, verified
  against the download on 2026-08-03. Re-downloading in CI to "confirm" a pin
  would verify the artefact against itself and prove nothing; the pin's value is
  that a human compared it with the upstream publication once. **Do not add a
  CI step that regenerates these digests.**
- **Redirects.** `-L` is passed, so `curl` follows redirects. That is harmless
  under the digest check — a redirected download still has to hash to the pin —
  and `--proto "=https"` bounds the redirect chain to HTTPS, so a `Location:`
  header cannot downgrade the transport or reach a `file://` URL.
- **The gap this sweep closed.** The two pin sets — `rustup-init.sha256:19-24`
  plus `install-rustup.sh:26`, and `runlib.sh:421,428-445` — carry the same six
  digests and the same version, and both files say in prose that they must move
  together (`runlib.sh:58-60`, `rustup-init.sha256:13-15`). Nothing asserted it.
  `tests/issue_2097_rustup_pin_parity.rs` now does: it compares the two
  `target → digest` maps in both directions and the two version pins, from file
  reads alone.

### #1912, in detail

The floating `nightly` channel (`fuzz-ci.sh:20-26`) is **accepted** with the
reason the script itself states: `cargo-fuzz` builds with `-Z sanitizer`, and a
dated `nightly-YYYY-MM-DD` goes stale against the sanitiser and `libfuzzer-sys`
support the targets need. The toolchain comes from rustup's signed channel, not
crates.io, so it carries no third-party `build.rs` — which is the risk the
`cargo-fuzz` version pin at `:42` closes.

## Findings

| issue | `file:line` | class | severity | status |
| --- | --- | --- | --- | --- |
| [#2126](https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/2126) | `scripts/install-rust-toolchain.sh:52,104` | arithmetic-context injection via unvalidated environment | low | open — filed before this sweep, re-confirmed here |
| [#2127](https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/2127) | `scripts/install-rustup.sh:115-117` | temporary path interpolated into an `EXIT` trap string | low | open — filed before this sweep, re-confirmed here |
| [#2139](https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/2139) | `scripts/check-pr-summary-location.sh:22,35` | fail-silent | low | open — filed by this sweep |
| [#2140](https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/2140) | `benchmark.sh:32` | fail-silent + `eval` on a variable | low | **fixed** — `benchmark.sh::run_benchmark` now takes the command as arguments and invokes `"$@"` (no `eval` remains in the file); a non-zero status prints the label, the argv and the last 20 lines of the captured output on stderr and `exit 1`s, so no duration or improvement figure is produced for a run that did not complete. Guarded by `tests/issue_2140_benchmark_failure_is_loud.rs::a_failing_benchmark_command_fails_the_script_loudly`, which drives the real script with a failing stub `cargo` |

Not a security finding, filed separately so it does not dilute the four above:
[#2141](https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/2141) —
`benchmark.sh:31,33` uses `date +%s.%N`, a GNU extension, so every timing fails
on macOS.

`negative-result` does **not** apply to this chunk: four findings survived
triage.

## Per-file detail

### `scripts/runlib.sh` — 1,164 lines

**Copy contract.** `runlib.sh:4-7` and `.github/workflows/family-sync.yml:1-8`:
this file has one home, `scripts/runlib.sh` on NEAT-AI-core `Develop`, and is
byte-synced into every Rust sibling on every pull request. **It is never edited
here.** A `runlib.sh` finding is filed in this repository in the house format
with a note that the fix lands in `stSoftwareAU/NEAT-AI-core`. This sweep filed
none — see the accepted disposition below.

**Examined and found clean:**

- **Bootstrap** — `_runlib_bootstrap_rustup` (`:514-562`). Every precondition — host target `:517-519`,
  pinned digest `:520-522`, a SHA-256 tool `:526-527`, `curl` `:530-531` — is
  checked *before* anything is fetched, so an unverifiable install is refused
  rather than downloaded. The `mktemp -d` (`:536`) is tracked (`:538`) and reaped
  by `_runlib_cleanup_temps` (`:865-879`) under a trap that names a **function**
  (`:539`), not an interpolated string — the defect #2127 records in
  `install-rustup.sh:117` does not exist here.
- **Toolchain names.** `_runlib_assert_toolchain_name` (`:594-599`) enforces
  `^[A-Za-z0-9][A-Za-z0-9._+-]*$` on every value handed to rustup: the
  `rust-toolchain.toml` channel (`:660`), the channel being updated (`:726`) and
  the required version from `cargo metadata` (`:786`). Both sources are
  repository input, and both are validated.
- **The `rm -rf` in `_runlib_remove_target` (`:839`).** Four guards stand before it: the path must be
  absolute (`:810-811`), must not be `/` (`:812-813`), both sides are resolved
  with `cd … && pwd -P` (`:819-824`), and the resolved target must sit under the
  resolved repository root (`:825-832`). Two specific questions were asked and
  answered by reading:
  - *A `target/` symlinked outside the checkout* — `:819` resolves the symlink
    to its real path, so the `case` at `:825` takes the `*)` arm, the directory
    is **kept**, and the fact is printed (`:828-829`). Confirmed.
  - *Can `CARGO_TARGET_DIR` move the deletion outside the repository root?* No.
    The path comes from `cargo metadata`'s `.target_directory` (`:991`), already
    canonicalised by cargo, and is deleted only when it resolves under the
    resolved repository root. A poisoned `CARGO_TARGET_DIR` can only make the
    directory be **kept**, never widen the deletion. Confirmed.
  - One residual, **accepted**: `case "$target_dir" in "$repo_root"/*)`
    (`:825-826`) expands `$repo_root` as a *glob pattern*, so a checkout path
    containing `*`, `?` or `[…]` would be matched loosely. It is not filed
    because the precondition is a checkout directory whose own name carries
    shell glob metacharacters — not something any actor in this chunk's attacker
    model (a co-tenant process, a poisoned environment variable, a malicious
    argument) can choose. Were it ever to matter, the fix is a prefix strip
    (`[[ "${target_dir#"$repo_root"/}" != "$target_dir" ]]`) rather than a
    `case` pattern, and it lands in NEAT-AI-core.
- **Arithmetic.** `_runlib_version_ge` (`:274-291`) rewrites every non-numeric
  component to `0` (`:285-286`) before `((10#$x > 10#$y))` (`:287-288`), so a
  `rust-version` typo or a build-metadata tail cannot reach an arithmetic
  context. `:835` regex-checks `kilobytes` against `^[0-9]+$` and dies loud
  (`:836`) before the multiplication at `:838`.
- **`PATH` handling.** `:559` and `:568` prepend `$CARGO_HOME/bin`. Accepted:
  `CARGO_HOME` is the variable rustup and cargo themselves honour, so anyone who
  can set it already directs where cargo reads and writes its toolchains. The
  bootstrap passes `--no-modify-path` (`:553`) so nothing is persisted into a
  shell rc file.
- **Sourcing side effects** (`:81-85`) are documented in the header: sourcing
  applies `set -euo pipefail` to the calling shell, prepends to its `PATH`, and
  clears its `EXIT`/`INT`/`TERM` traps. Declared, not silent.
- **Pipeline honesty.** `:466-476` captures `ldd --version` before matching it
  precisely because `ldd` exits non-zero on musl and a `ldd | grep` pipeline
  under `pipefail` would report the failure rather than the match — a musl host
  would otherwise have silently taken the gnu installer. This is the fail-silent
  class caught and handled.

**Accepted with reason — predictable staging names.** `runlib_install` (`:1058`, `:1083`, `:1098`) stages under `$CARGO_HOME/bin/<name>.runlib.$$`,
`$CARGO_HOME/lib/<name>.runlib.$$` and `$CARGO_HOME/bin/<name>.runlib-prev.$$`,
and the `cp` calls at `:1060`, `:1085` and `:1100` follow a symlink already
sitting at that path. The name is predictable — `$$` is a PID, which is
guessable. It is nonetheless **not filed**: `$CARGO_HOME` defaults to
`$HOME/.cargo`, a directory owned by the invoking user, and planting the symlink
requires write access to exactly the directory the script is about to install
into. An attacker holding that access does not need the race — they can replace
`$CARGO_HOME/bin/<crate>` directly, which is both simpler and more durable. No
privilege boundary is crossed, so there is no finding to file. (Were the
directory ever made group- or world-writable, this would become real
immediately; the mitigating fix is `mktemp` inside the destination directory.)

### `benchmark.sh` — 165 lines

- **Finding #2140** — `benchmark.sh::run_benchmark` (`:32`) runs `eval "$cmd" > /dev/null 2>&1 || true`. It is the
  only `eval` in the chunk. Every caller today passes a hard-coded literal
  (`:98-99`, `:126-127`), so there is no injection reachable now — the live
  defect is that `|| true` plus the discarded streams turn a suite that failed
  to build into a *timing*, which `calc_improvement` (`:137-145`) then prints as
  a speed-up (`:157-158`). Filed with the `eval` removal alongside, because
  `"$@"` closes the injection class for one line. **Fixed** — `run_benchmark`
  takes `label` then the command as arguments and runs `if ! "$@"`, capturing
  output to an `mktemp` log and printing its last 20 lines with the label and
  argv on stderr before `exit 1`. All four call sites pass argv rather than a
  string and carry an explicit `|| exit 1`, so the abort does not rely on the
  command-substitution form. No `eval` remains in the file.
- **Accepted with reason — blast radius.** `:62` `git stash push`, `:95`/`:111`
  `git checkout`, `:115` `git stash pop`: the script moves the working tree to a
  hard-coded baseline commit (`:17`) and back. The `cleanup` trap (`:66-88`)
  restores the original ref and the stash on every exit path and reports loudly
  when the stash cannot be popped (`:81-82`). This is a developer-run comparison
  script, documented as such in its header; the behaviour is the point, not a
  defect.
- **Accepted with reason — `2>/dev/null` on `cargo build`** (`:96`, `:124`).
  Unlike `:32` there is no `|| true`, so a failing build still aborts the script
  under `set -euo pipefail`. Only the diagnostics are lost, which is a usability
  cost rather than a masked failure. **Superseded** — the #2140 fix dropped both
  redirects, so a failing baseline build now says why.
- **Clean** — `:14` `PARQUET_FILE="${1:-}"` is only ever `-f`-tested (`:101`,
  `:129`) and printed; it reaches no command position.

### `scripts/fuzz-ci.sh` — 73 lines

**Accepted with reason — `MAX_TIME`.** `:15` `MAX_TIME="${1:-30}"` takes the
first argument with no numeric guard, and it reaches libFuzzer at `:59` as
`-max_total_time="${MAX_TIME}"`. It is not filed because the value is quoted and
arrives as a single `argv` element of `cargo fuzz run`; it never enters an
arithmetic context, a command position or an `eval`. A non-numeric value makes
libFuzzer reject its own flag, the branch at `:61-64` sets `FAILED=1`, and the
script exits `1` at `:72` — it fails loud. `0` means "unbounded", which is
libFuzzer's documented meaning and appropriate for a local soak. The script is
invoked from no workflow in this repository (README-documented local use), so
the argument is the operator's own. A `^[0-9]+$` guard would be a small
improvement in diagnostics, not a security fix, and was judged not to warrant an
issue on its own.

### `scripts/check-pr-summary-location.sh` — 35 lines

**Finding #2139.** `:22` runs `find … 2>/dev/null | sort` inside a process
substitution. `set -euo pipefail` (`:9`) does not check a process substitution's
exit status, and `sort` succeeds on empty input, so a `find` that never ran is
indistinguishable from one that found nothing — the script reaches `:35` and
prints ✅ with exit `0`. Reproduced at audit time by running the script in a tree
with no `docs/`: the ✅ line appeared and the exit status was `0`, both with and
without the `2>/dev/null` redirect. This is a **gate** (`quality.sh:28`), so the
silent pass is reported to the contributor as a clean check — the exact
"absence of a failure marker read as success" pattern the house standard names.

### `scripts/install-rust-toolchain.sh` — 126 lines

- **Finding #2126** (filed before this sweep, re-confirmed at `4f269d6`):
  `:52-53` read `RUST_TOOLCHAIN_MAX_ATTEMPTS` and `RUST_TOOLCHAIN_RETRY_DELAY`
  with no validation, and `:104` evaluates the first in the arithmetic context of
  `[[ "$attempt" -ge "$MAX_ATTEMPTS" ]]`.
- **Clean** — the argument path. `NAME_PATTERN` (`:57`) and `validate_name`
  (`:59-65`) allowlist every toolchain and component name before it reaches
  `rustup` (`:71`, `:82`), including the comma-separated form (`:74-85`). The
  composite action passes its inputs through `env:` only
  (`.github/actions/setup-rust/action.yml:49-58`), so no workflow value reaches
  the argument vector.
- **Clean** — the install is positively confirmed (`:116-119` runs
  `rustc --version` and `cargo --version` through the new toolchain) rather than
  assumed from a zero exit.

### `scripts/install-rustup.sh` — 151 lines

**Finding #2127** (filed before this sweep, re-confirmed at `4f269d6`): `:117`
`trap "rm -rf '$tmp_dir'" EXIT` interpolates the `mktemp -d` result (`:115`,
which honours `$TMPDIR`) into a string bash re-parses when the trap fires.
Everything else on this file is the #1911 path re-verified above.

### `scripts/benchmark-ci.sh` — 217 lines

Clean. The threshold — from `BENCHMARK_THRESHOLD` (`:32`) or `--threshold`
(`:74-81`) — is validated by `benchmark_threshold::require_valid` (`:101`) before
it is interpolated into a `bc` expression downstream; that guard is the
counter-example #1918 was filed for, and `tests/issue_1918_benchmark_threshold_validation.rs`
covers both the environment and the CLI route. The `source` at `:45` reads a
`BASH_SOURCE`-derived path (`:35`, `:38`) with a readability check first
(`:39-42`), so a caller cannot redirect it. `discover_benchmarks` (`:105-121`)
parses the repository's own `Cargo.toml` with `[[ =~ ]]`, never `eval`.

### `scripts/check-version-no-downgrade.sh` — 56 lines

Clean. Both versions are regex-validated against `^[0-9]+\.[0-9]+\.[0-9]+$`
(`:32-36`) before `cmp_triple` (`:42-48`) puts any component into a `[ -ne ]`
arithmetic test, and the single call site passes them quoted
(`.github/workflows/ci.yml:133`). Argument count is checked (`:27`); every
failure goes through `die` (`:22-25`) and exits non-zero.

### `scripts/doc-check.sh` — 21 lines

Clean. Two fixed `cargo` invocations (`:13`, `:19`), no inputs of any kind, and
both failures propagate under `set -euo pipefail` (`:2`).

### `scripts/benchmark_threshold.sh` — 29 lines

Clean. Three functions, each a pure `[[ =~ ]]` predicate (`:13-15`, `:19-21`,
`:24-29`). No `set` line by design — see the script table row.

### `quality.sh` — 69 lines

- **Accepted with reason** — `:5-8` sources `$HOME/.cargo/env` when it exists.
  The file is owned by the invoking user and is what rustup itself writes;
  anyone who can modify it can already run code as that user. Sourcing it is the
  documented way a non-login shell finds cargo.
- **Accepted with reason** — `:47` `cargo fmt --all` mutates the working tree.
  A quality gate that verifies should not normally write, but this one is the
  repository's documented **pre-commit** step and the formatting it applies is
  the formatting CI would otherwise reject. It is recorded here so the departure
  is deliberate and visible.
- **Clean** — `:30-36` documents why no dependency bump happens here (#1865),
  and `tests/issue_1865_quality_gate_no_dep_mutation.rs` enforces it against a
  stubbed `cargo`.

## Issues filed

- `#2139` — `check-pr-summary-location.sh` reports ✅ when its `find` never ran.
- `#2140` — `benchmark.sh` times a failed suite and prints it as a speed-up.
- `#2141` — `benchmark.sh` timings fail on macOS (`date +%s.%N`); a plain bug,
  not a security finding.

Re-confirmed, filed before this sweep: `#2126`, `#2127`.

## Related remediations (not sweep coverage)

Prior fixes touching this chunk, for context only. These do **not** count as a
sweep and never justify a non-null `last_swept`.

- `#1911` — the digest-verified rustup bootstrap that replaced
  `curl https://sh.rustup.rs | sh`.
- `#1912` — `--locked` and an explicit `--version` on the `cargo-fuzz` install,
  plus the `quality/cargo_install_pinning.sh` gate that keeps them there.
- `#1913` — the hash-pinned codespell requirements file.
- `#1918` — the shared threshold validation now in
  `scripts/benchmark_threshold.sh`.
- `#1755` / `#1898` — the committed `bash -n` and ShellCheck gates that make
  this chunk lintable at all.
- `#2072` — the canonical `runlib.sh` copy contract and its `family-sync`
  workflow.

## Verify this record

```bash
git diff 4f269d6b604eb480b9efe5267628d10484ea25b1..HEAD -- \
  scripts/ quality.sh benchmark.sh
```

An empty diff means this record still describes the current code. The fixes for
Issues #2126, #2127, #2139 and #2140 land on top of that baseline, so the first
non-empty diff is expected to be exactly them.
