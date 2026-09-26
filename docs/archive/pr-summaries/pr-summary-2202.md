## Summary

`./bump-deps.sh` exited 8 on every run (`could not pin js-sys back to 0.3.105`),
so dependency bumps were disabled. Closes #2202.

**Root cause:** the lockfile quarantine gate pinned each in-window package back
on its own with `cargo update -p name@new --precise old`. Lockstep releases
(wasm-bindgen / js-sys / web-sys / wasm-bindgen-futures, zerocopy /
zerocopy-derive) require each other with exact `=` versions. Pinning one of
them therefore conflicts with a partner that still needs the new version, and
`--precise` cannot be combined with `--recursive`.

**Fix:** the new `bump_deps::reapply_safe_lock_changes` never tries to pin a
single package back. It works like this:

1. Roll `Cargo.lock` back to its pre-bump state.
2. Resolve only what the bumped manifests strictly need (`cargo update --workspace`).
   If that alone needs an in-quarantine package, exit 8.
3. Re-apply each out-of-window change one package at a time with
   `cargo update -p name@old`. Roll back any step that fails or drags in an
   in-quarantine package; that change is deferred to a later run.

Lockstep partners move together or not at all, and the in-window versions stay
out.

```mermaid
flowchart TD
    A[cargo update] --> B{any lock change inside window?}
    B -- no --> OK[gate OK, lockfile untouched]
    B -- yes --> C[restore pre-bump Cargo.lock]
    C --> D[cargo update --workspace]
    D --> E{in-window package needed?}
    E -- yes --> X[exit 8, fail loud]
    E -- no --> F[for each out-of-window change]
    F --> G[cargo update -p name@old]
    G --> H{failed or drags in-window pkg?}
    H -- yes --> R[roll step back, defer]
    H -- no --> K[keep]
    R --> F
    K --> F
```

### What changed

- `bump-deps.sh` has the new `reapply_safe_lock_changes` (plus two helpers,
  `list_lock_removals` and `describe_lock_plan`), wired into phase 3a. The header
  and the `--help` exit-8 text are updated.
- `tests/bump_deps_test.sh` gains Tests 26–28, which run against a stub `cargo`.
- `AGENTS.md`'s "Dependency Bumps" section records the rollback approach and
  warns against going back to `--precise`.
- `Cargo.lock` carries the out-of-window refresh from a real run of the fixed
  script. The in-window zerocopy pair (0.8.59, 22h old) stays at 0.8.56.

## Evidence

A real `./bump-deps.sh < /dev/null` run exits **0**. Right now the in-window
lockstep pair is zerocopy/zerocopy-derive, and it is held back together:

```text
🛡️  Lockfile quarantine gate (window=24h)…
   🚧 zerocopy 0.8.56 → 0.8.59 (publish age 22h < 24h) — held back
   🚧 zerocopy-derive 0.8.56 → 0.8.59 (publish age 22h < 24h) — held back
   ✅ js-sys 0.3.105 → 0.3.106 re-applied
   ✅ wasm-bindgen 0.2.128 → 0.2.129 re-applied
   ✅ web-sys 0.3.105 → 0.3.106 re-applied
   …
   lockfile quarantine gate OK (held back=2, deferred=0)
✅ bump-deps: no bumps (quarantined=0, lock_pinned_back=2, audit_run=1)
```

`bash tests/bump_deps_test.sh < /dev/null` gives `Passed: 96, Failed: 0`.

## Test Plan

- [x] Test 26 (regression for #2202) checks that lockstep in-window releases
      stay held back. It also checks these outcomes:
      - A safe bump that drags in a young partner is deferred.
      - A safe independent bump and a safe lockstep pair are both kept.
      - A bump that pulls in a young new package is deferred.
      - A cargo failure is reported with its reason.
      - The final lockfile has nothing inside the window.
      Against the unfixed script it fails: the function does not exist.
- [x] Test 27 checks that a fully aged refresh is a no-op: no cargo calls, and
      the lockfile is byte-identical.
- [x] Test 28 checks that the run fails loud (non-zero, naming the package)
      when the bumped manifests cannot resolve without an in-quarantine package.
- [x] `quality/shellcheck.sh .` and `quality/bash_syntax.sh .` pass.
- [x] A real `./bump-deps.sh` run exits 0.
- [ ] `./quality.sh` passes (see the PR checks).

## Security self-check

- [x] No secrets or hidden files staged.
- [x] Cargo is invoked with argv arrays, not with strings built from input.
- [x] In-quarantine versions are never accepted. Any failure to hold them back
      exits 8.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
