## Summary

Replace the broad `.*` dotfile exclusion in `.gitignore` with specific exclusions for system and editor files. This ensures Rust configuration files (`.rustfmt.toml`, `.clippy.toml`, `.cargo/config.toml`) will be tracked if created, while system/editor files (`.DS_Store`, `.env`, `.idea/`, `.vscode/`) remain excluded. Closes #679.

## Evidence

Verified with `git check-ignore -v`:

- `.rustfmt.toml` — **not ignored** (would be tracked)
- `.clippy.toml` — **not ignored** (would be tracked)
- `.cargo/config.toml` — **not ignored** (would be tracked)
- `.gitignore` — **not ignored** (tracked as before)
- `.gitattributes` — **not ignored** (tracked as before)
- `.github/workflows/ci.yml` — **not ignored** (tracked as before)
- `.DS_Store` — **ignored** (system file)
- `.env`, `.env.local`, `.env.production` — **ignored** (secrets)
- `.idea/` — **ignored** (JetBrains editor)
- `.vscode/` — **ignored** (VS Code editor)
- `.*.swp`, `.*.swo` — **ignored** (vim swap files)

No Rust code changes; `./quality.sh` passes cleanly.

## Test Plan

- Verified `.gitignore` rules using `git check-ignore -v` for all relevant file patterns
- Confirmed Rust config files are not excluded
- Confirmed system/editor files remain excluded
- Ran `./quality.sh` — all checks pass
