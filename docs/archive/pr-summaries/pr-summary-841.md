## Summary

Unified and de-duplicated user documentation with README.md as the central hub.
Closes #841.

### Changes

- **AGENTS.md**: Replaced duplicated Project Overview/Sole Mission (§1), FFI
  Contract/Key Symbols (§7), and GPU Requirement/System Requirements (§8) with
  concise references to README.md and docs/FFI_API.md. Replaced Further Reading
  list with reference to README.md's comprehensive documentation table.
- **CONTRIBUTING.md**: Replaced duplicated Quality Gate steps, CI Pipeline list,
  Code Style/Coding Principles, Testing Guidelines (including code examples),
  and Project Structure layout with references to the authoritative versions in
  AGENTS.md. Replaced Further Reading with reference to README.md's
  documentation table.
- **README.md**: Added docs/BENCHMARKS.md to the Additional Documentation table
  so every documentation file is now reachable from README.md.

### Content ownership after this change

| Content | Authoritative location |
|---------|----------------------|
| Project overview and mission | README.md |
| FFI API reference | docs/FFI_API.md |
| GPU requirements and tuning | README.md + docs/GPU_GUIDE.md |
| Quality gate and CI pipeline | AGENTS.md §5 |
| Coding conventions and style | AGENTS.md §3 |
| Testing philosophy | AGENTS.md §4 |
| Source layout / architecture | AGENTS.md §2 |
| Discovery types reference | docs/DISCOVERY_TYPES.md |
| Benchmark tracking | docs/BENCHMARKS.md |

## Evidence

- No content is duplicated across documentation files
- README.md links (directly or transitively) to every documentation file
- All cross-references verified accurate
- `quality.sh` passes cleanly

## Test Plan

- Documentation-only change — no code or tests modified
- Verified all markdown cross-reference anchors resolve to existing headings
- Ran `./quality.sh` — all checks passed
