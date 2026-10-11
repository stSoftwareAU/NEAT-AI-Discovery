# Threat model

Reporting and supply-chain controls are in [SECURITY.md](../SECURITY.md); the
FFI surface is documented in [docs/FFI_API.md](../docs/FFI_API.md), and past
security sweeps are recorded in [docs/audits/](../docs/audits/README.md). This
file is the short brief for the scanner.

## What this project does and where untrusted input enters

NEAT-AI-Discovery is a Rust `cdylib` that the Deno library NEAT-AI loads with
`Deno.dlopen`. It records neuron activations and errors to Parquet during
evolution, then analyses the recordings on the GPU (wgpu compute shaders) to
propose new synapses and neurons, or removals, for a creature. Every entry
point in `src/ffi/` takes one NUL-terminated JSON C string and returns a
heap-allocated JSON C string the caller frees with `free_discovery_result`.
Treat as untrusted:

- the input JSON to every `extern "C"` entry point, including the embedded
  creature: creatures are pulled from shared repositories pushed by other
  machines in the fleet;
- the C string pointers themselves (null, invalid UTF-8, a pointer handed back
  to `free_discovery_result` twice or one it never allocated);
- Parquet files read back from a discovery directory, and the directory paths
  (`tempDir`, `baseDir`, output paths) the caller names for merge, export and
  cleanup.

## Components that matter most / least

Most important: `src/ffi/` and `src/ffi_internal/` (pointer handling,
`catch_unwind` around every entry point, string ownership); the discovery-
directory guard and recursive deletion in `discovery_cleanup`; Parquet loading
and schema validation in `src/parquet_format/` and `src/record/`; GPU buffer
sizing and readback in `src/analysis/gpu/`; other `unsafe` blocks.

Lower priority: candidate-scoring heuristics whose worst case is a poor
proposal, `benches/`, `examples/`, and `src/debug/` diagnostics.

## How to exercise it

From `/src`: `cargo test --lib --tests --all-features -- --test-threads=2` runs
the suite; `tests/ffi/` drives the C ABI directly. Two cargo-fuzz targets live in
`fuzz/` (`cargo +nightly fuzz run fuzz_ffi_entry_points`), built in the image
when the optional fuzz step succeeded. There is no GPU: Mesa lavapipe provides a
software Vulkan adapter, so GPU paths run slowly, and tests guarded by
`skip_without_gpu!` skip if no adapter is found.

## How you rate severity

- Critical: memory corruption, double free or use-after-free reachable from the
  input JSON or a Parquet file; deleting or overwriting files outside a
  discovery directory.
- High: out-of-bounds reads or other undefined behaviour in `unsafe` code
  (including GPU buffer readback) without a demonstrated exploit; a panic that
  unwinds across the FFI boundary, which aborts the host Deno process.
- Medium: a caught panic, unbounded allocation, or hang on malformed creature
  JSON or Parquet (denial of service of one discovery run).
- Low: wrong or degraded proposals with no safety impact.

## Anything to leave alone

- The `unsafe extern "C"` entry points are unsound by contract when handed a
  dangling pointer; report missing checks only where the contract promises one
  (null and UTF-8 are checked).
- The absence of a CPU analysis path is by design: without a GPU adapter,
  `analyze_parallel` returns a failure response.
- Process-global state (cancellation flags, environment-variable configuration
  read at start-up) is trusted operator input.
