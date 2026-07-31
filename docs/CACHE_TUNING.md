# Cache Tuning Guide

This guide explains how the record cache works during the analysis phase, how
to tune it for your deployment, and how to diagnose cache-related performance
issues.

For the streaming recording API (which writes the Parquet files that the cache
reads), see [STREAMING_GUIDE.md](STREAMING_GUIDE.md).

---

## Overview

After recording, the analysis phase loads neuron records from a Parquet file.
The record cache sits between the analysis engine and the Parquet file,
providing fast access to neuron data. The cache automatically selects one of
three tiers based on the Parquet file size and available system memory.

```
Analysis Engine
     │
     ▼
┌─────────────┐
│ Record Cache │  ← Automatic tier selection
├─────────────┤
│ PreloadAll  │  Small files   — load everything into memory
│ LRU Cache   │  Medium files  — keep hot neurons, evict cold ones
│ Streaming   │  Large files   — load on-demand per neuron
└─────────────┘
     │
     ▼
  Parquet File
```

---

## Cache Tiers

### Tier 1 — PreloadAll

**When selected:** Estimated expanded file size < available memory / 4.

The entire Parquet file is loaded into memory upfront, grouped by neuron UUID.
All subsequent lookups are instant (no I/O).

| Aspect | Detail |
|--------|--------|
| **Speed** | Fastest — all data in memory |
| **Memory** | Highest — entire dataset resident |
| **I/O** | Single bulk read at startup |
| **Best for** | Small-to-medium creatures with ample RAM |

**How it works:**
1. Reads all Parquet row groups in one pass.
2. Groups records by neuron UUID into a `HashMap`.
3. Wraps each group in `Arc` for zero-copy sharing across threads.
4. Subsequent `get(neuron_uuid)` calls return the `Arc` directly — no I/O.

### Tier 2 — LRU Cache

**When selected:** Estimated expanded file size is between available memory / 4
and available memory.

Keeps frequently-accessed neurons in a bounded memory pool. When the pool is
full, the least-recently-used neuron's records are evicted to make room.

| Aspect | Detail |
|--------|--------|
| **Speed** | Fast for hot neurons; cold neurons require Parquet I/O |
| **Memory** | Bounded — capacity is half of available memory |
| **I/O** | On-demand per neuron (cache misses) |
| **Best for** | Medium-sized creatures or memory-constrained systems |

**How it works:**
1. On first access for a neuron, loads its records from Parquet.
2. Stores the records in a `HashMap` with an LRU timestamp.
3. When total cached bytes exceed the capacity, evicts the
   least-recently-used entry.
4. Subsequent accesses for the same neuron are served from memory (cache hit).

**Compressed variant (LZ4):** For even greater memory efficiency, the
`CompressedLruRecordCache` compresses records using LZ4 before caching.
Typical compression ratios of 2–4x allow the cache to hold significantly more
neurons in the same memory footprint. LZ4 decompression runs at approximately
4 GB/s on modern hardware, so the CPU overhead is minimal compared to the I/O
savings from fewer cache misses.

### Tier 3 — Streaming

**When selected:** Estimated expanded file size >= available memory.

Loads records on-demand per neuron without retaining them in a cache. Each
neuron access scans the Parquet file. This is the slowest mode but works on
any system regardless of RAM.

| Aspect | Detail |
|--------|--------|
| **Speed** | Slowest — full Parquet scan per neuron |
| **Memory** | Minimal — only the current neuron's records in memory |
| **I/O** | One Parquet scan per `get()` call |
| **Best for** | Very large datasets that exceed available memory |

**Additional streaming features** (when using `StreamingRecordCache` directly):
- **Block-based loading** from Parquet row groups
- **LRU block eviction** for bounded memory usage
- **Prefetch** — background loading of adjacent blocks

---

## Tier Selection Logic

The cache uses a simple heuristic based on the projected decoded size of the
Parquet file and available system memory. Since Issue #1869 the projection is
read from the Parquet **footer** — the exact decompressed row count and error
value count — with the old `file_size × 3` compression heuristic kept only as a
floor, because dictionary and RLE encodings routinely beat 3:1 on this schema.

The projection remains an estimate. The enforced bound is the cumulative decode
budget the reader charges per record (see
`NEAT_AI_DISCOVERY_MAX_PARQUET_DECODE_MB` in
[CONFIGURATION.md](CONFIGURATION.md)), which aborts a decode mid-flight rather
than discovering the overrun afterwards.

```
estimated_expanded = max(footer_rows × per_record_bytes + error_values × 4,
                         file_size_bytes × 3)

if estimated_expanded < available_memory / 4:
    → PreloadAll

elif estimated_expanded < available_memory:
    → LRU Cache (capacity = available_memory / 2)

else:
    → Streaming
```

### Examples

| File Size | Available RAM | Estimated Expanded | Tier Selected |
|-----------|---------------|-------------------|---------------|
| 10 MB | 8 GB | 30 MB | PreloadAll |
| 500 MB | 8 GB | 1.5 GB | PreloadAll |
| 1 GB | 8 GB | 3 GB | LRU Cache |
| 4 GB | 8 GB | 12 GB | Streaming |
| 100 MB | 1 GB | 300 MB | LRU Cache |
| 500 MB | 1 GB | 1.5 GB | Streaming |

---

## How `max_analysis_memory_mb` Interacts with Cache Selection

The `max_analysis_memory_mb` parameter in `analyze_parallel` sets a memory
budget for the entire analysis phase, not just the cache. However, cache
selection runs before the memory budget is checked. The interaction is:

1. **Cache tier is selected first** based on system memory and file size.
2. **Memory budget is checked** at two points during analysis:
   - Before GPU work submission
   - After Parquet loading
3. **If the budget is exceeded**, analysis returns early with
   `memory_budget_exceeded: true` and whatever candidates have been found so
   far.

### Practical guidance

- **If analysis frequently hits the memory budget**, consider reducing the
  Parquet file size (record fewer training samples) or increasing the budget.
- **The cache tier adapts to system memory**, so `max_analysis_memory_mb` does
  not directly control which tier is used.
- **On memory-constrained systems** (e.g., 4 GB RAM), set
  `max_analysis_memory_mb` to 50–75% of available RAM to leave headroom for
  V8 and the OS.

---

## Environment Variables

The cache and streaming knobs (`NEAT_AI_DISCOVERY_PRELOAD_ALL`,
`NEAT_AI_DISCOVERY_MAX_CACHED_BLOCKS`, `NEAT_AI_DISCOVERY_PREFETCH_DEPTH`,
`NEAT_AI_DISCOVERY_BLOCK_SIZE`) — with their defaults and valid ranges — are
documented in the single authoritative reference,
[docs/CONFIGURATION.md § Streaming & Parquet](CONFIGURATION.md#streaming--parquet).

### Forcing a specific tier

- **Force PreloadAll:** Set `NEAT_AI_DISCOVERY_PRELOAD_ALL=1`. This bypasses
  the automatic tier selection and loads everything into memory. Only use this
  when you are confident the dataset fits comfortably in RAM.
- **There is no environment variable to force LRU or Streaming.** The
  automatic selection handles these cases based on available memory.

---

## Diagnosing Cache-Related Performance Issues

### Symptom: Analysis is slow despite having enough RAM

**Possible cause:** The cache selected Streaming or LRU when PreloadAll would
have been faster.

**Diagnosis:**
1. Enable verbose logging: `NEAT_AI_DISCOVERY_VERBOSE=1`
2. Look for the log line: `tiered loading strategy selected`
3. Check which strategy was chosen and whether memory estimates are accurate.

**Fix:** If the system has ample free RAM, set `NEAT_AI_DISCOVERY_PRELOAD_ALL=1`
to force PreloadAll mode.

### Symptom: Out of memory (exit code 137 / OOM killer)

**Possible cause:** PreloadAll was selected for a dataset that is too large for
the available memory, or the memory budget was set too high.

**Diagnosis:**
1. Check the Parquet file size: `ls -lh discovery_data.parquet`
2. Estimate the expanded size — the footer-derived projection, floored at
   `file_size × 3` (Issue #1869)
3. Compare against available RAM: `free -h` (Linux) or Activity Monitor (macOS)

**Fix:**
- Reduce the number of training samples recorded.
- Lower `max_analysis_memory_mb` to cap Rust-side memory usage.
- Ensure other processes are not consuming excessive memory.

### Symptom: High cache miss rate in LRU mode

**Possible cause:** The LRU cache capacity is too small for the working set.

**Diagnosis:**
1. Enable verbose logging: `NEAT_AI_DISCOVERY_VERBOSE=1`
2. Look for eviction count in cache statistics logs.
3. A high eviction count relative to cache hits indicates thrashing.

**Fix:**
- Free memory from other processes to increase available RAM (the LRU
  capacity scales with available memory at startup).
- Record fewer training samples to reduce the dataset size.

### Symptom: Analysis returns `memory_budget_exceeded: true`

**Possible cause:** The memory budget in `max_analysis_memory_mb` is too low
for the dataset size.

**Diagnosis:**
1. Check the `memory_budget_exceeded` field in the analysis output.
2. Use `discovery_memory_usage_bytes()` to monitor Rust-side memory usage
   during analysis.

**Fix:**
- Increase `max_analysis_memory_mb`.
- Reduce the number of training samples.
- Analysis returns partial results when the budget is exceeded — coverage
  improves over repeated runs.

### Symptom: Parquet file is missing or analysis fails to load records

**Possible cause:** The Parquet file was deleted while analysis was still
reading from it.

**Diagnosis:**
1. Check whether `is_analysis_active()` returns `1` — if so, analysis is
   still in flight.
2. On Unix, deleting the file path while a handle is open does not destroy
   the data (the inode stays alive). However, recreation at the same path
   will not contain the original data.

**Fix:**
- Follow the recommended shutdown sequence documented in
  [docs/FFI_API.md](FFI_API.md#analysis-lifecycle-guard-issue-1048) — cancel, wait
  for the FFI call to return, poll `is_analysis_active()` until idle, then delete
  the temp directory.

---

## Example Configurations

### Development (macOS, 16 GB RAM)

No configuration needed. The defaults work well:
- Small-to-medium datasets use PreloadAll (fastest).
- The LRU cache handles larger datasets automatically.

### Production Server (Linux, 8 GB RAM, shared with NEAT-AI)

```bash
# Cap Rust-side memory at 4 GB (leave 4 GB for V8 and the OS)
# Set via analyze_parallel input: max_analysis_memory_mb: 4096

# If datasets are consistently large, enable streaming prefetch
export NEAT_AI_DISCOVERY_PREFETCH_DEPTH=2

# Optionally limit cached blocks for predictable memory usage
export NEAT_AI_DISCOVERY_MAX_CACHED_BLOCKS=50
```

### Memory-Constrained (4 GB RAM)

```bash
# Conservative memory budget
# Set via analyze_parallel input: max_analysis_memory_mb: 2048

# Reduce block size for lower per-block memory usage
export NEAT_AI_DISCOVERY_BLOCK_SIZE=5000

# Limit prefetch to reduce peak memory
export NEAT_AI_DISCOVERY_PREFETCH_DEPTH=1
```

### Large Dataset (multi-GB Parquet files)

```bash
# Force streaming mode if automatic detection is not aggressive enough
# The cache will automatically select Streaming for files that exceed RAM,
# but you can disable preload explicitly:
export NEAT_AI_DISCOVERY_PRELOAD_ALL=0

# Increase prefetch depth for sequential access patterns
export NEAT_AI_DISCOVERY_PREFETCH_DEPTH=4

# Increase block size for fewer, larger reads
export NEAT_AI_DISCOVERY_BLOCK_SIZE=50000
```

---

## Related Documentation

- [STREAMING_GUIDE.md](STREAMING_GUIDE.md) — Streaming recording API guide
- [FFI_API.md](FFI_API.md) — Full FFI API reference and JSON schemas
- [GPU_GUIDE.md](GPU_GUIDE.md) — GPU performance tuning and troubleshooting
- [README.md](../README.md) — Project overview and quick start
