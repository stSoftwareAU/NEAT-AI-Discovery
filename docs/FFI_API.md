# 🔌 FFI API Reference

This document contains the detailed FFI API reference, JSON interface specifications,
and streaming recording API for NEAT-AI-Discovery. For a high-level overview, see
[README.md](../README.md).

---

## 📦 Exported Symbols

The library exposes a Deno FFI-friendly symbol set. The authoritative list of exported
symbols lives in `src/lib.rs` as `#[no_mangle] pub extern "C"` functions.

The most commonly used entry points are:

- **GPU probe**: `check_gpu_available()` (returns JSON)
- **Version probe**: `get_library_version()` (returns JSON)
- **Recording**:
  - Streaming: `start_discovery_session`, `append_discovery_records`, `finish_discovery_session`, `cancel_discovery_session`
  - Single-call: `record_discovery` (avoid for large runs; prefer streaming to prevent JS/V8 string limits)
- **Analysis**: `rank_focus_neurons`, `analyze_parallel`
- **Utilities**: `merge_discovery_parquet`, `read_discovery_records_ffi`, `export_visualisation_snapshot`
- **Memory management**: `free_discovery_result`

---

## 🖥️ Checking for a Usable GPU

Discovery **requires a GPU** — there is no CPU fallback. On machines without a
suitable GPU, controllers must disable discovery entirely.

- **Symbol**: `check_gpu_available`
- **Input**: no arguments
- **Output**: JSON string:

  ```json
  {
    "success": true,
    "gpuAvailable": true,
    "reason": null
  }
  ```

  When GPU is unavailable:

  ```json
  {
    "success": true,
    "gpuAvailable": false,
    "reason": "No GPU adapter found. Discovery disabled on this machine..."
  }
  ```

- When `"gpuAvailable"` is `false`, controllers should treat discovery as disabled.
- When `"gpuAvailable"` is `true`, controllers may safely schedule discovery jobs.

### 🌏 Platform-specific GPU behaviour

- **macOS**: GPU (Metal) should always be available. If `gpuAvailable` is `false`,
  this is treated as an error (`success: false`).
- **Linux**: GPU may not be available on headless servers without GPU hardware
  or without proper permissions to access `/dev/dri` devices. If `gpuAvailable`
  is `false`, this is **not** an error (`success: true`) — discovery is simply
  disabled on that machine.

  **Note:** On Linux, the library only probes the Vulkan backend (not OpenGL/EGL)
  to avoid panics from EGL initialisation errors on systems without proper GPU
  drivers.

---

## 📋 JSON Interface

### Input Format (record_discovery)

```json
{
  "creature": {
    "neurons": [
      {
        "uuid": "hidden-1",
        "type": "hidden",
        "squash": "TANH",
        "bias": 0.0
      }
    ],
    "synapses": [
      {
        "from_uuid": "input-0",
        "to_uuid": "hidden-1",
        "weight": 0.5
      }
    ],
    "input": 20,
    "output": 2
  },
  "training_data": [
    {"input": [0.1, 0.2, "..."], "output": [0.5, 0.3]},
    "..."
  ],
  "temp_dir": ".discovery/abc123_456789",
  "binary_file_path": "/path/to/binary.bin",
  "record_indices": [0, 5, 10, "..."],
  "timeout_seconds": 300
}
```

### Output Format

Success:
```json
{
  "success": true,
  "temp_dir": ".discovery/abc123_456789",
  "file": "discovery_data.parquet"
}
```

Error:
```json
{
  "success": false,
  "error": "Error message here"
}
```

---

## 🔄 Streaming Recording API (v0.2.8+)

The streaming API solves the JavaScript "Invalid string length" error that occurs when
trying to serialise large datasets (6+ minutes of recording) into a single JSON string.
Instead of one monolithic `record_discovery` call, data is streamed incrementally.

**Why this matters**: JavaScript/V8 has a maximum string length (~2^28 chars). When
TypeScript accumulated 6+ minutes of discovery data and tried to JSON.stringify it all
at once for the FFI call, it hit this limit. The streaming API keeps each FFI call small.

### 🛠️ Usage Pattern

```mermaid
sequenceDiagram
    participant TS as 🟦 TypeScript
    participant RS as 🦀 Rust Library

    TS->>RS: 1. start_discovery_session()
    RS-->>TS: sessionId

    loop Collect data batches
        Note over TS: Collect records<br/>(estimate size)
        TS->>RS: 2. append_discovery_records()
        Note over RS: Writes batch<br/>to Parquet
        RS-->>TS: recordsWritten
    end

    TS->>RS: 3. finish_discovery_session()
    RS-->>TS: tempDir, file, totalRecords
```

### ⚙️ FFI Functions

**`start_discovery_session`** — Start a new recording session

Input:
```json
{
  "creature": { "neurons": ["..."], "synapses": ["..."], "input": 20, "output": 2 },
  "tempDir": ".discovery/abc123_456789"
}
```

Output:
```json
{
  "success": true,
  "sessionId": "550e8400-e29b-41d4-a716-446655440000"
}
```

**`append_discovery_records`** — Append records to an existing session

Input:
```json
{
  "sessionId": "550e8400-e29b-41d4-a716-446655440000",
  "observations": [
    {
      "obsIndex": 0,
      "neuronData": [
        { "neuronUuid": "hidden-1", "activation": 0.5, "value": 0.4, "errors": [0.1] }
      ],
      "inputs": [0.1, 0.2, 0.3]
    }
  ]
}
```

Output:
```json
{
  "success": true,
  "recordsWritten": 42
}
```

**`finish_discovery_session`** — Finalise and close the Parquet file

Input:
```json
{
  "sessionId": "550e8400-e29b-41d4-a716-446655440000"
}
```

Output:
```json
{
  "success": true,
  "tempDir": ".discovery/abc123_456789",
  "file": "discovery_data.parquet",
  "totalRecords": 12345
}
```

**`cancel_discovery_session`** — Cancel a session (cleanup without finalising)

Input:
```json
{
  "sessionId": "550e8400-e29b-41d4-a716-446655440000"
}
```

Output:
```json
{
  "success": true
}
```

### 📐 Size Estimation for TypeScript

To decide when to flush, estimate the JSON size before serialising:

```typescript
// Rough estimate: ~200 bytes per neuron record + input array
const estimatedBytes = observations.length * (
  200 * creature.neurons.length +
  4 * creature.input
);

// Flush when approaching 50MB (well under JS string limits)
const FLUSH_THRESHOLD = 50 * 1024 * 1024;
if (estimatedBytes > FLUSH_THRESHOLD) {
  await appendDiscoveryRecords(sessionId, observations);
  observations = []; // Reset batch
}
```

### ✅ Benefits

- **No string length limits**: Each batch is small enough to serialise
- **Unlimited sample sizes**: Can record for hours without memory issues
- **Fail-safe**: If process crashes, already-written data is preserved in the Parquet file
- **Reduced memory pressure**: TypeScript can discard batches after flushing

---

## 🚨 Critical Requirements

### ⚛️ Atomic Record Writes

**For each discovery record, all data (observations, activations, errors) MUST come from the same training record.** This is essential because:
- The analysis phase matches records by index
- When evaluating synapse candidates, records must align
- If observations, activations, and errors don't line up, analysis will be incorrect

**Implementation Requirements:**
- **Atomic writes**: For each training record, activate creature, collect ALL neuron data, then write ALL neuron rows together
- **Parallelisation allowed**: Since training dataset is already randomised, we CAN process different training records in parallel
- **Per-record atomicity**: Each parallel task must process one complete training record
- **Cross-neuron alignment**: Records with the same `obs_index` across different neurons correspond to the same training record
- **No mixing**: Never mix data from different training records within a single discovery record write
- **Matching by obs_index**: TypeScript matches records across neurons by `obs_index` (not by array position)

### ➡️ Forward-only Activation Order (no feedback)

Discovery assumes **forward-only** networks (no recurrent feedback). This is critical for both recording and for applying discovery candidates:

- **Evaluation order matters**: A neuron may only read activations from **earlier** neurons in the creature's evaluation order.
- **Synapse direction constraint**: For feed-forward creatures, synapses must point from an **earlier** neuron to a **later** neuron.
- **Discovered neurons must be inserted, not appended**: When applying an add-neuron candidate, the new neuron must be inserted at the correct index.
- **No "remembering" across samples**: Discovery explicitly does **not** support recurrent connections.

---

## 📁 File Format

### Single Parquet File

File location: `.discovery/{creature_uuid}_{random}/discovery_data.parquet`

Schema:
- `obs_index: u32` — Observation index (training record index) for ordering
- `neuron_uuid: string` — Neuron identifier
- `value: f32` — Neuron value (optional, can be null)
- `activation: f32` — Neuron activation
- `errors: list<f32>` — Array of error values

**Benefits:**
- Single file handle (eliminates small-file problems)
- Columnar format excellent for filtering by neuron during analysis
- Viewable with standard tools for debugging

### 🔍 Debugging Parquet Files

**Python:**
```python
import pandas as pd
df = pd.read_parquet('discovery_data.parquet')
print(df.head())
```

**DuckDB:**
```sql
SELECT * FROM 'discovery_data.parquet' LIMIT 10;
```

**Command-line:**
- `parquet-tools` (Java-based)
- `parquet-cli` (Rust-based)
