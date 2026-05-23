//! End-to-end cost-compatibility integration tests (Issue #1246).
//!
//! Discovery is **cost-agnostic by construction** — per-neuron errors arrive
//! pre-computed in `DiscoverRecord.errors[]` via `Neuron.record()` and the
//! library never reads the cost name. These tests exercise the full
//! `record_discovery → analyze_parallel` pipeline against fixtures shaped
//! like the residual distribution of every built-in NEAT-AI cost, so a
//! regression in cost-agnostic behaviour fails CI rather than surfacing as
//! silent production drift.

#[path = "../common/mod.rs"]
mod common;

mod end_to_end;
