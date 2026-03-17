//! Discovery recording and parquet I/O integration tests.

#[path = "../common/mod.rs"]
mod common;

mod issue_193_streaming_parquet;
mod issue_215_tiered_parquet_loader;
mod issue_227_discovery_history;
mod issue_420_streaming_memory;
mod issue_481_record_loading_helper;
mod issue_493_creature_record_loading;
mod issue_648_deadline_aware_parquet_loading;
mod parquet_read_limit_ordering;
