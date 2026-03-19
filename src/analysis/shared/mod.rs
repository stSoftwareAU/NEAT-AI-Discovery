//! Shared types and structures used across analysis modules.
//!
//! Organised into focused submodules:
//! - [`timing`] — GPU/CPU timing types and the [`TimingCollector`]
//! - [`metadata`] — Analysis result structures and diagnostic types
//! - [`gpu_info`] — GPU adapter information and zero-copy configuration

pub mod gpu_info;
pub mod metadata;
pub mod timing;

// Re-export all public types for backward compatibility so that existing
// `use crate::analysis::shared::TypeName` imports continue to work.
pub use gpu_info::*;
pub use metadata::*;
pub use timing::*;
