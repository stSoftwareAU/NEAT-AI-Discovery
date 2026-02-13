//! Recommendation modules — candidate recommendation engines.
//!
//! This subdirectory groups modules that proactively recommend mutation
//! candidates based on analysis of recorded discovery data.

pub mod activation_recommendation;
pub mod epistatic;
pub mod gradient_discovery;
pub mod multi_hop;
pub mod output_bias_drift;
pub mod sample_weighted;
