//! Type definitions for discovery records

use serde::{Deserialize, Serialize};

/// Represents a single discovery record for a neuron
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DiscoverRecord {
    /// Observation index (training record index)
    pub obs_index: u32,
    /// Neuron identity string — may be an RFC 4122 UUID or a stringified
    /// integer from TypeScript runtime `neuron.id` (Issue #950).
    pub neuron_uuid: String,
    /// Neuron value (optional, can be None)
    pub value: Option<f32>,
    /// Neuron activation
    pub activation: f32,
    /// Array of error values
    pub errors: Vec<f32>,
}

impl DiscoverRecord {
    /// Create a new discovery record
    pub fn new(
        obs_index: u32,
        neuron_uuid: String,
        value: Option<f32>,
        activation: f32,
        errors: Vec<f32>,
    ) -> Self {
        Self {
            obs_index,
            neuron_uuid,
            value,
            activation,
            errors,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discover_record_creation() {
        let record = DiscoverRecord::new(
            0,
            "hidden-1".to_string(),
            Some(0.5),
            0.7,
            vec![0.1, 0.2, -0.1],
        );

        assert_eq!(record.obs_index, 0);
        assert_eq!(record.neuron_uuid, "hidden-1");
        assert_eq!(record.value, Some(0.5));
        assert_eq!(record.activation, 0.7);
        assert_eq!(record.errors, vec![0.1, 0.2, -0.1]);
    }

    #[test]
    fn test_discover_record_without_value() {
        let record = DiscoverRecord::new(1, "output-0".to_string(), None, 0.9, vec![0.05]);

        assert_eq!(record.obs_index, 1);
        assert_eq!(record.value, None);
    }
}
