//! Issue #1032 — Verify documentation accuracy after recent changes.
//!
//! These tests confirm that modules, FFI symbols, and configuration accessors
//! referenced in the project documentation actually exist in the codebase.

// ============================================================================
// Module existence — verify documented source modules are accessible
// ============================================================================

#[test]
fn documented_temperature_module_exists() {
    // constants::temperature module documented in AGENTS.md
    use neat_ai_discovery::analysis::constants::temperature::CoolingSchedule;
    use neat_ai_discovery::analysis::constants::temperature::compute_scheduled_temperature;

    // Verify the function is callable with valid inputs.
    let result = compute_scheduled_temperature(CoolingSchedule::Linear, 1.0, 10, 100);
    assert!(result > 0.0, "Temperature should be positive, got {result}");
    assert!(
        result <= 1.0,
        "Linear cooling from 1.0 at generation 10/100 should be <= 1.0"
    );
}

#[test]
fn documented_adaptive_proposal_module_exists() {
    // synapse::adaptive_proposal module documented in AGENTS.md
    use neat_ai_discovery::analysis::synapse::adaptive_proposal::AcceptanceTracker;

    let tracker = AcceptanceTracker::new();
    // Verify it initialises without panic and returns a valid sigma.
    let sigma = tracker.sigma_for("test_type");
    assert!(sigma > 0.0, "Initial sigma should be positive, got {sigma}");
}

// ============================================================================
// FFI symbol existence — verify documented FFI functions are accessible
// ============================================================================

#[test]
fn documented_ffi_memory_usage_exists() {
    // discovery_memory_usage_bytes documented in FFI_API.md
    let usage = neat_ai_discovery::ffi::discovery_memory_usage_bytes();
    // Just verify it returns without panic; value is non-deterministic.
    let _ = usage;
}

// ============================================================================
// Configuration accessor existence — verify documented env var accessors exist
// ============================================================================

#[test]
fn documented_mh_temperature_config_exists() {
    // NEAT_AI_DISCOVERY_MH_TEMPERATURE documented in README.md and AGENTS.md
    use neat_ai_discovery::config::mh_temperature;

    // The function must exist and return Option<f32>.
    // We cannot assert None because the env var might be set in the test environment.
    let result = mh_temperature();
    let _: Option<f32> = result;
}

// ============================================================================
// Temperature scheduling — verify cooling schedules work as documented
// ============================================================================

#[test]
fn temperature_linear_cooling_produces_decreasing_values() {
    use neat_ai_discovery::analysis::constants::temperature::CoolingSchedule;
    use neat_ai_discovery::analysis::constants::temperature::compute_scheduled_temperature;

    let max_generation = 100;
    let mut prev = f64::MAX;
    for generation in [0, 25, 50, 75, 100] {
        let temp =
            compute_scheduled_temperature(CoolingSchedule::Linear, 1.0, generation, max_generation);
        assert!(
            f64::from(temp) <= prev,
            "Linear cooling should be non-increasing: generation {generation} temp {temp} > prev {prev}"
        );
        prev = f64::from(temp);
    }
}

#[test]
fn temperature_exponential_cooling_produces_decreasing_values() {
    use neat_ai_discovery::analysis::constants::temperature::CoolingSchedule;
    use neat_ai_discovery::analysis::constants::temperature::compute_scheduled_temperature;

    let max_generation = 100;
    let mut prev = f64::MAX;
    for generation in [0, 25, 50, 75, 100] {
        let temp = compute_scheduled_temperature(
            CoolingSchedule::Exponential,
            1.0,
            generation,
            max_generation,
        );
        assert!(
            f64::from(temp) <= prev,
            "Exponential cooling should be non-increasing: generation {generation} temp {temp} > prev {prev}"
        );
        prev = f64::from(temp);
    }
}
