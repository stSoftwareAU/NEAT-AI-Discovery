//! Temperature scheduling for exploration-exploitation balance (Issue #1020).
//!
//! Provides cooling schedule functions that compute a temperature value from
//! an evolutionary generation count. The temperature controls how aggressively
//! the candidate selection pipeline filters marginal candidates:
//!
//! - **High temperature** (early evolution) → lower effective thresholds → accept
//!   more candidates → exploration
//! - **Low temperature** (late evolution) → higher effective thresholds → only
//!   accept strong candidates → exploitation
//!
//! A default temperature of 1.0 preserves existing behaviour (backward compatible).

// =============================================================================
// Cooling Schedule Types
// =============================================================================

/// Available cooling schedule strategies.
///
/// Each schedule maps a generation count to a temperature in (0, `initial_temp`].
/// The `compute_temperature` function dispatches to the appropriate formula.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CoolingSchedule {
    /// Linear decay: `T = initial × max(min_temp, 1 - generation / total_generations)`.
    ///
    /// Temperature decreases uniformly over the run. Simple and predictable,
    /// but may cool too quickly in early generations when exploration is most
    /// valuable.
    Linear,

    /// Exponential decay: `T = initial × max(min_temp, decay_rate ^ generation)`.
    ///
    /// Temperature decreases rapidly at first, then slows. This preserves
    /// higher temperatures during early exploration and converges smoothly
    /// toward exploitation.
    Exponential,
}

// =============================================================================
// Constants
// =============================================================================

/// Default temperature value that preserves existing behaviour.
///
/// At temperature 1.0, all scaling factors are identity operations:
/// - `effective_threshold = threshold / 1.0 = threshold`
/// - `effective_ratio = ratio / 1.0 = ratio`
/// - MH temperature is unscaled
///
/// ## Valid Range
/// Must be exactly 1.0 for backward compatibility.
pub const DEFAULT_TEMPERATURE: f32 = 1.0;

/// Minimum temperature floor to prevent division by zero or near-zero.
///
/// Even at the coldest point of a cooling schedule, temperature never drops
/// below this value. This prevents infinite effective thresholds and ensures
/// at least some candidates can still be accepted.
///
/// ## Valid Range
/// Must be > 0.0 and < 1.0.
pub const MIN_TEMPERATURE: f32 = 0.01;

/// Maximum temperature ceiling to prevent excessively permissive acceptance.
///
/// ## Valid Range
/// Must be >= 1.0.
pub const MAX_TEMPERATURE: f32 = 5.0;

/// Default decay rate for exponential cooling schedule.
///
/// With decay rate 0.995 and 1000 generations:
/// - Generation 0: T = 1.0
/// - Generation 100: T ≈ 0.606
/// - Generation 500: T ≈ 0.082
/// - Generation 1000: T ≈ 0.0067 (clamped to `MIN_TEMPERATURE`)
///
/// ## Valid Range
/// Must be in (0.0, 1.0). Values near 1.0 cool slowly; values near 0.0
/// cool almost immediately.
pub const DEFAULT_EXPONENTIAL_DECAY_RATE: f32 = 0.995;

// =============================================================================
// Cooling Schedule Functions
// =============================================================================

/// Compute the temperature for a given generation using the specified cooling schedule.
///
/// # Arguments
///
/// * `schedule` — The cooling strategy to use.
/// * `initial_temperature` — Starting temperature (typically 1.0 or higher).
/// * `generation` — Current evolutionary generation (0-based).
/// * `total_generations` — Total expected generations (used by linear schedule).
///
/// # Returns
///
/// A temperature value clamped to [`MIN_TEMPERATURE`, `MAX_TEMPERATURE`].
pub fn compute_scheduled_temperature(
    schedule: CoolingSchedule,
    initial_temperature: f32,
    generation: u32,
    total_generations: u32,
) -> f32 {
    #[allow(clippy::cast_precision_loss)] // Intentional: generation counts fit comfortably in f32
    let raw = match schedule {
        CoolingSchedule::Linear => {
            if total_generations == 0 {
                initial_temperature
            } else {
                let progress = generation as f32 / total_generations as f32;
                initial_temperature * (1.0 - progress)
            }
        }
        CoolingSchedule::Exponential => {
            initial_temperature * DEFAULT_EXPONENTIAL_DECAY_RATE.powi(generation.cast_signed())
        }
    };

    raw.clamp(MIN_TEMPERATURE, MAX_TEMPERATURE)
}

/// Apply temperature scaling to an acceptance threshold.
///
/// Higher temperature lowers the effective threshold (more permissive).
/// Lower temperature raises the effective threshold (more selective).
/// Temperature 1.0 returns the threshold unchanged.
///
/// Formula: `effective_threshold = base_threshold / temperature`
#[inline]
pub fn scale_threshold_by_temperature(base_threshold: f32, temperature: f32) -> f32 {
    let clamped_temp = temperature.clamp(MIN_TEMPERATURE, MAX_TEMPERATURE);
    base_threshold / clamped_temp
}

/// Apply temperature scaling to a minimum improved ratio.
///
/// Higher temperature lowers the effective ratio (more permissive).
/// Lower temperature raises the effective ratio (more selective).
/// Temperature 1.0 returns the ratio unchanged.
///
/// Formula: `effective_ratio = base_ratio / temperature`, clamped to [0.0, 1.0].
#[inline]
pub fn scale_ratio_by_temperature(base_ratio: f32, temperature: f32) -> f32 {
    let clamped_temp = temperature.clamp(MIN_TEMPERATURE, MAX_TEMPERATURE);
    (base_ratio / clamped_temp).clamp(0.0, 1.0)
}

/// Apply temperature scaling to the Metropolis-Hastings temperature parameter.
///
/// The input temperature from the evolutionary schedule scales the base MH
/// temperature, controlling how willingly the system accepts marginal candidates.
///
/// Formula: `effective_mh_temp = base_mh_temp × temperature`
///
/// - High schedule temperature → higher effective MH temp → more acceptance
/// - Low schedule temperature → lower effective MH temp → less acceptance
#[inline]
pub fn scale_mh_temperature(base_mh_temp: f32, temperature: f32) -> f32 {
    let clamped_temp = temperature.clamp(MIN_TEMPERATURE, MAX_TEMPERATURE);
    (base_mh_temp * clamped_temp).max(f32::MIN_POSITIVE)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_temperature_is_one() {
        assert!((DEFAULT_TEMPERATURE - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn min_temperature_is_positive() {
        let min = MIN_TEMPERATURE;
        assert!(min > 0.0);
        assert!(min < 1.0);
    }

    #[test]
    fn linear_schedule_at_generation_zero_returns_initial() {
        let temp = compute_scheduled_temperature(CoolingSchedule::Linear, 1.0, 0, 1000);
        assert!((temp - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn linear_schedule_at_midpoint_returns_half() {
        let temp = compute_scheduled_temperature(CoolingSchedule::Linear, 1.0, 500, 1000);
        assert!((temp - 0.5).abs() < 0.01);
    }

    #[test]
    fn linear_schedule_at_end_returns_min_temperature() {
        let temp = compute_scheduled_temperature(CoolingSchedule::Linear, 1.0, 1000, 1000);
        assert!((temp - MIN_TEMPERATURE).abs() < f32::EPSILON);
    }

    #[test]
    fn exponential_schedule_at_generation_zero_returns_initial() {
        let temp = compute_scheduled_temperature(CoolingSchedule::Exponential, 1.0, 0, 1000);
        assert!((temp - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn exponential_schedule_decays_monotonically() {
        let t0 = compute_scheduled_temperature(CoolingSchedule::Exponential, 1.0, 0, 1000);
        let t100 = compute_scheduled_temperature(CoolingSchedule::Exponential, 1.0, 100, 1000);
        let t500 = compute_scheduled_temperature(CoolingSchedule::Exponential, 1.0, 500, 1000);
        assert!(t0 > t100);
        assert!(t100 > t500);
    }

    #[test]
    fn temperature_never_below_minimum() {
        let temp = compute_scheduled_temperature(CoolingSchedule::Linear, 1.0, 10000, 1000);
        assert!(temp >= MIN_TEMPERATURE);

        let temp_exp =
            compute_scheduled_temperature(CoolingSchedule::Exponential, 1.0, 10000, 1000);
        assert!(temp_exp >= MIN_TEMPERATURE);
    }

    #[test]
    fn temperature_never_above_maximum() {
        let temp = compute_scheduled_temperature(CoolingSchedule::Linear, 10.0, 0, 1000);
        assert!(temp <= MAX_TEMPERATURE);
    }

    #[test]
    fn scale_threshold_at_default_temperature_is_identity() {
        let threshold = 0.05;
        let scaled = scale_threshold_by_temperature(threshold, DEFAULT_TEMPERATURE);
        assert!((scaled - threshold).abs() < f32::EPSILON);
    }

    #[test]
    fn high_temperature_lowers_effective_threshold() {
        let threshold = 0.05;
        let scaled = scale_threshold_by_temperature(threshold, 2.0);
        assert!(scaled < threshold, "High temp should lower threshold");
    }

    #[test]
    fn low_temperature_raises_effective_threshold() {
        let threshold = 0.05;
        let scaled = scale_threshold_by_temperature(threshold, 0.5);
        assert!(scaled > threshold, "Low temp should raise threshold");
    }

    #[test]
    fn scale_ratio_at_default_temperature_is_identity() {
        let ratio = 0.6;
        let scaled = scale_ratio_by_temperature(ratio, DEFAULT_TEMPERATURE);
        assert!((scaled - ratio).abs() < f32::EPSILON);
    }

    #[test]
    fn high_temperature_lowers_effective_ratio() {
        let ratio = 0.6;
        let scaled = scale_ratio_by_temperature(ratio, 2.0);
        assert!(scaled < ratio, "High temp should lower ratio");
    }

    #[test]
    fn low_temperature_raises_effective_ratio_clamped() {
        let ratio = 0.6;
        let scaled = scale_ratio_by_temperature(ratio, 0.5);
        assert!(scaled > ratio, "Low temp should raise ratio");
        assert!(scaled <= 1.0, "Ratio must not exceed 1.0");
    }

    #[test]
    fn scale_mh_temperature_at_default_is_identity() {
        let base_mh = 0.01;
        let scaled = scale_mh_temperature(base_mh, DEFAULT_TEMPERATURE);
        assert!((scaled - base_mh).abs() < f32::EPSILON);
    }

    #[test]
    fn high_schedule_temperature_increases_mh_temperature() {
        let base_mh = 0.01;
        let scaled = scale_mh_temperature(base_mh, 2.0);
        assert!(
            scaled > base_mh,
            "High schedule temp should increase MH temp"
        );
    }

    #[test]
    fn zero_total_generations_returns_initial_for_linear() {
        let temp = compute_scheduled_temperature(CoolingSchedule::Linear, 1.5, 50, 0);
        assert!((temp - 1.5).abs() < f32::EPSILON);
    }
}
