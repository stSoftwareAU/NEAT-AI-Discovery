//! CPU-only pin of the numeric bounds behind the chunk 9 evaluation ledger's
//! refuted verdicts (Issue #2238, part of #2112).
//!
//! Each test recomputes a bound from the real constants — `WORKGROUP_SIZE`,
//! `wgpu::Limits::default()`, `GPU_MAX_BATCH_ALLOC_BYTES`, the host struct
//! strides and `get_bias_range` — and checks the record quotes the same value.
//! If any of them moves, the `## Refuted / not findings` row named in the
//! failing message must be re-examined. No GPU adapter is needed.
//!
//! Checks that became findings (#2313, #2314) are not asserted here; those
//! issues carry their own failing-first tests.

use std::path::PathBuf;

use neat_ai_discovery::analysis::activation::specs::{ACTIVATION_SPECS, get_bias_range};
use neat_ai_discovery::analysis::gpu::{GPU_MAX_BATCH_ALLOC_BYTES, WORKGROUP_SIZE};
use neat_ai_discovery::analysis::samples::{
    ActivationOutput, BiasResult, GpuHelpfulSample, ReluContribution,
};

const RECORD: &str = "docs/audits/security-sweep-chunk-9-gpu-wgsl.md";

const ROW_DISPATCH: &str = "refuted row `Dispatch-limit overflow at relu L167 or activation \
                            L193/L263/L568/L632 (CWE-190)`";
const ROW_NO_BYTE_CAP: &str =
    "evaluation region `No byte cap` paragraph (bias, relu and activation)";
const ROW_BIAS_DISPATCH: &str = "refuted row `Dispatch-limit overflow at bias L183 (CWE-190)`";
const ROW_NUM_STEPS: &str = "refuted row ``bias_evaluation.rs` L87 `num_steps` is unbounded``";
const ROW_TRUNCATION: &str = "refuted row ``usize as u32` length truncation in bias \
                              (L125/L126/L182), relu (L117/L166) or activation`";

fn record() -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(RECORD);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{} must exist and be readable: {e}", path.display()))
}

/// `n` with thousands separators, as the record spells lengths.
fn grouped(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// Smallest set length whose `stride`-byte buffer exceeds `limit` bytes.
fn first_failing_len(limit: u64, stride: usize) -> u64 {
    limit / stride as u64 + 1
}

/// `bias_evaluation.rs` L87: `((max_bias - min_bias) / step).ceil() as i32 + 1`.
#[allow(clippy::cast_possible_truncation)]
fn bias_num_steps((min_bias, max_bias, step): (f32, f32, f32)) -> i32 {
    ((max_bias - min_bias) / step).ceil() as i32 + 1
}

#[test]
fn per_dispatch_element_ceiling_matches_the_record() {
    let limits = wgpu::Limits::default();
    let ceiling =
        u64::from(WORKGROUP_SIZE) * u64::from(limits.max_compute_workgroups_per_dimension);
    // The #2238 brief quoted 16,777,215 (2^24 - 1); the product is 16,776,960.
    assert_eq!(
        ceiling, 16_776_960,
        "{ROW_DISPATCH}: 256 x 65,535 elements per dispatch"
    );
    let doc = record();
    assert!(
        doc.contains(&grouped(ceiling)) && doc.contains(&grouped(ceiling + 1)),
        "{ROW_DISPATCH}: the record must quote the ceiling {} and first failing length {}",
        grouped(ceiling),
        grouped(ceiling + 1)
    );
}

#[test]
fn batch_alloc_cap_matches_the_record() {
    assert_eq!(
        GPU_MAX_BATCH_ALLOC_BYTES, 268_435_456,
        "{ROW_NO_BYTE_CAP}: the record quotes GPU_MAX_BATCH_ALLOC_BYTES as 268,435,456 B"
    );
    assert!(
        record().contains("`GPU_MAX_BATCH_ALLOC_BYTES` (268,435,456 B"),
        "{ROW_NO_BYTE_CAP}: the record must quote the cap's value"
    );
}

#[test]
fn binding_limit_trips_before_the_dispatch_limit_for_relu_and_activation() {
    let limits = wgpu::Limits::default();
    let dispatch_fail =
        u64::from(WORKGROUP_SIZE) * u64::from(limits.max_compute_workgroups_per_dimension) + 1;
    let binding = limits.max_storage_buffer_binding_size;
    let doc = record();
    for (name, stride, quoted) in [
        (
            "ReluContribution",
            std::mem::size_of::<ReluContribution>(),
            3_355_444,
        ),
        (
            "ActivationOutput",
            std::mem::size_of::<ActivationOutput>(),
            4_793_491,
        ),
    ] {
        let first = first_failing_len(binding, stride);
        assert_eq!(
            first, quoted,
            "{ROW_DISPATCH}: {stride}-byte {name} binding first fails at {quoted}"
        );
        assert!(
            first < dispatch_fail,
            "{ROW_DISPATCH}: the {name} binding must trip before the dispatch limit"
        );
        assert!(
            doc.contains(&format!("{stride} B `{name}`")) && doc.contains(&grouped(first)),
            "{ROW_DISPATCH}: the record must quote the {stride} B {name} stride and {}",
            grouped(first)
        );
    }
}

#[test]
fn bias_num_steps_is_at_most_41_for_every_bias_range() {
    let names = ACTIVATION_SPECS.iter().map(|spec| spec.name).chain([
        "BIPOLAR",
        "IDENTITY",
        "",
        "not-a-squash",
    ]);
    let mut max_steps = 0;
    for name in names {
        let range = get_bias_range(name);
        let steps = bias_num_steps(range);
        assert!(
            (1..=41).contains(&steps),
            "{ROW_NUM_STEPS}: get_bias_range({name:?}) = {range:?} gives {steps} steps"
        );
        max_steps = max_steps.max(steps);
    }
    assert_eq!(
        max_steps, 41,
        "{ROW_NUM_STEPS}: the default range yields 41"
    );

    let candidates = u32::try_from(max_steps).expect("positive");
    assert_eq!(
        candidates.div_ceil(WORKGROUP_SIZE),
        1,
        "{ROW_BIAS_DISPATCH}: 41 candidates dispatch one workgroup"
    );
    let count = usize::try_from(max_steps).expect("positive");
    let results_bytes = std::mem::size_of::<BiasResult>() * count;
    let candidate_bytes = std::mem::size_of::<f32>() * count;
    assert_eq!(
        (candidate_bytes, results_bytes),
        (164, 656),
        "{ROW_BIAS_DISPATCH}: the record quotes 164 B of candidates and 656 B of results"
    );
    let doc = record();
    assert!(
        doc.contains("164 B, 656 B and one workgroup"),
        "{ROW_BIAS_DISPATCH}: the record must quote the candidate and result sizes"
    );
    assert!(
        doc.contains("`specs.rs:219`") && doc.contains("`calculation.rs:290`"),
        "{ROW_NUM_STEPS}: the verdict must cite specs.rs:219 and calculation.rs:290"
    );
}

#[test]
fn every_length_cast_follows_a_buffer_that_fails_far_below_u32_max() {
    let limits = wgpu::Limits::default();
    let doc = record();
    // The buffer each cast runs after, per the record: samples (8 B, bias L104 /
    // activation L448), relu contributions (40 B, L108), activation outputs
    // (28 B, L137/L470).
    for (name, stride, quoted) in [
        (
            "GpuHelpfulSample",
            std::mem::size_of::<GpuHelpfulSample>(),
            33_554_433,
        ),
        (
            "ReluContribution",
            std::mem::size_of::<ReluContribution>(),
            6_710_887,
        ),
        (
            "ActivationOutput",
            std::mem::size_of::<ActivationOutput>(),
            9_586_981,
        ),
    ] {
        let first = first_failing_len(limits.max_buffer_size, stride);
        assert_eq!(
            first, quoted,
            "{ROW_TRUNCATION}: the {stride}-byte {name} buffer first fails max_buffer_size at {quoted}"
        );
        assert!(
            first <= u64::from(u32::MAX),
            "{ROW_TRUNCATION}: the {name} buffer must fail before a `usize as u32` cast can wrap"
        );
        assert!(
            doc.contains(&grouped(first)),
            "{ROW_TRUNCATION}: the record must quote {}",
            grouped(first)
        );
    }
}
