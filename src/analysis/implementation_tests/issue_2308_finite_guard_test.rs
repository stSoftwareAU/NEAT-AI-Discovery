//! WGSL `is_finite_value` NaN/Inf guard (Issue #2308).
//!
//! WGSL has no `isNan`/`isInf`, and a float self-comparison (`v != v`) may be
//! folded to `false` under fast-math (Metal's default), letting an overflowed
//! `±Inf`/`NaN` through as `valid = 1u`. The guard must be an integer bit test
//! on the exponent, which fast-math cannot fold.

use super::common::*;
use crate::analysis::gpu::shaders::{
    ACTIVATION_REDUCE_SHADER, ACTIVATION_SHADER, HARMFUL_REDUCE_SHADER, HARMFUL_SHADER,
    HELPFUL_REDUCE_SHADER, HELPFUL_SHADER, RELU_SHADER,
};

const GUARD: &str = "is_finite_value";

/// Every embedded shader, paired with a name for assertions.
const EMBEDDED_SHADERS: &[(&str, &str)] = &[
    ("helpful", HELPFUL_SHADER),
    ("harmful", HARMFUL_SHADER),
    ("relu", RELU_SHADER),
    ("activation", ACTIVATION_SHADER),
    ("helpful_reduce", HELPFUL_REDUCE_SHADER),
    ("harmful_reduce", HARMFUL_REDUCE_SHADER),
    ("activation_reduce", ACTIVATION_REDUCE_SHADER),
];

/// Edge inputs spanning every exponent class: zeros, subnormals, normals,
/// the largest finite values, both infinities and quiet/signalling NaNs.
fn edge_inputs() -> Vec<f32> {
    vec![
        0.0,
        -0.0,
        f32::from_bits(1),            // smallest subnormal
        -f32::from_bits(0x007f_ffff), // largest negative subnormal
        f32::MIN_POSITIVE,
        1.0,
        -1.0e20,
        f32::MAX,
        f32::MIN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        -f32::NAN,
        f32::from_bits(0x7f80_0001), // signalling NaN
        f32::from_bits(0xffc0_0000),
    ]
}

#[test]
fn wgsl_is_finite_value_is_an_integer_exponent_test() {
    let mut checked = Vec::new();
    for (name, source) in EMBEDDED_SHADERS {
        let module = naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|e| panic!("{name} shader is not valid WGSL: {e}"));
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{name} shader failed WGSL validation: {e:?}"));

        let Some((handle, function)) = module
            .functions
            .iter()
            .find(|(_, f)| f.name.as_deref() == Some(GUARD))
        else {
            continue;
        };
        let fn_info = &info[handle];
        let is_float = |expr: naga::Handle<naga::Expression>| {
            matches!(
                fn_info[expr].ty.inner_with(&module.types),
                naga::TypeInner::Scalar(naga::Scalar {
                    kind: naga::ScalarKind::Float,
                    ..
                })
            )
        };

        let mut has_bitcast = false;
        let mut masks = Vec::new();
        for (_, expr) in function.expressions.iter() {
            match *expr {
                naga::Expression::As {
                    kind: naga::ScalarKind::Uint,
                    convert: None,
                    ..
                } => has_bitcast = true,
                naga::Expression::Literal(naga::Literal::U32(v)) => masks.push(v),
                naga::Expression::Binary { op, left, .. } => {
                    use naga::BinaryOperator as B;
                    let is_comparison = matches!(
                        op,
                        B::Equal
                            | B::NotEqual
                            | B::Less
                            | B::LessEqual
                            | B::Greater
                            | B::GreaterEqual
                    );
                    assert!(
                        !(is_comparison && is_float(left)),
                        "{name}: {GUARD} must not compare floats ({op:?}) — fast-math may fold it"
                    );
                }
                _ => {}
            }
        }
        assert!(
            has_bitcast,
            "{name}: {GUARD} must bitcast its argument to u32"
        );

        // The exponent mask, read from the shader itself, must classify every
        // edge input exactly as `f32::is_finite` does.
        masks.dedup();
        let [mask] = masks[..] else {
            panic!("{name}: {GUARD} should use exactly one u32 mask literal, found {masks:?}");
        };
        for x in edge_inputs() {
            assert_eq!(
                (x.to_bits() & mask) != mask,
                x.is_finite(),
                "{name}: {GUARD} mask {mask:#010x} misclassifies {x:?} ({:#010x})",
                x.to_bits()
            );
        }
        checked.push(*name);
    }
    // The two shaders that gate samples on finiteness must both be covered.
    assert_eq!(checked, ["relu", "activation"], "shaders defining {GUARD}");
}

#[test]
fn activation_shader_drops_overflowing_sample() {
    skip_if_no_gpu!();
    let analyzer = GpuAnalyzer::new().expect("GPU analysis should be available");

    // Identity activation (GPU id 6) with scale 1e20: the first sample gives
    // pre-activation 1.0, the second overflows to +Inf and must be marked
    // `valid == 0`, so it contributes nothing to the sums.
    let sample = |activation| HelpfulSample {
        activation,
        avg_error: 1.0,
        target_value: None,
        target_activation: None,
    };
    let samples = [sample(1.0e-20), sample(1.0e20)];
    let (sum_sq, sum_err_act, _, _) = analyzer
        .evaluate_activation_gpu(&samples, 6, 1.0, 1.0e20)
        .expect("activation evaluation should succeed");

    assert!(
        sum_sq.is_finite(),
        "overflowing sample leaked into sum_sq: {sum_sq}"
    );
    assert!(
        sum_err_act.is_finite(),
        "overflowing sample leaked into sum_error_activation: {sum_err_act}"
    );
    assert!(
        (sum_sq - 1.0).abs() < 1.0e-3,
        "sum_sq {sum_sq} should be the finite sample's 1.0"
    );
    assert!(
        (sum_err_act - 1.0).abs() < 1.0e-3,
        "sum_error_activation {sum_err_act} should be the finite sample's 1.0"
    );
}
