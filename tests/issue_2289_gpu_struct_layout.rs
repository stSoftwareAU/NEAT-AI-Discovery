//! Host ↔ WGSL struct-layout parity for the 12 GPU `Pod` structs (Issue #2289).
//!
//! CPU-only: no GPU adapter is requested. Each Rust struct's `size_of` /
//! `align_of` is pinned, then every WGSL mirror is parsed and validated with
//! naga (the front-end wgpu uses) and its naga-computed layout — size,
//! alignment, member names, member order, member offsets and scalar types — is
//! compared with the Rust definition. Any drift on either side fails CI.
//! The audit record is `docs/audits/security-sweep-chunk-9-gpu-wgsl.md`.

use std::mem::{align_of, offset_of, size_of};

use naga::proc::{Alignment, Layouter};
use naga::{Module, ScalarKind, TypeInner};
use neat_ai_discovery::analysis::gpu::shaders::{
    ACTIVATION_REDUCE_SHADER, ACTIVATION_SHADER, BIAS_SHADER, HARMFUL_REDUCE_SHADER,
    HARMFUL_SHADER, HELPFUL_REDUCE_SHADER, HELPFUL_SHADER, RELU_SHADER,
};
use neat_ai_discovery::analysis::samples::{
    ActivationOutput, ActivationUniforms, BiasResult, BiasUniforms, GpuHelpfulSample,
    HarmfulContribution, HarmfulUniforms, HelpfulContribution, HelpfulUniforms, ReductionUniforms,
    ReluContribution, ReluUniforms,
};

/// Host-side layout of one Rust struct: size, alignment and `(name, offset, scalar)` per field.
struct RustLayout {
    name: &'static str,
    size: usize,
    align: usize,
    fields: Vec<(&'static str, usize, &'static str)>,
}

/// Builds a [`RustLayout`]; the `field: ty` list is type-checked against the struct,
/// so a field whose Rust type changes stops this test compiling.
macro_rules! rust_layout {
    ($s:ident { $($f:ident : $t:ident),* $(,)? }) => {{
        $( let _: fn(&$s) -> &$t = |s| &s.$f; )*
        RustLayout {
            name: stringify!($s),
            size: size_of::<$s>(),
            align: align_of::<$s>(),
            fields: vec![$( (stringify!($f), offset_of!($s, $f), stringify!($t)) ),*],
        }
    }};
}

fn gpu_helpful_sample() -> RustLayout {
    rust_layout!(GpuHelpfulSample {
        activation: f32,
        avg_error: f32
    })
}

fn helpful_contribution() -> RustLayout {
    rust_layout!(HelpfulContribution {
        positive_flag: u32,
        negative_flag: u32,
        positive_improvement: f32,
        negative_improvement: f32,
        positive_activation: f32,
        negative_activation: f32,
        error_squared: f32,
        activation_squared: f32,
        error_activation: f32,
        pad0: f32,
        pad1: f32,
        pad2: f32,
    })
}

fn helpful_uniforms() -> RustLayout {
    rust_layout!(HelpfulUniforms {
        length: u32,
        pad0: u32,
        epsilon: f32,
        pad1: f32
    })
}

fn harmful_contribution() -> RustLayout {
    rust_layout!(HarmfulContribution {
        harmful_flag: u32,
        helpful_flag: u32,
        error_magnitude: f32,
        pad0: f32,
    })
}

fn harmful_uniforms() -> RustLayout {
    rust_layout!(HarmfulUniforms {
        length: u32,
        pad0: u32,
        epsilon: f32,
        weight: f32
    })
}

fn relu_contribution() -> RustLayout {
    rust_layout!(ReluContribution {
        positive_activation_sq: f32,
        positive_error_activation: f32,
        positive_count: u32,
        negative_activation_sq: f32,
        negative_error_activation: f32,
        negative_count: u32,
        error_sq: f32,
        pad0: f32,
        pad1: u32,
        pad2: u32,
    })
}

fn relu_uniforms() -> RustLayout {
    rust_layout!(ReluUniforms {
        length: u32,
        threshold: f32,
        epsilon: f32,
        pad0: f32
    })
}

fn bias_result() -> RustLayout {
    rust_layout!(BiasResult {
        bias_value: f32,
        error_reduction: f32,
        valid_sample_count: u32,
        pad0: u32,
    })
}

fn bias_uniforms() -> RustLayout {
    rust_layout!(BiasUniforms {
        sample_count: u32,
        bias_count: u32,
        incoming_weight: f32,
        outgoing_weight: f32,
        activation_type: u32,
        epsilon: f32,
        min_sample_count: u32,
        pad0: u32,
    })
}

fn activation_output() -> RustLayout {
    rust_layout!(ActivationOutput {
        output: f32,
        output_sq: f32,
        error_output: f32,
        valid: u32,
        pad0: u32,
        pad1: u32,
        pad2: u32,
    })
}

fn activation_uniforms() -> RustLayout {
    rust_layout!(ActivationUniforms {
        sample_count: u32,
        orientation: f32,
        scale: f32,
        activation_type: u32,
        epsilon: f32,
        pad0: f32,
        pad1: f32,
    })
}

fn reduction_uniforms() -> RustLayout {
    rust_layout!(ReductionUniforms {
        contribution_count: u32,
        pad0: u32,
        pad1: u32,
        pad2: u32
    })
}

/// One WGSL mirror: `(shader file, shader source, WGSL struct name)`.
type Mirror = (&'static str, &'static str, &'static str);

/// Every host struct and each WGSL struct that mirrors it.
fn parity_map() -> Vec<(RustLayout, Vec<Mirror>)> {
    vec![
        (
            gpu_helpful_sample(),
            vec![
                ("helpful.wgsl", HELPFUL_SHADER, "HelpfulSample"),
                ("harmful.wgsl", HARMFUL_SHADER, "HarmfulSample"),
                ("relu.wgsl", RELU_SHADER, "HelpfulSample"),
                ("activation.wgsl", ACTIVATION_SHADER, "HelpfulSample"),
                ("bias.wgsl", BIAS_SHADER, "HelpfulSample"),
            ],
        ),
        (
            helpful_contribution(),
            vec![
                ("helpful.wgsl", HELPFUL_SHADER, "HelpfulContribution"),
                (
                    "helpful_reduce.wgsl",
                    HELPFUL_REDUCE_SHADER,
                    "HelpfulContribution",
                ),
            ],
        ),
        (
            helpful_uniforms(),
            vec![("helpful.wgsl", HELPFUL_SHADER, "HelpfulUniforms")],
        ),
        (
            harmful_contribution(),
            vec![
                ("harmful.wgsl", HARMFUL_SHADER, "HarmfulContribution"),
                (
                    "harmful_reduce.wgsl",
                    HARMFUL_REDUCE_SHADER,
                    "HarmfulContribution",
                ),
            ],
        ),
        (
            harmful_uniforms(),
            vec![("harmful.wgsl", HARMFUL_SHADER, "HarmfulUniforms")],
        ),
        (
            relu_contribution(),
            vec![("relu.wgsl", RELU_SHADER, "ReluContribution")],
        ),
        (
            relu_uniforms(),
            vec![("relu.wgsl", RELU_SHADER, "ReluUniforms")],
        ),
        (
            bias_result(),
            vec![("bias.wgsl", BIAS_SHADER, "BiasResult")],
        ),
        (
            bias_uniforms(),
            vec![("bias.wgsl", BIAS_SHADER, "BiasUniforms")],
        ),
        (
            activation_output(),
            vec![
                ("activation.wgsl", ACTIVATION_SHADER, "ActivationOutput"),
                (
                    "activation_reduce.wgsl",
                    ACTIVATION_REDUCE_SHADER,
                    "ActivationOutput",
                ),
            ],
        ),
        (
            activation_uniforms(),
            vec![("activation.wgsl", ACTIVATION_SHADER, "ActivationUniforms")],
        ),
        (
            reduction_uniforms(),
            vec![
                (
                    "helpful_reduce.wgsl",
                    HELPFUL_REDUCE_SHADER,
                    "ReductionUniforms",
                ),
                (
                    "harmful_reduce.wgsl",
                    HARMFUL_REDUCE_SHADER,
                    "ReductionUniforms",
                ),
                (
                    "activation_reduce.wgsl",
                    ACTIVATION_REDUCE_SHADER,
                    "ReductionUniforms",
                ),
            ],
        ),
    ]
}

/// Parses and fully validates a shader, as wgpu does when it builds the module.
fn parse_and_validate(file: &str, source: &str) -> (Module, Layouter) {
    let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|e| panic!("{file} is not valid WGSL: {e}"));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("{file} failed naga validation (layout rules included): {e:?}"));
    let mut layouter = Layouter::default();
    layouter
        .update(module.to_ctx())
        .unwrap_or_else(|e| panic!("{file}: naga could not lay out its types: {e}"));
    (module, layouter)
}

/// A Rust `align_of` value as a naga [`Alignment`]; `None` if it is not a power of two.
fn naga_alignment(align: usize) -> Option<Alignment> {
    u32::try_from(align).ok().and_then(Alignment::new)
}

/// WGSL scalar name for a member type; anything else (a `vec3`, a nested struct) is named as such.
fn member_scalar(module: &Module, ty: naga::Handle<naga::Type>) -> String {
    match &module.types[ty].inner {
        TypeInner::Scalar(s) if s.width == 4 => match s.kind {
            ScalarKind::Float => "f32".to_string(),
            ScalarKind::Uint => "u32".to_string(),
            ScalarKind::Sint => "i32".to_string(),
            other => format!("{other:?}"),
        },
        other => format!("{other:?}"),
    }
}

#[test]
fn rust_pod_structs_have_pinned_size_and_alignment() {
    // Pinned values from Issue #2289; a change here must be matched in the WGSL mirror.
    let expected: [(&str, usize, usize); 12] = [
        ("GpuHelpfulSample", 8, 4),
        ("HelpfulContribution", 48, 4),
        ("HelpfulUniforms", 16, 4),
        ("HarmfulContribution", 16, 4),
        ("HarmfulUniforms", 16, 4),
        ("ReluContribution", 40, 4),
        ("ReluUniforms", 16, 4),
        ("BiasResult", 16, 4),
        ("BiasUniforms", 32, 4),
        ("ActivationOutput", 28, 4),
        ("ActivationUniforms", 28, 4),
        ("ReductionUniforms", 16, 4),
    ];
    let actual: Vec<(&str, usize, usize)> = parity_map()
        .iter()
        .map(|(r, _)| (r.name, r.size, r.align))
        .collect();
    assert_eq!(actual, expected.to_vec());

    // The same pins read straight from `std::mem`, independent of the helper above.
    assert_eq!(
        (
            size_of::<GpuHelpfulSample>(),
            align_of::<GpuHelpfulSample>()
        ),
        (8, 4)
    );
    assert_eq!(
        (
            size_of::<HelpfulContribution>(),
            align_of::<HelpfulContribution>()
        ),
        (48, 4)
    );
    assert_eq!(
        (size_of::<HelpfulUniforms>(), align_of::<HelpfulUniforms>()),
        (16, 4)
    );
    assert_eq!(
        (
            size_of::<HarmfulContribution>(),
            align_of::<HarmfulContribution>()
        ),
        (16, 4)
    );
    assert_eq!(
        (size_of::<HarmfulUniforms>(), align_of::<HarmfulUniforms>()),
        (16, 4)
    );
    assert_eq!(
        (
            size_of::<ReluContribution>(),
            align_of::<ReluContribution>()
        ),
        (40, 4)
    );
    assert_eq!(
        (size_of::<ReluUniforms>(), align_of::<ReluUniforms>()),
        (16, 4)
    );
    assert_eq!((size_of::<BiasResult>(), align_of::<BiasResult>()), (16, 4));
    assert_eq!(
        (size_of::<BiasUniforms>(), align_of::<BiasUniforms>()),
        (32, 4)
    );
    assert_eq!(
        (
            size_of::<ActivationOutput>(),
            align_of::<ActivationOutput>()
        ),
        (28, 4)
    );
    assert_eq!(
        (
            size_of::<ActivationUniforms>(),
            align_of::<ActivationUniforms>()
        ),
        (28, 4)
    );
    assert_eq!(
        (
            size_of::<ReductionUniforms>(),
            align_of::<ReductionUniforms>()
        ),
        (16, 4)
    );
}

#[test]
fn every_wgsl_mirror_matches_its_rust_struct_layout() {
    let mut mirrors_checked = 0usize;
    for (rust, mirrors) in parity_map() {
        assert!(!mirrors.is_empty(), "{} has no WGSL mirror", rust.name);
        for (file, source, wgsl_name) in mirrors {
            let (module, layouter) = parse_and_validate(file, source);
            let (handle, ty) = module
                .types
                .iter()
                .find(|(_, t)| t.name.as_deref() == Some(wgsl_name))
                .unwrap_or_else(|| panic!("{file} no longer declares struct {wgsl_name}"));
            let TypeInner::Struct { members, span } = &ty.inner else {
                panic!("{file}: {wgsl_name} is not a struct");
            };
            let layout = layouter[handle];
            let at = format!("{} ↔ {file}::{wgsl_name}", rust.name);

            assert_eq!(*span as usize, rust.size, "{at}: WGSL span vs size_of");
            assert_eq!(
                layout.size as usize, rust.size,
                "{at}: naga layout size vs size_of"
            );
            assert_eq!(
                Some(layout.alignment),
                naga_alignment(rust.align),
                "{at}: naga alignment vs align_of"
            );

            let wgsl_fields: Vec<(String, usize, String)> = members
                .iter()
                .map(|m| {
                    (
                        m.name.clone().unwrap_or_default(),
                        m.offset as usize,
                        member_scalar(&module, m.ty),
                    )
                })
                .collect();
            let rust_fields: Vec<(String, usize, String)> = rust
                .fields
                .iter()
                .map(|(n, o, t)| ((*n).to_string(), *o, (*t).to_string()))
                .collect();
            assert_eq!(
                wgsl_fields, rust_fields,
                "{at}: member (name, offset, scalar) order differs"
            );
            mirrors_checked += 1;
        }
    }
    // 12 structs across 21 shader declarations — guards against a silently shrunk map.
    assert_eq!(mirrors_checked, 21);
}

#[test]
fn host_shared_array_strides_equal_rust_size() {
    // `array<T>` in storage/workgroup space: naga's stride must equal the host
    // `size_of::<T>() * len` sizing — covers the 28-byte `ActivationOutput` stride.
    let sizes: Vec<(&str, usize)> = parity_map().iter().map(|(r, _)| (r.name, r.size)).collect();
    let rust_size_for = |wgsl_name: &str| -> Option<usize> {
        let rust_name = match wgsl_name {
            "HelpfulSample" | "HarmfulSample" => "GpuHelpfulSample",
            other => other,
        };
        sizes.iter().find(|(n, _)| *n == rust_name).map(|(_, s)| *s)
    };

    let shaders = [
        ("helpful.wgsl", HELPFUL_SHADER),
        ("harmful.wgsl", HARMFUL_SHADER),
        ("relu.wgsl", RELU_SHADER),
        ("activation.wgsl", ACTIVATION_SHADER),
        ("bias.wgsl", BIAS_SHADER),
        ("helpful_reduce.wgsl", HELPFUL_REDUCE_SHADER),
        ("harmful_reduce.wgsl", HARMFUL_REDUCE_SHADER),
        ("activation_reduce.wgsl", ACTIVATION_REDUCE_SHADER),
    ];
    let mut struct_arrays = 0usize;
    for (file, source) in shaders {
        let (module, _) = parse_and_validate(file, source);
        for (_, var) in module.global_variables.iter() {
            let TypeInner::Array { base, stride, .. } = module.types[var.ty].inner else {
                continue;
            };
            let Some(base_name) = module.types[base].name.as_deref() else {
                continue;
            };
            if let Some(size) = rust_size_for(base_name) {
                assert_eq!(
                    stride as usize, size,
                    "{file}: array<{base_name}> stride differs from the host struct size"
                );
                struct_arrays += 1;
            }
        }
    }
    assert!(
        struct_arrays > 0,
        "no struct-typed arrays found — the stride check is vacuous"
    );
}

#[test]
fn uniform_bindings_accept_28_byte_activation_uniforms() {
    // Decision record (Issue #2289): WGSL's 16-byte uniform rounding applies to a
    // struct *nested* in a uniform buffer or an array *element* there, not to the
    // top-level struct a `var<uniform>` binds. A 28-byte top-level uniform is valid;
    // naga's validator (run in `parse_and_validate`) is the executable proof.
    let (module, layouter) = parse_and_validate("activation.wgsl", ACTIVATION_SHADER);
    let uniform = module
        .global_variables
        .iter()
        .find(|(_, v)| v.space == naga::AddressSpace::Uniform)
        .map(|(_, v)| v.ty)
        .expect("activation.wgsl must bind a var<uniform>");
    assert_eq!(
        module.types[uniform].name.as_deref(),
        Some("ActivationUniforms")
    );
    assert_eq!(
        layouter[uniform].size as usize,
        size_of::<ActivationUniforms>()
    );
    assert_eq!(
        size_of::<ActivationUniforms>() % 16,
        12,
        "28 bytes is not a multiple of 16"
    );
}

#[test]
fn a_drifted_wgsl_mirror_is_detected() {
    // Error path: a WGSL mirror with a `vec3<f32>` (align 16) must not look like the host struct.
    let drifted = "struct ReductionUniforms { contribution_count: u32, pad: vec3<u32> };\n\
                   @group(0) @binding(0) var<uniform> u: ReductionUniforms;\n\
                   @compute @workgroup_size(1) fn main() { let x = u.contribution_count; }";
    let (module, layouter) = parse_and_validate("drifted.wgsl", drifted);
    let (handle, ty) = module
        .types
        .iter()
        .find(|(_, t)| t.name.as_deref() == Some("ReductionUniforms"))
        .expect("struct present");
    let TypeInner::Struct { members, .. } = &ty.inner else {
        panic!("not a struct");
    };
    let rust = reduction_uniforms();
    assert_ne!(
        layouter[handle].size as usize, rust.size,
        "vec3 pads the struct to 32 bytes"
    );
    assert_ne!(Some(layouter[handle].alignment), naga_alignment(rust.align));
    assert_eq!(members[1].offset, 16, "vec3<u32> is 16-byte aligned");
    assert!(
        matches!(module.types[members[1].ty].inner, TypeInner::Vector { .. }),
        "the drifted member is a vector, not a host scalar"
    );
}
