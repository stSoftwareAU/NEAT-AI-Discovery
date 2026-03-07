//! Macros for reducing boilerplate in discovery module spec builders (Issue #773).
//!
//! The `discovery_spec!` macro encapsulates the common clone → guard → detect →
//! convert pipeline shared by all discovery module specs. Each invocation produces
//! a `DiscoveryModuleSpec` and pushes it onto the target vector.
//!
//! # Why a macro?
//!
//! A helper function cannot capture the heterogeneous closure types produced by
//! each module's detect/convert pair without boxing them individually. The macro
//! generates the boilerplate inline, preserving zero-cost move semantics while
//! reducing ~33 nearly identical code blocks to concise declarations.
//!
//! # Usage
//!
//! See the four `*_specs.rs` files for real-world examples of every variant.

/// Build and push a `DiscoveryModuleSpec` with the standard detect → convert
/// pipeline.
///
/// # Variants
///
/// ## Standard (with optional pre-record guard)
///
/// ```ignore
/// discovery_spec!(modules, "name", "phase", cache = shared_cache, hidden = hidden_neurons =>
///     guard: hidden,
///     records: cache.load_records_for_hidden(&hidden),
///     detect: |records| module::detect(&hidden, &records),
///     convert: |detected| module::to_candidates(&detected),
/// );
/// ```
///
/// ## Post-record emptiness guard
///
/// ```ignore
/// discovery_spec!(modules, "name", "phase", cache = shared_cache, creature = creature =>
///     records: cache.load_records_for_neuron_types(&creature, &["input"]),
///     guard_records,
///     detect: |records| module::detect(&creature, &records),
///     convert: |detected| module::to_candidates(&detected),
/// );
/// ```
///
/// ## Minimum count guard
///
/// ```ignore
/// discovery_spec!(modules, "name", "phase", cache = shared_cache, hidden = hidden_neurons =>
///     guard_min: hidden 2,
///     records: cache.load_records_for_hidden(&hidden),
///     detect: |records| module::detect(&creature, &records),
///     convert: |detected| module::to_candidates(&detected),
/// );
/// ```
///
/// ## Custom closure (non-standard logic)
///
/// ```ignore
/// discovery_spec!(modules, "name", "phase", cache = shared_cache, creature = creature =>
///     custom: move || { /* ... */ },
/// );
/// ```
macro_rules! discovery_spec {
    // ── Standard pipeline with optional pre-record guard ──
    ($modules:expr, $name:expr, $phase:expr,
     $($clone_name:ident = $clone_src:expr),+ =>
        $(guard: $guard:ident,)?
        records: $load_records:expr,
        detect: |$rec:ident| $detect_expr:expr,
        convert: |$det:ident| $convert_expr:expr $(,)?
    ) => {
        {
            $( let $clone_name = ::std::sync::Arc::clone(&$clone_src); )+
            $modules.push($crate::analysis::discovery_dispatch::DiscoveryModuleSpec {
                module_name: $name.to_string(),
                phase_name: $phase,
                detect_fn: Box::new(move || {
                    $( if $guard.is_empty() { return None; } )?
                    let $rec = $load_records;
                    let $det = $detect_expr;
                    if $det.is_empty() {
                        return None;
                    }
                    let candidates = $convert_expr;
                    Some($crate::analysis::discovery_dispatch::DiscoveryDetectionResult {
                        detected_count: $det.len(),
                        candidates,
                    })
                }),
            });
        }
    };

    // ── Standard pipeline with post-record emptiness guard ──
    ($modules:expr, $name:expr, $phase:expr,
     $($clone_name:ident = $clone_src:expr),+ =>
        records: $load_records:expr,
        guard_records,
        detect: |$rec:ident| $detect_expr:expr,
        convert: |$det:ident| $convert_expr:expr $(,)?
    ) => {
        {
            $( let $clone_name = ::std::sync::Arc::clone(&$clone_src); )+
            $modules.push($crate::analysis::discovery_dispatch::DiscoveryModuleSpec {
                module_name: $name.to_string(),
                phase_name: $phase,
                detect_fn: Box::new(move || {
                    let $rec = $load_records;
                    if $rec.is_empty() {
                        return None;
                    }
                    let $det = $detect_expr;
                    if $det.is_empty() {
                        return None;
                    }
                    let candidates = $convert_expr;
                    Some($crate::analysis::discovery_dispatch::DiscoveryDetectionResult {
                        detected_count: $det.len(),
                        candidates,
                    })
                }),
            });
        }
    };

    // ── Minimum-count guard variant ──
    ($modules:expr, $name:expr, $phase:expr,
     $($clone_name:ident = $clone_src:expr),+ =>
        guard_min: $guard_var:ident $min_count:expr,
        records: $load_records:expr,
        detect: |$rec:ident| $detect_expr:expr,
        convert: |$det:ident| $convert_expr:expr $(,)?
    ) => {
        {
            $( let $clone_name = ::std::sync::Arc::clone(&$clone_src); )+
            $modules.push($crate::analysis::discovery_dispatch::DiscoveryModuleSpec {
                module_name: $name.to_string(),
                phase_name: $phase,
                detect_fn: Box::new(move || {
                    if $guard_var.len() < $min_count {
                        return None;
                    }
                    let $rec = $load_records;
                    let $det = $detect_expr;
                    if $det.is_empty() {
                        return None;
                    }
                    let candidates = $convert_expr;
                    Some($crate::analysis::discovery_dispatch::DiscoveryDetectionResult {
                        detected_count: $det.len(),
                        candidates,
                    })
                }),
            });
        }
    };

    // ── Custom closure (for non-standard detection logic) ──
    ($modules:expr, $name:expr, $phase:expr,
     $($clone_name:ident = $clone_src:expr),+ =>
        custom: $closure:expr $(,)?
    ) => {
        {
            $( let $clone_name = ::std::sync::Arc::clone(&$clone_src); )+
            $modules.push($crate::analysis::discovery_dispatch::DiscoveryModuleSpec {
                module_name: $name.to_string(),
                phase_name: $phase,
                detect_fn: Box::new($closure),
            });
        }
    };
}
