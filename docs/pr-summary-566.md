## Summary

Update wgpu from 0.19 to 28.0.0 (latest stable release) and pollster from 0.3 to 0.4. Closes #566.

This is a major version jump (0.19 → 28) that brings improved Metal backend performance, better shader compilation, reduced GPU resource allocation overhead, and improved error messages. The wgpu project renumbered from 0.x to integer versions at 22.0.

### Breaking API changes addressed

| Change | Files affected |
|--------|---------------|
| `Instance::new()` takes `&InstanceDescriptor` (reference) | `device.rs` |
| `InstanceDescriptor` fields `dx12_shader_compiler` and `gles_minor_version` removed | `device.rs` |
| `Maintain::Poll` renamed to `PollType::Poll` | `device.rs` |
| `device.poll()` returns `Result<PollStatus, PollError>` instead of `MaintainResult` | `device.rs` |
| `request_adapter()` returns `Result<Adapter, RequestAdapterError>` instead of `Option<Adapter>` | `analyzer.rs`, `device.rs` |
| `DeviceDescriptor` gained `experimental_features`, `memory_hints`, `trace` fields | `analyzer.rs` |
| `request_device()` no longer takes a second `trace_path` argument | `analyzer.rs` |
| `PipelineLayoutDescriptor::push_constant_ranges` replaced by `immediate_size` | All evaluation files |
| `ComputePipelineDescriptor::entry_point` is now `Option<&str>` | All evaluation files |
| `ComputePipelineDescriptor` gained `compilation_options` and `cache` fields | All evaluation files |
| `AdapterInfo` gained `device_pci_bus_id`, `subgroup_min_size`, `subgroup_max_size`, `transient_saves_memory` fields | Tests |
| `Backend::Empty` renamed to `Backend::Noop` | Tests |

### WGSL shaders

All 8 WGSL compute shaders use standard WGSL syntax and required no changes.

## Evidence

This is a backend dependency update with no UI changes. Evidence of correctness:

- `quality.sh` passes (fmt, clippy, check, test, release build)
- All existing tests pass (`cargo test --lib --tests --all-features -- --test-threads=1`)
- All WGSL shaders compile and validate correctly with the new wgpu version
- No benchmark comparison included as this is a dependency update, not a performance optimisation — the issue notes "still proceed if the update is clean" regardless of performance

## Test Plan

- No new tests added — this is a dependency update with no behaviour changes
- All existing GPU tests pass unchanged (except `AdapterInfo` construction which gained new struct fields)
- `test_adapter_info()` helper introduced to reduce test boilerplate for `AdapterInfo` construction
