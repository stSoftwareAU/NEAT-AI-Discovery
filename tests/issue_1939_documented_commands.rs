//! Issue #1939 — documented commands and samples must run as written.
//!
//! Four copy-paste paths failed exactly as documented:
//!
//! 1. The README's cross-repo `runlib.sh` invocation aborts, because the script
//!    hard-requires `Cargo.toml` in the caller's working directory.
//! 2. The streaming guide's append payload spelt the nested key `neuronUuid`,
//!    but [`NeuronData`] has no `rename_all`, so the wire key is `neuron_uuid`.
//! 3. `NEAT_AI_DISCOVERY_PRELOAD_ALL=0` is indistinguishable from unset — the
//!    documented "disable preload explicitly" recipe forces nothing.
//! 4. The README's fuzz commands omitted `--locked`, so a contributor did not
//!    reproduce the pinned `fuzz/Cargo.lock` resolution that CI uses.
//!
//! Every test below either drives the real code path the doc describes, or
//! locks the corrected doc text against the artefact it must agree with.

use neat_ai_discovery::ffi_types::AppendRecordsInput;
use serial_test::serial;

const README: &str = include_str!("../README.md");
const STREAMING_GUIDE: &str = include_str!("../docs/STREAMING_GUIDE.md");
const CACHE_TUNING: &str = include_str!("../docs/CACHE_TUNING.md");
const RECORDING_FFI: &str = include_str!("../src/ffi/recording.rs");
const CI_FUZZING_WORKFLOW: &str = include_str!("../docs/ci-fuzzing-workflow.yml");
const FUZZ_CI_SCRIPT: &str = include_str!("../scripts/fuzz-ci.sh");

// ============================================================================
// 1. Cross-repo `runlib.sh` invocation (README).
// ============================================================================

/// The behavioural reason the README had to change: the script aborts when the
/// caller's working directory holds no `Cargo.toml`, which is every cross-repo
/// invocation from the NEAT-AI (Deno) directory.
#[test]
fn runlib_aborts_when_invoked_from_a_directory_without_cargo_toml() {
    let script = concat!(env!("CARGO_MANIFEST_DIR"), "/scripts/runlib.sh");
    let elsewhere = tempfile::tempdir().expect("temp dir");

    let output = std::process::Command::new("bash")
        .arg(script)
        .current_dir(elsewhere.path())
        .stdin(std::process::Stdio::null())
        .output()
        .expect("runlib.sh should be executable via bash");

    assert!(
        !output.status.success(),
        "runlib.sh must fail when the caller's cwd has no Cargo.toml"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("Cargo.toml not found"),
        "expected the missing-manifest abort, got stderr: {stderr}"
    );
}

/// The documented cross-repo command must `cd` into the crate root first,
/// otherwise it reproduces the abort above.
#[test]
fn readme_cross_repo_invocation_changes_directory_first() {
    assert!(
        README.contains("(cd ../NEAT-AI-Discovery && ./scripts/runlib.sh)"),
        "README must document the subshell form that cds to the crate root"
    );
    assert!(
        !README.contains("\n../NEAT-AI-Discovery/scripts/runlib.sh"),
        "README must not document the bare cross-repo path, which aborts"
    );
}

// ============================================================================
// 2. Streaming append payload — the nested key is `neuron_uuid`.
// ============================================================================

/// The documented payload, spelt as the guide now spells it, must deserialise.
#[test]
fn documented_append_payload_deserialises() {
    let payload = r#"{
      "sessionId": "550e8400-e29b-41d4-a716-446655440000",
      "observations": [
        {
          "obsIndex": 0,
          "neuronData": [
            { "neuron_uuid": "hidden-1", "activation": 0.5, "value": 0.4, "errors": [0.1] }
          ],
          "inputs": [0.1, 0.2, 0.3]
        }
      ]
    }"#;

    let input: AppendRecordsInput =
        serde_json::from_str(payload).expect("documented payload must deserialise");

    assert_eq!(input.session_id, "550e8400-e29b-41d4-a716-446655440000");
    assert_eq!(input.observations.len(), 1);
    assert_eq!(input.observations[0].obs_index, 0);
    assert_eq!(input.observations[0].neuron_data[0].neuron_uuid, "hidden-1");
    assert_eq!(input.observations[0].inputs, vec![0.1, 0.2, 0.3]);
}

/// The previously documented spelling is a hard deserialisation failure — this
/// is the regression the doc fix prevents.
#[test]
fn camel_case_neuron_uuid_is_rejected_at_the_json_boundary() {
    let payload = r#"{
      "sessionId": "550e8400-e29b-41d4-a716-446655440000",
      "observations": [
        {
          "obsIndex": 0,
          "neuronData": [
            { "neuronUuid": "hidden-1", "activation": 0.5, "value": 0.4, "errors": [0.1] }
          ],
          "inputs": [0.1, 0.2, 0.3]
        }
      ]
    }"#;

    let err = serde_json::from_str::<AppendRecordsInput>(payload)
        .expect_err("`neuronUuid` must not deserialise into NeuronData");
    assert!(
        err.to_string().contains("neuron_uuid"),
        "error should name the missing `neuron_uuid` field, got: {err}"
    );
}

/// The same payload through the real FFI entry point returns a structured
/// failure rather than appending anything.
#[test]
fn ffi_append_reports_a_parse_failure_for_camel_case_neuron_uuid() {
    let payload = std::ffi::CString::new(
        r#"{"sessionId":"550e8400-e29b-41d4-a716-446655440000","observations":[{"obsIndex":0,"neuronData":[{"neuronUuid":"hidden-1","activation":0.5,"errors":[0.1]}],"inputs":[0.1]}]}"#,
    )
    .expect("payload has no interior NUL");

    // SAFETY: `payload` is a valid null-terminated C string that outlives the
    // call, and the returned pointer is freed below.
    let result_ptr = unsafe { neat_ai_discovery::ffi::append_discovery_records(payload.as_ptr()) };
    assert!(!result_ptr.is_null(), "FFI must return a JSON result");

    // SAFETY: the pointer came from the call above and is a valid C string.
    let result = unsafe { std::ffi::CStr::from_ptr(result_ptr) }
        .to_str()
        .expect("FFI result is UTF-8")
        .to_owned();
    // SAFETY: frees the pointer returned by `append_discovery_records`.
    unsafe { neat_ai_discovery::ffi::free_discovery_result(result_ptr) };

    let parsed: serde_json::Value = serde_json::from_str(&result).expect("FFI result is JSON");
    assert_eq!(
        parsed["success"], false,
        "append must not succeed: {result}"
    );
    assert!(
        parsed["error"]
            .as_str()
            .unwrap_or_default()
            .contains("neuron_uuid"),
        "error should name the missing `neuron_uuid` field, got: {result}"
    );
}

/// The guide and the rustdoc that mirrors it must both use the wire spelling.
#[test]
fn docs_use_the_wire_spelling_for_the_nested_neuron_key() {
    assert!(
        !STREAMING_GUIDE.contains("neuronUuid"),
        "docs/STREAMING_GUIDE.md must not document the non-deserialisable `neuronUuid`"
    );
    assert!(
        STREAMING_GUIDE.contains("neuron_uuid"),
        "docs/STREAMING_GUIDE.md must document the `neuron_uuid` wire key"
    );
    assert!(
        !RECORDING_FFI.contains(r#""neuronUuid""#),
        "src/ffi/recording.rs rustdoc must not document `neuronUuid`"
    );
}

// ============================================================================
// 3. `NEAT_AI_DISCOVERY_PRELOAD_ALL=0` is a no-op.
// ============================================================================

/// `=0` is indistinguishable from unset: it forces nothing, so the doc must not
/// present it as a way to disable preload.
#[test]
#[serial]
fn preload_all_zero_behaves_exactly_like_unset() {
    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::remove_var("NEAT_AI_DISCOVERY_PRELOAD_ALL") };
    let unset_preload = neat_ai_discovery::config::preload_all();
    let unset_streaming = neat_ai_discovery::config::streaming_enabled();

    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::set_var("NEAT_AI_DISCOVERY_PRELOAD_ALL", "0") };
    assert_eq!(
        neat_ai_discovery::config::preload_all(),
        unset_preload,
        "`=0` must not differ from unset"
    );
    assert_eq!(
        neat_ai_discovery::config::streaming_enabled(),
        unset_streaming,
        "`=0` must not force streaming"
    );

    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::remove_var("NEAT_AI_DISCOVERY_PRELOAD_ALL") };
}

/// Only `=1` forces preload, which is the sole documented override.
#[test]
#[serial]
fn preload_all_one_disables_streaming() {
    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::set_var("NEAT_AI_DISCOVERY_PRELOAD_ALL", "1") };
    assert!(neat_ai_discovery::config::preload_all());
    assert!(!neat_ai_discovery::config::streaming_enabled());

    // SAFETY: serialised via #[serial] — no concurrent env access.
    unsafe { std::env::remove_var("NEAT_AI_DISCOVERY_PRELOAD_ALL") };
}

/// The cache-tuning recipe must not tell operators to export the no-op.
#[test]
fn cache_tuning_drops_the_no_op_preload_recipe() {
    assert!(
        !CACHE_TUNING.contains("NEAT_AI_DISCOVERY_PRELOAD_ALL=0"),
        "docs/CACHE_TUNING.md must not present `=0` as a way to force streaming"
    );
    assert!(
        CACHE_TUNING.contains("There is no environment variable to force LRU or Streaming"),
        "docs/CACHE_TUNING.md must keep the accurate statement about forcing tiers"
    );
}

// ============================================================================
// 4. Fuzz commands must pin the committed lockfile.
// ============================================================================

/// The README's fuzz commands must carry the same `--locked` flag CI uses, so a
/// contributor reproduces the pinned `fuzz/Cargo.lock` resolution.
#[test]
fn readme_fuzz_commands_match_ci_by_passing_locked() {
    assert!(
        CI_FUZZING_WORKFLOW.contains("cargo +nightly fuzz run --locked"),
        "the CI workflow reference must keep using --locked"
    );

    let unlocked: Vec<&str> = README
        .lines()
        .filter(|line| line.contains("cargo +nightly fuzz run"))
        .filter(|line| !line.contains("--locked"))
        .collect();
    assert!(
        unlocked.is_empty(),
        "README fuzz commands must pass --locked, found: {unlocked:?}"
    );
    assert!(
        README.contains("cargo +nightly fuzz run --locked"),
        "README must document at least one fuzz command"
    );
}

/// The README points contributors at `./scripts/fuzz-ci.sh` as the CI helper,
/// so the committed script — not just the proposed workflow file — must deliver
/// the `--locked` guarantee the README annotates (Issue #1992).
#[test]
fn committed_fuzz_helper_passes_locked_like_the_readme() {
    let unlocked: Vec<&str> = FUZZ_CI_SCRIPT
        .lines()
        .filter(|line| line.contains("cargo +nightly fuzz run"))
        .filter(|line| !line.contains("--locked"))
        .collect();
    assert!(
        unlocked.is_empty(),
        "scripts/fuzz-ci.sh must run fuzz targets with --locked so CI resolves the committed \
         fuzz/Cargo.lock the README promises, found: {unlocked:?}"
    );
    assert!(
        FUZZ_CI_SCRIPT.contains("cargo +nightly fuzz run --locked"),
        "scripts/fuzz-ci.sh must run at least one fuzz target"
    );
}
