//! Issue #1923: per-neuron mean absolute activation, aggregated in one
//! projected Parquet pass.
//!
//! `remove-low-impact` ranked on `meanActivation: 0.0` because the structural
//! path had no way to measure activations without decoding whole records — the
//! cost Issue #1766 removed from focus. These tests pin the cheap replacement:
//! a two-column streaming aggregation that returns a mean per neuron, and that
//! **omits** any neuron it could not measure rather than reporting `0.0`.

use neat_ai_discovery::parquet_format::{
    read_mean_abs_activation_by_neuron, write_records_to_parquet,
};
use neat_ai_discovery::types::DiscoverRecord;
use std::collections::HashSet;
use tempfile::NamedTempFile;

/// Write `records` to a temporary parquet file, returning the handle (which
/// must outlive the read) and its path.
fn parquet_with(records: &[DiscoverRecord]) -> (NamedTempFile, String) {
    let file = NamedTempFile::new().expect("temp file");
    let path = file.path().to_str().expect("utf-8 path").to_string();
    write_records_to_parquet(&path, records).expect("write parquet");
    (file, path)
}

fn record(obs: u32, uuid: &str, activation: f32) -> DiscoverRecord {
    DiscoverRecord::new(obs, uuid.to_string(), Some(0.5), activation, vec![0.1])
}

#[test]
fn means_the_absolute_activation_of_each_wanted_neuron() {
    let (_file, path) = parquet_with(&[
        record(0, "n-a", 0.2),
        record(1, "n-a", -0.4),
        record(2, "n-a", 0.6),
        record(0, "n-b", 4.0),
        record(1, "n-b", 6.0),
    ]);

    let wanted: HashSet<&str> = ["n-a", "n-b"].into_iter().collect();
    let summaries = read_mean_abs_activation_by_neuron(&path, &wanted, None).expect("summaries");

    let a = summaries.get("n-a").expect("n-a measured");
    assert!(
        (a.mean_abs_activation - 0.4).abs() < 1e-6,
        "mean |activation| of 0.2/-0.4/0.6 is 0.4, got {}",
        a.mean_abs_activation
    );
    assert_eq!(a.sample_count, 3);

    let b = summaries.get("n-b").expect("n-b measured");
    assert!((b.mean_abs_activation - 5.0).abs() < 1e-6);
    assert_eq!(b.sample_count, 2);
}

#[test]
fn ignores_neurons_the_caller_did_not_ask_for() {
    let (_file, path) = parquet_with(&[record(0, "n-a", 0.5), record(0, "n-other", 9.0)]);

    let wanted: HashSet<&str> = ["n-a"].into_iter().collect();
    let summaries = read_mean_abs_activation_by_neuron(&path, &wanted, None).expect("summaries");

    assert_eq!(summaries.len(), 1, "only the wanted neuron: {summaries:?}");
    assert!(summaries.contains_key("n-a"));
}

/// A neuron with no rows must be **absent**, not present with a zero mean:
/// "never measured" and "measured as inactive" rank differently, and conflating
/// them is exactly the defect this issue fixes.
#[test]
fn omits_a_neuron_with_no_recorded_rows() {
    let (_file, path) = parquet_with(&[record(0, "n-a", 0.5)]);

    let wanted: HashSet<&str> = ["n-a", "n-missing"].into_iter().collect();
    let summaries = read_mean_abs_activation_by_neuron(&path, &wanted, None).expect("summaries");

    assert!(summaries.contains_key("n-a"));
    assert!(
        !summaries.contains_key("n-missing"),
        "an unmeasured neuron must be omitted, not reported as 0.0: {summaries:?}"
    );
}

/// Non-finite activations must not poison the mean — and a neuron whose every
/// sample is non-finite has no usable measurement at all.
#[test]
fn skips_non_finite_activations_and_omits_wholly_non_finite_neurons() {
    let (_file, path) = parquet_with(&[
        record(0, "n-mixed", f32::NAN),
        record(1, "n-mixed", 0.25),
        record(2, "n-mixed", f32::INFINITY),
        record(0, "n-nan", f32::NAN),
    ]);

    let wanted: HashSet<&str> = ["n-mixed", "n-nan"].into_iter().collect();
    let summaries = read_mean_abs_activation_by_neuron(&path, &wanted, None).expect("summaries");

    let mixed = summaries.get("n-mixed").expect("n-mixed measured");
    assert!(
        (mixed.mean_abs_activation - 0.25).abs() < 1e-6,
        "only the finite sample counts, got {}",
        mixed.mean_abs_activation
    );
    assert_eq!(mixed.sample_count, 1);

    assert!(
        !summaries.contains_key("n-nan"),
        "a wholly non-finite neuron yields no measurement: {summaries:?}"
    );
}

#[test]
fn an_empty_request_reads_nothing_and_succeeds() {
    let wanted: HashSet<&str> = HashSet::new();
    let summaries =
        read_mean_abs_activation_by_neuron("/nonexistent/never-opened.parquet", &wanted, None)
            .expect("an empty request must not open the file");
    assert!(summaries.is_empty());
}

/// Fail loud: an unreadable file is an error, never an empty (and therefore
/// "everything is unmeasured") result.
#[test]
fn a_missing_file_is_an_error_not_an_empty_result() {
    let wanted: HashSet<&str> = ["n-a"].into_iter().collect();
    let err = read_mean_abs_activation_by_neuron("/nonexistent/missing.parquet", &wanted, None)
        .expect_err("a missing parquet must surface as an error");
    assert!(
        err.to_string().contains("missing.parquet"),
        "the error must name the file: {err}"
    );
}

/// An already-expired deadline aborts before any decoding, rather than
/// returning a partial scan the caller would read as complete.
#[test]
fn an_expired_deadline_aborts_rather_than_returning_partial_results() {
    let (_file, path) = parquet_with(&[record(0, "n-a", 0.5)]);

    let wanted: HashSet<&str> = ["n-a"].into_iter().collect();
    let expired = std::time::SystemTime::now() - std::time::Duration::from_secs(60);
    let err = read_mean_abs_activation_by_neuron(&path, &wanted, Some(expired))
        .expect_err("an expired deadline must abort");
    assert!(
        err.to_string().contains("deadline"),
        "the error must name the deadline: {err}"
    );
}
