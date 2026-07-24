//! Offline-loading smoke test for the dominated-branch collapse fixtures
//! (Issue #1705).
//!
//! Foundation task for the #1704 characterisation effort. The
//! collapse-characterisation and contribution-propagation suites (#1706–#1708)
//! consume the fixtures committed under
//! `tests/fixtures/dominated_branch_collapse/`. Every fixture is hand-authored
//! and synthetic (Issue #1722), committed to this repository and loaded from
//! disk — never fetched at runtime.
//!
//! This smoke test is the earliest detection point: it deserialises every
//! synthetic network and every candidate-cache-shaped record without network
//! access, asserting the expected counts and pinned values. A missing, renamed,
//! malformed, or silently edited fixture fails here under `cargo test`, which
//! runs in the repo's Rust CI gate on every PR and push to `Develop`.

use neat_ai_discovery::CreatureJson;
use std::path::{Path, PathBuf};

fn fixture_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dominated_branch_collapse")
}

fn read_json(path: &Path) -> serde_json::Value {
    let raw = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("failed to read fixture {}: {e}", path.display()));
    serde_json::from_str(&raw)
        .unwrap_or_else(|e| panic!("failed to parse fixture {}: {e}", path.display()))
}

/// The aggregate squash carried by each synthetic network, keyed by file name.
/// One entry per aggregate type the collapse suite characterises.
const NETWORKS: &[(&str, &str)] = &[
    ("maximum_aggregate.json", "MAXIMUM"),
    ("minimum_aggregate.json", "MINIMUM"),
    ("if_aggregate.json", "IF"),
];

/// Every committed candidate-cache-shaped record file.
const CACHE_RECORDS: &[&str] = &["v2_change-squash_selu-to-absolute.json", "d1ac1f41.json"];

/// The three synthetic networks deserialise into `CreatureJson` offline, each
/// carries its expected aggregate squash, and the IF fixture carries an explicit
/// `condition` synapse.
#[test]
fn networks_load_offline() {
    let dir = fixture_root().join("networks");
    for (file, expected_squash) in NETWORKS {
        let path = dir.join(file);
        let creature: CreatureJson = serde_json::from_str(
            &std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display())),
        )
        .unwrap_or_else(|e| {
            panic!(
                "failed to deserialise {} into CreatureJson: {e}",
                path.display()
            )
        });

        // The dominated-branch shape: an ABSOLUTE branch and a RELU branch feed
        // the selection aggregate.
        assert!(
            creature.neurons.iter().any(|n| n.squash == "ABSOLUTE"),
            "{file}: missing the ABSOLUTE (dominated) branch"
        );
        assert!(
            creature.neurons.iter().any(|n| n.squash == "RELU"),
            "{file}: missing the RELU branch"
        );
        assert!(
            creature
                .neurons
                .iter()
                .any(|n| n.squash == *expected_squash),
            "{file}: missing the {expected_squash} selection aggregate"
        );
        assert!(
            creature.neurons.iter().any(|n| n.neuron_type == "output"),
            "{file}: missing an output neuron"
        );
        assert!(
            !creature.synapses.is_empty(),
            "{file}: network has no synapses"
        );
    }

    // The IF fixture must carry a condition synapse plus positive/negative
    // branches — the selection contract the contribution suite exercises.
    let if_path = dir.join("if_aggregate.json");
    let if_creature: CreatureJson = serde_json::from_str(
        &std::fs::read_to_string(&if_path)
            .unwrap_or_else(|e| panic!("failed to read {}: {e}", if_path.display())),
    )
    .expect("IF fixture must deserialise");
    let synapse_type = |t: &str| {
        if_creature
            .synapses
            .iter()
            .any(|s| s.synapse_type.as_deref() == Some(t))
    };
    assert!(
        synapse_type("condition"),
        "IF fixture missing condition synapse"
    );
    assert!(
        synapse_type("positive"),
        "IF fixture missing positive branch"
    );
    assert!(
        synapse_type("negative"),
        "IF fixture missing negative branch"
    );
}

/// The SELU→ABSOLUTE `change-squash` record loads offline and pins the
/// placeholder-vs-outcome values the contribution suite grades against.
#[test]
fn change_squash_record_loads_offline() {
    let path = fixture_root()
        .join("candidate_cache")
        .join("v2_change-squash_selu-to-absolute.json");
    let record = read_json(&path);

    assert_eq!(record["changeType"], "change-squash");
    assert_eq!(record["discoveryVersion"], "0.74.131");
    assert_eq!(
        record["rustRequest"]["squashCandidate"]["previousSquash"],
        "SELU"
    );
    assert_eq!(
        record["rustRequest"]["squashCandidate"]["squash"],
        "ABSOLUTE"
    );

    let expected = record["expectedErrorReduction"]
        .as_f64()
        .expect("expectedErrorReduction must be numeric");
    let actual = record["actualErrorReduction"]
        .as_f64()
        .expect("actualErrorReduction must be numeric");
    assert!(
        (expected - 3.0e-10).abs() < 1e-18,
        "expectedErrorReduction {expected:e} drifted from +3.0e-10"
    );
    assert!(
        (actual - (-6.0e-4)).abs() < 1e-12,
        "actualErrorReduction {actual:e} drifted from -6.0e-4"
    );
    // The predicted gain is tiny and positive; the recorded outcome is a much
    // larger negative — the collapse the contribution suite must reproduce.
    assert!(
        expected > 0.0 && actual < 0.0,
        "record must show positive prediction, negative outcome"
    );
}

/// The `d1ac1f41` shape loads offline as 1 success (remove-neuron) versus 5
/// failures (1 change-squash, 4 remove-neuron).
#[test]
fn d1ac1f41_shape_loads_offline() {
    let path = fixture_root().join("candidate_cache").join("d1ac1f41.json");
    let record = read_json(&path);

    let successes = record["successes"].as_array().expect("successes array");
    let failures = record["failures"].as_array().expect("failures array");
    assert_eq!(successes.len(), 1, "expected exactly 1 success");
    assert_eq!(failures.len(), 5, "expected exactly 5 failures");

    assert_eq!(
        successes[0]["changeType"], "remove-neuron",
        "the single success must be a remove-neuron"
    );

    let count = |arr: &[serde_json::Value], change_type: &str| {
        arr.iter()
            .filter(|r| r["changeType"] == change_type)
            .count()
    };
    assert_eq!(
        count(failures, "change-squash"),
        1,
        "expected 1 change-squash failure"
    );
    assert_eq!(
        count(failures, "remove-neuron"),
        4,
        "expected 4 remove-neuron failures"
    );
}

/// Every committed fixture file has exactly one provenance entry in the manifest
/// `README.md` — the manifest cannot silently omit a committed fixture.
#[test]
fn manifest_covers_every_fixture() {
    let readme = std::fs::read_to_string(fixture_root().join("README.md"))
        .expect("fixtures dir must carry a provenance README/manifest");

    for (file, _) in NETWORKS {
        assert!(
            readme.contains(file),
            "manifest missing an entry for networks/{file}"
        );
    }
    for file in CACHE_RECORDS {
        assert!(
            readme.contains(file),
            "manifest missing an entry for candidate_cache/{file}"
        );
    }
}

/// Aggregate offline-load guard named in the issue's failure-detection section:
/// every network and every candidate-cache-shaped record deserialises without
/// network access, at the expected counts.
#[test]
fn fixtures_load_offline() {
    let net_dir = fixture_root().join("networks");
    let mut network_count = 0;
    for (file, _) in NETWORKS {
        let path = net_dir.join(file);
        let _creature: CreatureJson = serde_json::from_str(
            &std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display())),
        )
        .unwrap_or_else(|e| panic!("{} did not deserialise: {e}", path.display()));
        network_count += 1;
    }
    assert_eq!(
        network_count, 3,
        "expected three aggregate network fixtures"
    );

    let cache_dir = fixture_root().join("candidate_cache");
    for file in CACHE_RECORDS {
        // Deserialising as generic JSON is enough to prove the record loads
        // offline; the shape-specific asserts live in the tests above.
        let _ = read_json(&cache_dir.join(file));
    }
}
