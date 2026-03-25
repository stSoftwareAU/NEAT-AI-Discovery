//! Scenario test: discovery finds a missing fan-in synapse (hidden→hidden) (Issue #928).
//!
//! Topology (whole creature — fan-in of 3 to hidden-A):
//! ```text
//! input-0 ──(0.7)──▶ hidden-B (RELU) ──(0.5)──▶ hidden-A   ← fan-in from hidden neuron
//! input-0 ──(0.8)──▶ hidden-A (TANH)
//! input-1 ──(0.6)──▶ hidden-A
//! hidden-B ──(0.3)──▶ output-0 (IDENTITY)
//! hidden-A ──(1.0)──▶ output-0
//! ```
//!
//! Crippled creature (hidden-B → hidden-A removed, fan-in reduced to 2):
//! ```text
//! input-0 ──(0.7)──▶ hidden-B (RELU) ──(0.3)──▶ output-0
//! input-0 ──(0.8)──▶ hidden-A (TANH) ──(1.0)──▶ output-0
//! input-1 ──(0.6)──▶ hidden-A
//! ```
//!
//! Discovery should identify the missing hidden-B → hidden-A synapse and return
//! an `addSynapse` candidate reconnecting them.

#![allow(clippy::cast_precision_loss)]

use neat_ai_discovery::analysis::GpuAnalyzer;
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson, analyze_parallel_internal};

/// Build the "crippled" creature (hidden-B → hidden-A synapse removed).
fn crippled_creature() -> CreatureJson {
    CreatureJson {
        input: 2,
        output: 1,
        neurons: vec![
            NeuronJson {
                uuid: "hidden-B".into(),
                neuron_type: "hidden".into(),
                squash: "ReLU".into(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "hidden-A".into(),
                neuron_type: "hidden".into(),
                squash: "TANH".into(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "output-0".into(),
                neuron_type: "output".into(),
                squash: "IDENTITY".into(),
                bias: 0.0,
            },
        ],
        synapses: vec![
            SynapseJson {
                from_uuid: "input-0".into(),
                to_uuid: "hidden-B".into(),
                weight: 0.7,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "input-0".into(),
                to_uuid: "hidden-A".into(),
                weight: 0.8,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "input-1".into(),
                to_uuid: "hidden-A".into(),
                weight: 0.6,
                synapse_type: None,
            },
            // hidden-B → hidden-A synapse REMOVED (the missing fan-in connection)
            SynapseJson {
                from_uuid: "hidden-B".into(),
                to_uuid: "output-0".into(),
                weight: 0.3,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "hidden-A".into(),
                to_uuid: "output-0".into(),
                weight: 1.0,
                synapse_type: None,
            },
        ],
    }
}

/// Generate discovery records from the CRIPPLED creature (without the
/// hidden-B → hidden-A synapse) but with error signals derived from the
/// FULL creature's expected output.
fn generate_records(num_observations: u32) -> Vec<DiscoverRecord> {
    let mut records = Vec::new();
    for obs in 0..num_observations {
        let t = obs as f32 / num_observations as f32;
        let x0 = t * 2.0 - 1.0; // [-1, 1]
        let x1 = (t * 3.7).sin() * 0.8; // Varied pattern in [-0.8, 0.8]

        // --- hidden-B: ReLU(input-0 * 0.7) ---
        let hb_pre = x0 * 0.7;
        let hb_act = hb_pre.max(0.0);

        // --- hidden-A (CRIPPLED): TANH(input-0 * 0.8 + input-1 * 0.6) ---
        // Missing: hidden-B * 0.5
        let ha_pre_crippled = x0 * 0.8 + x1 * 0.6;
        let ha_act_crippled = ha_pre_crippled.tanh();

        // --- hidden-A (FULL): TANH(input-0 * 0.8 + input-1 * 0.6 + hidden-B * 0.5) ---
        let ha_pre_full = x0 * 0.8 + x1 * 0.6 + hb_act * 0.5;
        let ha_act_full = ha_pre_full.tanh();

        // --- output-0 (CRIPPLED): hidden-B * 0.3 + hidden-A_crippled * 1.0 ---
        let out_crippled = hb_act * 0.3 + ha_act_crippled * 1.0;
        // --- output-0 (FULL): hidden-B * 0.3 + hidden-A_full * 1.0 ---
        let out_full = hb_act * 0.3 + ha_act_full * 1.0;

        // Error = what the full creature would produce minus the crippled output
        let out_err = out_full - out_crippled;

        // hidden-A error: how much hidden-A is wrong due to the missing fan-in synapse
        let ha_err = ha_act_full - ha_act_crippled;

        // hidden-B error: propagated from output (small, since hidden-B itself is correct)
        let hb_err = out_err * 0.1;

        records.push(DiscoverRecord::new(
            obs,
            "hidden-B".into(),
            Some(hb_pre),
            hb_act,
            vec![hb_err],
        ));
        records.push(DiscoverRecord::new(
            obs,
            "hidden-A".into(),
            Some(ha_pre_crippled),
            ha_act_crippled,
            vec![ha_err],
        ));
        records.push(DiscoverRecord::new(
            obs,
            "output-0".into(),
            Some(out_crippled),
            out_crippled,
            vec![out_err],
        ));
    }
    records
}

/// Run the full discovery pipeline and return the parsed JSON output.
fn run_analysis(creature: &CreatureJson, records: &[DiscoverRecord]) -> serde_json::Value {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let parquet_file = temp_dir
        .path()
        .join("records.parquet")
        .to_str()
        .expect("valid UTF-8 path")
        .to_string();

    write_records_to_parquet(&parquet_file, records).expect("write parquet");

    let input_json = serde_json::json!({
        "parquetFile": parquet_file,
        "creature": creature,
        "focusNeurons": ["output-0", "hidden-A"],
        "maxSynapseCandidates": 32,
        "maxNeuronCandidates": 32,
        "randomSeed": 42
    })
    .to_string();

    let output_json = analyze_parallel_internal(&input_json).expect("analysis should return JSON");
    let output: serde_json::Value =
        serde_json::from_str(&output_json).expect("output should be valid JSON");

    assert_eq!(
        output["success"],
        true,
        "Analysis failed: {}",
        output["error"].as_str().unwrap_or("unknown")
    );

    output
}

/// Check whether any coordinated structural candidate contains an addSynapse
/// operation from `from_uuid` to `to_uuid`.
fn find_add_synapse_in_coordinated(
    coordinated: &[serde_json::Value],
    from_uuid: &str,
    to_uuid: &str,
) -> Option<(serde_json::Value, serde_json::Value)> {
    for group in coordinated {
        if let Some(ops) = group["operations"].as_array() {
            for op in ops {
                if op["type"].as_str() == Some("addSynapse")
                    && op["fromNeuronUuid"].as_str() == Some(from_uuid)
                    && op["toNeuronUuid"].as_str() == Some(to_uuid)
                {
                    return Some((group.clone(), op.clone()));
                }
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Test: Discovery finds the missing hidden-B → hidden-A fan-in synapse
// ---------------------------------------------------------------------------

/// The discovery engine should identify the missing hidden-B → hidden-A
/// fan-in synapse in the crippled creature. The candidate may appear as a
/// direct helpful synapse or as an addSynapse operation within a coordinated
/// structural candidate.
#[test]
fn issue_928_discovers_missing_fan_in_synapse() {
    if !GpuAnalyzer::gpu_is_available() {
        eprintln!("Skipping: no GPU available");
        return;
    }

    let creature = crippled_creature();
    let records = generate_records(128);
    let output = run_analysis(&creature, &records);

    // --- Check helpfulSynapses for a direct hidden-B → hidden-A candidate ---
    let empty = vec![];
    let helpful = output["helpfulSynapses"].as_array().unwrap_or(&empty);

    let direct_candidate = helpful.iter().find(|c| {
        c["fromNeuronUuid"].as_str() == Some("hidden-B")
            && c["toNeuronUuid"].as_str() == Some("hidden-A")
    });

    // --- Check coordinatedStructuralCandidates for an addSynapse operation ---
    let coordinated = output["coordinatedStructuralCandidates"]
        .as_array()
        .unwrap_or(&empty);

    let coordinated_candidate =
        find_add_synapse_in_coordinated(coordinated, "hidden-B", "hidden-A");

    // At least one pathway should discover the missing fan-in synapse
    assert!(
        direct_candidate.is_some() || coordinated_candidate.is_some(),
        "Expected an addSynapse candidate from hidden-B → hidden-A in either \
         helpfulSynapses or coordinatedStructuralCandidates. \
         helpfulSynapses count: {}, coordinated count: {}",
        helpful.len(),
        coordinated.len()
    );

    // Verify properties of the discovered candidate
    if let Some(candidate) = direct_candidate {
        let weight = candidate["weight"]
            .as_f64()
            .expect("weight should be a number");
        assert!(
            weight > 0.0,
            "Expected positive weight for hidden-B → hidden-A synapse, got {weight}"
        );

        let impact = candidate["targetNeuronImpact"]
            .as_f64()
            .expect("targetNeuronImpact should be a number");
        assert!(
            impact > 0.0,
            "Expected positive targetNeuronImpact, got {impact}"
        );

        let score_gain = candidate["expectedCreatureScoreGain"]
            .as_f64()
            .expect("expectedCreatureScoreGain should be a number");
        assert!(
            score_gain > 0.0,
            "Expected positive expectedCreatureScoreGain, got {score_gain}"
        );

        eprintln!(
            "Issue #928 PASS (helpfulSynapses): hidden-B → hidden-A weight={weight:.4}, \
             impact={impact:.4}, scoreGain={score_gain:.6}"
        );
    } else if let Some((group, op)) = coordinated_candidate {
        let weight = op["weight"].as_f64().expect("weight should be a number");
        assert!(
            weight > 0.0,
            "Expected positive weight for hidden-B → hidden-A addSynapse operation, got {weight}"
        );

        let group_gain = group["expectedCreatureScoreGain"]
            .as_f64()
            .expect("group expectedCreatureScoreGain should be a number");
        assert!(
            group_gain > 0.0,
            "Expected positive expectedCreatureScoreGain for the coordinated group, \
             got {group_gain}"
        );

        eprintln!(
            "Issue #928 PASS (coordinatedStructural): hidden-B → hidden-A weight={:.4}, \
             groupScoreGain={:.6}, comment={}",
            weight,
            group_gain,
            group["comment"].as_str().unwrap_or("none")
        );
    }
}

// ---------------------------------------------------------------------------
// Test: The candidate correctly identifies source and target neurons
// ---------------------------------------------------------------------------

/// Verify that the discovered candidate identifies hidden-B as the source
/// neuron and hidden-A as the target neuron (not reversed).
#[test]
fn issue_928_candidate_identifies_correct_source_and_target() {
    if !GpuAnalyzer::gpu_is_available() {
        eprintln!("Skipping: no GPU available");
        return;
    }

    let creature = crippled_creature();
    let records = generate_records(128);
    let output = run_analysis(&creature, &records);

    let empty = vec![];
    let helpful = output["helpfulSynapses"].as_array().unwrap_or(&empty);
    let coordinated = output["coordinatedStructuralCandidates"]
        .as_array()
        .unwrap_or(&empty);

    // The candidate must go from hidden-B to hidden-A (forward direction)
    let has_forward = helpful.iter().any(|c| {
        c["fromNeuronUuid"].as_str() == Some("hidden-B")
            && c["toNeuronUuid"].as_str() == Some("hidden-A")
    }) || find_add_synapse_in_coordinated(coordinated, "hidden-B", "hidden-A")
        .is_some();

    assert!(
        has_forward,
        "Expected a forward hidden-B → hidden-A candidate (not reversed). \
         helpfulSynapses: {:?}, coordinated addSynapse operations: {:?}",
        helpful
            .iter()
            .map(|c| format!(
                "{} → {}",
                c["fromNeuronUuid"].as_str().unwrap_or("?"),
                c["toNeuronUuid"].as_str().unwrap_or("?")
            ))
            .collect::<Vec<_>>(),
        coordinated
            .iter()
            .filter_map(|g| g["operations"].as_array())
            .flatten()
            .filter(|op| op["type"].as_str() == Some("addSynapse"))
            .map(|op| format!(
                "{} → {}",
                op["fromNeuronUuid"].as_str().unwrap_or("?"),
                op["toNeuronUuid"].as_str().unwrap_or("?")
            ))
            .collect::<Vec<_>>()
    );

    eprintln!("Issue #928: correctly identified hidden-B as source and hidden-A as target");
}

// ---------------------------------------------------------------------------
// Test: All returned candidates have positive expected improvement
// ---------------------------------------------------------------------------

/// Verify that all candidates returned by the pipeline have positive expected
/// score gain (the project mission: only return improvements).
#[test]
fn issue_928_all_candidates_have_positive_improvement() {
    if !GpuAnalyzer::gpu_is_available() {
        eprintln!("Skipping: no GPU available");
        return;
    }

    let creature = crippled_creature();
    let records = generate_records(128);
    let output = run_analysis(&creature, &records);

    let empty = vec![];

    // Check helpful synapses
    if let Some(helpful) = output["helpfulSynapses"].as_array() {
        for (i, c) in helpful.iter().enumerate() {
            if let Some(gain) = c["expectedCreatureScoreGain"].as_f64() {
                assert!(
                    gain > 0.0,
                    "helpfulSynapses[{i}] expected positive score gain, got {gain}"
                );
            }
        }
    }

    // Check helpful neurons
    if let Some(neurons) = output["helpfulNeurons"].as_array() {
        for (i, c) in neurons.iter().enumerate() {
            if let Some(gain) = c["expectedCreatureScoreGain"].as_f64() {
                assert!(
                    gain > 0.0,
                    "helpfulNeurons[{i}] expected positive score gain, got {gain}"
                );
            }
        }
    }

    // Check coordinated structural candidates
    let coordinated = output["coordinatedStructuralCandidates"]
        .as_array()
        .unwrap_or(&empty);
    for (i, group) in coordinated.iter().enumerate() {
        if let Some(gain) = group["expectedCreatureScoreGain"].as_f64() {
            assert!(
                gain > 0.0,
                "coordinatedStructural[{i}] expected positive gain, got {gain}"
            );
        }
    }

    eprintln!(
        "Issue #928: all candidates have positive improvement \
         (helpful={}, neurons={}, coordinated={})",
        output["helpfulSynapses"]
            .as_array()
            .map_or(0, std::vec::Vec::len),
        output["helpfulNeurons"]
            .as_array()
            .map_or(0, std::vec::Vec::len),
        coordinated.len()
    );
}
