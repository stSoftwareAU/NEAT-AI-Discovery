//! Regression coverage for topology scans (Issue #2369).

use neat_ai_discovery::analysis::detection::topology::{
    detect_topology_issues, topology_issues_to_coordinated_candidates,
};
use neat_ai_discovery::analysis::detection::topology_cache::CreatureTopologyCache;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};
use serial_test::serial;

// Every test is `#[serial]`: one test sets the process-global cancellation flag,
// which would otherwise make the other detector calls return empty.

fn neuron(uuid: &str, kind: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: kind.to_string(),
        squash: "IDENTITY".to_string(),
        bias: 0.0,
    }
}

fn edge(from: &str, to: &str) -> SynapseJson {
    SynapseJson {
        from_uuid: from.to_string(),
        to_uuid: to.to_string(),
        weight: 0.5,
        synapse_type: None,
    }
}

fn records_for(creature: &CreatureJson) -> Vec<(String, Vec<DiscoverRecord>)> {
    creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "hidden")
        .map(|n| {
            let error = match n.uuid.as_str() {
                "h0" => 0.6,
                "h1" => 0.3,
                _ => 0.1,
            };
            let rows = (0..20)
                .map(|i| DiscoverRecord::new(i, n.uuid.clone(), Some(0.5), 0.5, vec![error]))
                .collect();
            (n.uuid.clone(), rows)
        })
        .collect()
}

fn both_branches_creature() -> CreatureJson {
    CreatureJson {
        neurons: vec![
            neuron("i0", "input"),
            neuron("i1", "input"),
            neuron("i2", "input"),
            neuron("h0", "hidden"),
            neuron("h1", "hidden"),
            neuron("h2", "hidden"),
            neuron("h3", "hidden"),
            neuron("h4", "hidden"),
            neuron("rich", "hidden"),
            neuron("out", "output"),
        ],
        synapses: vec![
            edge("i0", "h0"),
            edge("h0", "h1"),
            edge("h1", "h2"),
            edge("h2", "h3"),
            edge("h3", "h4"),
            edge("h4", "out"),
            edge("i0", "rich"),
            edge("i1", "rich"),
            edge("i2", "rich"),
        ],
        input: 3,
        output: 1,
    }
}

#[test]
#[serial]
fn local_and_cached_topology_preserve_long_path_and_imbalance_candidates() {
    let creature = both_branches_creature();
    let records = records_for(&creature);
    let locally_built = detect_topology_issues(&creature, &records, None);
    let topology = CreatureTopologyCache::new(&creature);
    let cached = detect_topology_issues(&creature, &records, Some(&topology));

    assert_eq!(locally_built.len(), cached.len());
    assert!(locally_built.iter().any(|c| c.issue_type == "long_path"));
    assert!(
        locally_built
            .iter()
            .any(|c| c.issue_type == "connectivity_imbalance")
    );
    let mut local_candidates: Vec<_> = locally_built
        .iter()
        .map(|c| {
            (
                c.neuron_uuid.as_str(),
                c.issue_type.as_str(),
                c.path_length,
                c.skip_target_uuid.as_deref(),
                c.estimated_improvement,
            )
        })
        .collect();
    let mut cached_candidates: Vec<_> = cached
        .iter()
        .map(|c| {
            (
                c.neuron_uuid.as_str(),
                c.issue_type.as_str(),
                c.path_length,
                c.skip_target_uuid.as_deref(),
                c.estimated_improvement,
            )
        })
        .collect();
    local_candidates.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
    cached_candidates.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
    assert_eq!(local_candidates, cached_candidates);
    assert!(
        locally_built
            .windows(2)
            .all(|pair| pair[0].estimated_improvement >= pair[1].estimated_improvement)
    );

    let h0_long_path = locally_built
        .iter()
        .find(|c| c.neuron_uuid == "h0" && c.issue_type == "long_path")
        .expect("h0 should receive a long-path candidate");
    assert_eq!(h0_long_path.path_length, 5);
    assert_eq!(h0_long_path.skip_target_uuid.as_deref(), Some("out"));
    assert!((h0_long_path.estimated_improvement - 0.006).abs() < 1e-6);

    let h0_imbalance = locally_built
        .iter()
        .find(|c| c.neuron_uuid == "h0" && c.issue_type == "connectivity_imbalance")
        .expect("h0 should receive an imbalance candidate");
    assert_ne!(h0_imbalance.add_source_uuid.as_deref(), Some("i0"));
    assert!(h0_imbalance.add_source_uuid.is_some());
    assert!(
        cached
            .iter()
            .filter(|c| c.neuron_uuid == "h0" && c.issue_type == "connectivity_imbalance")
            .all(|c| c.add_source_uuid.as_deref() != Some("i0"))
    );

    let converted =
        topology_issues_to_coordinated_candidates(&locally_built, &creature, Some(&topology));
    assert!(
        converted
            .iter()
            .all(|candidate| !candidate.operations.is_empty())
    );
}

#[test]
#[serial]
fn existing_output_connection_is_not_proposed_as_a_long_path() {
    let creature = CreatureJson {
        neurons: vec![
            neuron("i", "input"),
            neuron("h0", "hidden"),
            neuron("h1", "hidden"),
            neuron("h2", "hidden"),
            neuron("h3", "hidden"),
            neuron("out", "output"),
        ],
        synapses: vec![
            edge("i", "h0"),
            edge("h0", "h1"),
            edge("h1", "h2"),
            edge("h2", "h3"),
            edge("h3", "out"),
            edge("h0", "out"),
        ],
        input: 1,
        output: 1,
    };
    let records = records_for(&creature);
    let candidates = detect_topology_issues(&creature, &records, None);
    assert!(
        !candidates
            .iter()
            .any(|c| c.neuron_uuid == "h0" && c.issue_type == "long_path")
    );
}

#[test]
#[serial]
fn absent_inputs_or_outputs_do_not_create_invalid_sources_or_targets() {
    let creature = CreatureJson {
        neurons: vec![
            neuron("seed", "hidden"),
            neuron("a", "hidden"),
            neuron("b", "hidden"),
            neuron("c", "hidden"),
            neuron("d", "hidden"),
        ],
        synapses: vec![
            edge("seed", "a"),
            edge("seed", "b"),
            edge("seed", "c"),
            edge("seed", "d"),
            edge("a", "d"),
            edge("b", "d"),
            edge("c", "d"),
        ],
        input: 0,
        output: 0,
    };
    let records: Vec<_> = records_for(&creature)
        .into_iter()
        .filter(|(uuid, _)| uuid != "seed")
        .collect();
    let candidates = detect_topology_issues(&creature, &records, None);
    assert!(
        candidates
            .iter()
            .any(|c| c.issue_type == "connectivity_imbalance")
    );
    assert!(candidates.iter().all(|c| c.skip_target_uuid.is_none()));
    assert!(
        candidates
            .iter()
            .filter(|c| c.issue_type == "connectivity_imbalance")
            .all(|c| c.add_source_uuid.is_none())
    );
}

#[test]
#[serial]
fn existing_detector_stops_when_global_cancellation_is_requested() {
    let creature = both_branches_creature();
    let records = records_for(&creature);
    neat_ai_discovery::cancellation::reset_cancellation();
    neat_ai_discovery::cancellation::request_cancellation();

    let candidates = detect_topology_issues(&creature, &records, None);

    neat_ai_discovery::cancellation::reset_cancellation();
    assert!(
        candidates.is_empty(),
        "cancelled scans must not keep processing neurons"
    );
}
