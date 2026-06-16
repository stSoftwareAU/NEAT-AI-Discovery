//! Issue #1406: eliminate the cross-phase parquet reload.
//!
//! Focus selection (`rank_focus_neurons`) and the analysis phase
//! (`RecordCache::new_adaptive_*`, used by `analyze_all`) are two separate FFI
//! calls that historically each read the same parquet file from scratch. These
//! tests assert the file is fully read **at most once** when the two phases run
//! back-to-back on the same file: the analysis cache must serve the *same*
//! record allocation the focus phase loaded.
//!
//! The proof is allocation identity (`Arc::as_ptr`), which is deterministic and
//! safe under parallel test execution: a second full read would allocate fresh
//! `Vec`s with different pointers, so pointer equality can only hold when no
//! reload occurred.

use neat_ai_discovery::analysis::cache::{RecordCache, shared_records};
use neat_ai_discovery::focus::rank_focus_neurons;
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};
use std::sync::Arc;
use tempfile::NamedTempFile;

/// Build a minimal creature with one hidden and one output neuron wired from a
/// single input. Both selectable neurons (`h1`, `o1`) have recorded data.
fn create_creature() -> CreatureJson {
    CreatureJson {
        neurons: vec![
            NeuronJson {
                uuid: "h1".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "o1".to_string(),
                neuron_type: "output".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
        ],
        synapses: vec![
            SynapseJson {
                from_uuid: "i0".to_string(),
                to_uuid: "h1".to_string(),
                weight: 0.5,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "h1".to_string(),
                to_uuid: "o1".to_string(),
                weight: 0.5,
                synapse_type: None,
            },
        ],
        input: 1,
        output: 1,
    }
}

/// Records for `obs_count` observations across the input, hidden, and output
/// neurons. Records are intentionally written out of `obs_index` order to also
/// exercise the shared cache's sort.
fn create_records(obs_count: u32) -> Vec<DiscoverRecord> {
    let mut records = Vec::new();
    for obs in (0..obs_count).rev() {
        records.push(DiscoverRecord::new(
            obs,
            "i0".to_string(),
            Some(0.5),
            0.5,
            vec![0.1],
        ));
        records.push(DiscoverRecord::new(
            obs,
            "h1".to_string(),
            Some(0.4),
            0.3,
            vec![0.2],
        ));
        records.push(DiscoverRecord::new(
            obs,
            "o1".to_string(),
            Some(0.6),
            0.7,
            vec![0.15],
        ));
    }
    records
}

fn write_parquet(records: &[DiscoverRecord]) -> (NamedTempFile, String) {
    let temp_file = NamedTempFile::new().expect("create temp file");
    let path = temp_file.path().to_str().unwrap().to_string();
    write_records_to_parquet(&path, records).expect("write parquet");
    (temp_file, path)
}

#[test]
fn focus_then_analysis_reads_parquet_once() {
    let (_temp, path) = write_parquet(&create_records(4));
    let creature = create_creature();

    // Phase 1: focus selection loads the parquet and publishes it to the bridge.
    let stats = rank_focus_neurons(&path, &creature, None, None).expect("focus ranking succeeds");
    assert!(
        !stats.neurons.is_empty(),
        "focus ranking should rank the selectable neurons"
    );

    let bridged =
        shared_records::get(&path).expect("focus phase must populate the cross-phase bridge");
    let focus_o1_ptr = Arc::as_ptr(bridged.get("o1").expect("bridge holds o1 records"));
    let focus_h1_ptr = Arc::as_ptr(bridged.get("h1").expect("bridge holds h1 records"));

    // Phase 2: the analysis record cache must reuse the bridged allocations
    // rather than re-reading the file.
    let cache =
        RecordCache::new_adaptive_with_deadline(&path, None).expect("analysis cache builds");
    let analysis_o1 = cache.get("o1").expect("o1 records from analysis cache");
    let analysis_h1 = cache.get("h1").expect("h1 records from analysis cache");

    assert_eq!(
        Arc::as_ptr(&analysis_o1),
        focus_o1_ptr,
        "analysis must reuse the focus-loaded o1 records (single parquet read)"
    );
    assert_eq!(
        Arc::as_ptr(&analysis_h1),
        focus_h1_ptr,
        "analysis must reuse the focus-loaded h1 records (single parquet read)"
    );

    // The records served are still correct.
    assert_eq!(analysis_o1.len(), 4, "o1 has one record per observation");
    assert_eq!(analysis_h1.len(), 4, "h1 has one record per observation");

    // Once the analysis cache owns the records the bridge is released so it does
    // not pin the data across cycles.
    assert!(
        shared_records::get(&path).is_none(),
        "bridge is cleared after the analysis cache is built"
    );
}

#[test]
fn shared_cache_reuses_records_for_unchanged_file() {
    let (_temp, path) = write_parquet(&create_records(3));

    // Cold: nothing cached for this freshly created path.
    assert!(shared_records::get(&path).is_none());

    let first = shared_records::get_or_load_with_deadline(&path, None).expect("first load");
    let second = shared_records::get_or_load_with_deadline(&path, None).expect("second load");

    assert!(
        Arc::ptr_eq(&first, &second),
        "an unchanged file must be served from the cache, not reloaded"
    );

    // Records are grouped by neuron and sorted by obs_index.
    let o1 = first.get("o1").expect("o1 group present");
    assert_eq!(o1.len(), 3);
    let obs: Vec<u32> = o1.iter().map(|r| r.obs_index).collect();
    assert_eq!(obs, vec![0, 1, 2], "records sorted by obs_index");

    shared_records::clear();
    assert!(
        shared_records::get(&path).is_none(),
        "clear() releases the bridged records"
    );
}

#[test]
fn shared_cache_invalidates_when_file_changes() {
    let (_temp, path) = write_parquet(&create_records(2));
    let first = shared_records::get_or_load_with_deadline(&path, None).expect("first load");
    assert_eq!(first.get("o1").expect("o1").len(), 2);

    // Overwrite the same path with a different number of records: the size (and
    // typically mtime) changes, so the key no longer matches.
    write_records_to_parquet(&path, &create_records(5)).expect("rewrite parquet");
    assert!(
        shared_records::get(&path).is_none(),
        "a changed file must miss the cache"
    );

    let reloaded = shared_records::get_or_load_with_deadline(&path, None).expect("reload");
    assert!(
        !Arc::ptr_eq(&first, &reloaded),
        "a changed file must be reloaded into a fresh allocation"
    );
    assert_eq!(
        reloaded.get("o1").expect("o1").len(),
        5,
        "reload reflects the new file contents"
    );

    shared_records::clear();
}
