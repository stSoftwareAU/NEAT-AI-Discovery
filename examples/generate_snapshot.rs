//! Generate a visualisation snapshot JSON from a parquet file and creature JSON.
//!
//! Usage:
//!   cargo run --example `generate_snapshot` -- <`parquet_file`> <`creature_json`> <`output_json`> [`max_obs`]
//!
//! The inputs are a discovery-data parquet written by a discovery run (found
//! under the creature's `.discovery/<session-id>/` directory) and the creature
//! JSON that run analysed.
//!
//! Example:
//!   cargo run --release --example `generate_snapshot` -- \
//!     `.discovery/<session-id>/discovery_data.parquet` \
//!     creature.json \
//!     snapshot.json \
//!     1000

use neat_ai_discovery::CreatureJson;
use neat_ai_discovery::export::{ExportOptions, export_visualisation_snapshot};
use std::env;
use std::fs;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 4 {
        eprintln!(
            "Usage: {} <parquet_file> <creature_json> <output_json> [max_obs]",
            args[0]
        );
        eprintln!();
        eprintln!("Example:");
        eprintln!("  cargo run --release --example generate_snapshot -- \\");
        eprintln!("    .discovery/<session-id>/discovery_data.parquet \\");
        eprintln!("    creature.json \\");
        eprintln!("    snapshot.json \\");
        eprintln!("    1000");
        std::process::exit(1);
    }

    let parquet_file = &args[1];
    let creature_json_path = &args[2];
    let output_json = &args[3];
    let max_obs: Option<u32> = args.get(4).and_then(|s| s.parse().ok());

    println!("Reading creature from: {creature_json_path}");
    let creature_json_str =
        fs::read_to_string(creature_json_path).expect("Failed to read creature JSON file");

    let creature: CreatureJson =
        serde_json::from_str(&creature_json_str).expect("Failed to parse creature JSON");

    println!("  Neurons: {}", creature.neurons.len());
    println!("  Synapses: {}", creature.synapses.len());
    println!("  Inputs: {}, Outputs: {}", creature.input, creature.output);

    println!("Parquet file: {parquet_file}");
    println!("Output file: {output_json}");
    if let Some(max) = max_obs {
        println!("Max observations: {max}");
    } else {
        println!("Max observations: unlimited (may be slow/large)");
    }

    let options = ExportOptions {
        include_per_synapse_series: true,
        include_reconstruction_checks: true,
        max_obs,
        top_k_worst_samples: 5,
    };

    println!();
    println!("Generating snapshot...");

    match export_visualisation_snapshot(parquet_file, &creature, output_json, &options) {
        Ok(stats) => {
            println!();
            println!("✓ Snapshot written to: {output_json}");
            println!("  Observations: {}", stats.obs_count);
            println!("  Neurons: {}", stats.neuron_count);
            println!("  Synapses: {}", stats.synapse_count);
        }
        Err(e) => {
            eprintln!("✗ Error: {e:?}");
            std::process::exit(1);
        }
    }
}
