//! Quick tool to examine parquet data for debugging
use neat_ai_discovery::parquet_format::read_records_from_parquet;

fn main() -> anyhow::Result<()> {
    let parquet_file = std::env::args()
        .nth(1)
        .expect("Usage: examine_parquet <file> <neuron_uuid>");
    let neuron_uuid = std::env::args()
        .nth(2)
        .expect("Usage: examine_parquet <file> <neuron_uuid>");

    println!("Reading records for {neuron_uuid} from {parquet_file}");

    let records = read_records_from_parquet(&parquet_file, &neuron_uuid)?;

    println!("Found {} records", records.len());

    if records.is_empty() {
        println!("No records found!");
        return Ok(());
    }

    // Sample first 5 records
    println!("\nFirst 5 records:");
    for (i, r) in records.iter().take(5).enumerate() {
        println!(
            "  [{i}] obs={}, value={:?}, activation={:.6}, errors={:?}",
            r.obs_index, r.value, r.activation, r.errors
        );
    }

    // Stats
    let mut total_error = 0.0f64;
    let mut error_count = 0usize;
    let mut has_value_count = 0usize;
    let mut activation_sum = 0.0f64;

    for r in &records {
        for &e in &r.errors {
            if e.is_finite() {
                total_error += (e as f64).abs();
                error_count += 1;
            }
        }
        if r.value.is_some() {
            has_value_count += 1;
        }
        activation_sum += r.activation as f64;
    }

    println!("\nStats:");
    println!(
        "  Records with value field: {has_value_count}/{}",
        records.len()
    );
    println!("  Total error samples: {error_count}");
    if error_count > 0 {
        println!(
            "  Mean absolute error: {:.6}",
            total_error / error_count as f64
        );
    }
    println!(
        "  Mean activation: {:.6}",
        activation_sum / records.len() as f64
    );

    // Error distribution
    let mut positive_errors = 0;
    let mut negative_errors = 0;
    let mut zero_errors = 0;
    for r in &records {
        for &e in &r.errors {
            if e > 0.001 {
                positive_errors += 1;
            } else if e < -0.001 {
                negative_errors += 1;
            } else {
                zero_errors += 1;
            }
        }
    }
    println!("\nError direction:");
    println!("  Positive (output should be higher): {positive_errors}");
    println!("  Negative (output should be lower): {negative_errors}");
    println!("  Near zero: {zero_errors}");

    Ok(())
}
