use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

fn extract_import_path(line: &str) -> Option<String> {
    // Example:
    // import { Softplus } from "./types/Softplus.ts";
    // import { HYPOT } from "../../deprecated/HYPOT.ts";
    let from_idx = line.find(" from ")?;
    let first_quote = line[from_idx..].find('"')? + from_idx;
    let rest = &line[(first_quote + 1)..];
    let second_quote = rest.find('"')?;
    Some(rest[..second_quote].to_string())
}

fn extract_activation_name(ts_source: &str) -> Option<String> {
    // Handles:
    // public static readonly NAME = "Softplus";
    // public static NAME = "Cosine";
    // public static readonly NAME = "BIPOLAR_SIGMOID";
    // public static NAME = "StdInverse";
    //
    // We intentionally keep this simple to avoid adding a regex crate.
    let needle = "NAME";
    for line in ts_source.lines() {
        if !line.contains(needle) || !line.contains('=') || !line.contains('"') {
            continue;
        }
        // Find the first occurrence of `NAME = "..."`.
        //
        // Important: do NOT use `?` in this loop, because a malformed line should not
        // cause the entire function to return `None`. We want to keep scanning until
        // we find a valid `NAME = "..."` declaration. (See regression test below.)
        let Some(name_pos) = line.find("NAME") else {
            continue;
        };

        // Only parse '=' that occurs *after* NAME. (Some comment lines may contain both,
        // but in a different order, eg `// x = "5" NAME`.)
        let after_name = &line[(name_pos + "NAME".len())..];
        let Some(eq_pos) = after_name.find('=') else {
            continue;
        };

        let after_eq = after_name[(eq_pos + 1)..].trim_start();
        if !after_eq.starts_with('"') {
            continue;
        }

        let after_quote = &after_eq[1..];
        let Some(end_quote) = after_quote.find('"') else {
            continue;
        };

        let value = after_quote[..end_quote].trim();
        if !value.is_empty() {
            return Some(value.to_string());
        }
    }
    None
}

fn activations_ts_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../NEAT-AI/src/methods/activations/Activations.ts")
}

#[test]
fn discovery_knows_all_neat_ai_activation_names() {
    let activations_ts = activations_ts_path();
    if !activations_ts.exists() {
        eprintln!(
            "Skipping: sibling NEAT-AI repo not found at {}",
            activations_ts.display()
        );
        return;
    }

    let base_dir = activations_ts
        .parent()
        .expect("Activations.ts must have a parent directory");
    let source = fs::read_to_string(&activations_ts).expect("Failed to read Activations.ts");

    let mut names: BTreeSet<String> = BTreeSet::new();

    // Gather activation files from imports and read each NAME constant.
    for line in source.lines() {
        let line = line.trim();
        if !line.starts_with("import ") || !line.contains(" from ") {
            continue;
        }
        let import_path = match extract_import_path(line) {
            Some(path) => path,
            None => continue,
        };

        // Only consider local TS files.
        if !import_path.ends_with(".ts") {
            continue;
        }

        // Resolve relative to Activations.ts directory.
        let file_path = base_dir.join(import_path);
        if !file_path.exists() {
            // Some imports may be resolved differently in Deno; ignore missing files.
            continue;
        }

        let file_source =
            fs::read_to_string(&file_path).expect("Failed to read imported activation file");
        if let Some(name) = extract_activation_name(&file_source) {
            names.insert(name);
        }
    }

    // NEAT-AI registry aliases (see `Activations.ts`).
    names.insert("CLIPPED".to_string());
    names.insert("RELU".to_string());
    names.insert("INVERSE".to_string());
    names.insert("SINUSOID".to_string());

    assert!(
        !names.is_empty(),
        "Expected to discover activation names from NEAT-AI imports, but found none"
    );

    let mut unknown: Vec<String> = Vec::new();
    let mut not_scalar_but_expected_scalar: Vec<String> = Vec::new();

    for name in &names {
        if !neat_ai_discovery::activations::is_known_squash_name(name) {
            unknown.push(name.clone());
            continue;
        }

        let is_aggregate = neat_ai_discovery::activations::is_aggregate_squash(name);
        let applied = neat_ai_discovery::activations::apply_scalar_squash(name, 0.123);

        if is_aggregate {
            assert!(
                applied.is_none(),
                "Aggregate squash {name} must not be treated as a scalar f(x)"
            );
        } else if applied.is_none() {
            not_scalar_but_expected_scalar.push(name.clone());
        }
    }

    if !unknown.is_empty() {
        panic!("Discovery does not recognise these NEAT-AI activation names: {unknown:?}");
    }

    if !not_scalar_but_expected_scalar.is_empty() {
        panic!(
            "Discovery recognises these squashes but cannot compute a scalar f(x) for them: {not_scalar_but_expected_scalar:?}"
        );
    }
}

#[test]
fn extract_activation_name_skips_malformed_lines_and_keeps_scanning() {
    // Regression test (24-Dec-2025):
    // A line may contain `NAME`, `=` and quotes but still be malformed for our parser,
    // for example when '=' appears before 'NAME'. The extractor must not stop early;
    // it should keep scanning until it finds a valid `NAME = "..."` declaration.
    let ts_source = r#"
// x = "5" NAME
public static readonly NAME = "Softplus";
"#;

    assert_eq!(
        extract_activation_name(ts_source),
        Some("Softplus".to_string())
    );
}
