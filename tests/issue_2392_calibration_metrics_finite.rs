use neat_ai_discovery::get_calibration_summary_internal;
use serde_json::{Value, json};

#[test]
fn extreme_finite_history_returns_numeric_metrics() {
    let histories = [
        r#"{"neurons":{},"calibration":{"observations":{"saturation::addSynapse":[{"predicted":1.7e308,"actual":-1.7e308},{"predicted":-1.7e308,"actual":1.7e308}]}}}"#,
        r#"{"neurons":{},"calibration":{"observations":{"saturation::addSynapse":[{"predicted":1e-9,"actual":1e308},{"predicted":1e-9,"actual":-1e308}]}}}"#,
        r#"{"neurons":{},"calibration":{"observations":{"saturation::addSynapse":[{"predicted":1e308,"actual":0.0},{"predicted":1e308,"actual":0.0},{"predicted":1e308,"actual":0.0}]}}}"#,
    ];

    for history in histories {
        let request = json!({"discoveryHistory": history}).to_string();
        let response = get_calibration_summary_internal(&request)
            .expect("calibration summary FFI must return JSON");
        let response: Value = serde_json::from_str(&response).expect("valid FFI JSON");
        assert_eq!(response["success"], true, "{response}");

        let entry = &response["calibrationSummary"][0];
        for field in ["meanAbsoluteError", "bias", "calibrationFactor"] {
            assert!(
                entry[field].is_number(),
                "FFI field {field} must be a JSON number, got {} in {response}",
                entry[field]
            );
        }
        let factor = entry["calibrationFactor"]
            .as_f64()
            .expect("calibration factor is numeric");
        assert!(
            (0.1..=10.0).contains(&factor),
            "factor outside clamp: {factor}"
        );
    }
}
