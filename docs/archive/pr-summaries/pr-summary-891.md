## Summary

Add per-candidate-type prediction calibration scaling to correct systematic overestimation in `expected_creature_score_gain`. Production discovery cache reveals predictions overestimate actual outcomes by 100–10,000×, with the magnitude varying by candidate type — making cross-type comparisons unreliable. Closes #891.

Three calibration constants are applied post-pessimism-discount:
- `SYNAPSE_PREDICTION_CALIBRATION` = 0.001 (corrects ~1,000× overestimation)
- `NEURON_PREDICTION_CALIBRATION` = 0.01 (corrects ~100× overestimation)
- `COORDINATED_PREDICTION_CALIBRATION` = 0.0001 (corrects ~10,000× overestimation)

## Evidence

Calibration factors are derived from production discovery-cache data comparing predicted vs actual score gains:

| Candidate Type | Predicted Gain | Actual Gain | Overestimation | Calibration Factor |
|----------------|---------------|-------------|----------------|-------------------|
| Add-Neuron     | 0.003–0.01    | 1e-7 to 3e-6 | 100–10,000×  | 0.01              |
| Add-Synapse    | 0.001–0.01    | often negative | ~1,000×+    | 0.001             |
| Coordinated    | 0.001–0.01    | ~2.2e-14    | ~10,000×+      | 0.0001            |

After calibration, cross-type ranking is corrected: a neuron prediction of 0.003 (calibrated to 0.00003) correctly ranks higher than a synapse prediction of 0.01 (calibrated to 0.00001).

## Test Plan

- Added compile-time const assertions verifying calibration factor ranges and relative ordering
- `test_apply_prediction_calibration_positive_gain` — verifies correct multiplicative scaling
- `test_apply_prediction_calibration_zero_gain` — verifies zero preservation
- `test_apply_prediction_calibration_negative_gain` — verifies sign preservation
- `test_calibration_improves_cross_type_ranking` — verifies cross-type ranking correction
- `test_calibration_preserves_small_gains` — verifies no precision loss for tiny gains
- All 137 existing tests continue to pass
- `quality.sh` passes cleanly
