// Unit tests for `analysis::implementation`.
//
// Kept as unit tests (not integration tests) so we can test private helpers
// without changing visibility.

    use super::*;
    use crate::analysis::ACTIVATION_SPECS;

    // ==================== GPU Tier Detection Tests ====================

    #[test]
    fn impact_discounting_uses_activation_based_selection_stats_for_minimum() -> Result<()> {
        // This is a targeted regression test for impact discounting consistency:
        // - MINIMUM/MAXIMUM/IF impacts should use activation-based win probabilities when
        //   recorded activations are available (via RecordCache).
        //
        // Without activation stats, MINIMUM uses a conservative 1/N model, which can
        // under-estimate impact and cause downstream discounting to be too aggressive.

        let creature = crate::CreatureJson {
            neurons: vec![
                crate::NeuronJson {
                    uuid: "hidden-a".to_string(),
                    neuron_type: "hidden".to_string(),
                    squash: "IDENTITY".to_string(),
                    bias: 0.0,
                },
                crate::NeuronJson {
                    uuid: "hidden-b".to_string(),
                    neuron_type: "hidden".to_string(),
                    squash: "IDENTITY".to_string(),
                    bias: 0.0,
                },
                crate::NeuronJson {
                    uuid: "min-0".to_string(),
                    neuron_type: "hidden".to_string(),
                    squash: "MINIMUM".to_string(),
                    bias: 0.0,
                },
                crate::NeuronJson {
                    uuid: "output-0".to_string(),
                    neuron_type: "output".to_string(),
                    squash: "IDENTITY".to_string(),
                    bias: 0.0,
                },
            ],
            synapses: vec![
                crate::SynapseJson {
                    from_uuid: "hidden-a".to_string(),
                    to_uuid: "min-0".to_string(),
                    weight: 1.0,
                    synapse_type: None,
                },
                crate::SynapseJson {
                    from_uuid: "hidden-b".to_string(),
                    to_uuid: "min-0".to_string(),
                    weight: 1.0,
                    synapse_type: None,
                },
                crate::SynapseJson {
                    from_uuid: "min-0".to_string(),
                    to_uuid: "output-0".to_string(),
                    weight: 1.0,
                    synapse_type: None,
                },
            ],
            input: 0,
            output: 1,
        };

        // Build a cache that returns activations where hidden-a ALWAYS wins MINIMUM.
        let cache = Arc::new(RecordCache::with_loader(
            "unused.parquet",
            Arc::new(|_file, uuid| {
                let mut records = Vec::new();
                for obs_index in 0..10u32 {
                    let activation = match uuid {
                        "hidden-a" => 0.0,
                        "hidden-b" => 1.0,
                        _ => 0.0,
                    };
                    records.push(DiscoverRecord {
                        obs_index,
                        neuron_uuid: uuid.to_string(),
                        value: None,
                        activation,
                        errors: vec![0.0],
                    });
                }
                Ok(records)
            }),
        ));

        let impacts = compute_impact_scores_for_discounting(&creature, cache.as_ref());
        let a = impacts.get("hidden-a").copied().unwrap_or(0.0);
        let b = impacts.get("hidden-b").copied().unwrap_or(0.0);

        assert!(
            a > 0.9,
            "hidden-a should have near-full impact via MINIMUM win probability, got {a}"
        );
        assert!(
            b < 0.1,
            "hidden-b should have near-zero impact via MINIMUM win probability, got {b}"
        );
        Ok(())
    }

    /// Test that M4 is detected as high-performance tier.
    #[test]
    fn gpu_tier_detects_m4_as_high_performance() {
        let info = wgpu::AdapterInfo {
            name: "Apple M4".to_string(),
            vendor: 0,
            device: 0,
            device_type: wgpu::DeviceType::IntegratedGpu,
            driver: String::new(),
            driver_info: String::new(),
            backend: wgpu::Backend::Metal,
        };
        assert_eq!(
            detect_gpu_tier(&info),
            GpuPerformanceTier::High,
            "M4 should be detected as high-performance"
        );
    }

    /// Test that M4 Pro/Max are detected as high-performance tier.
    #[test]
    fn gpu_tier_detects_m4_pro_max_as_high_performance() {
        for name in ["Apple M4 Pro", "Apple M4 Max", "Apple M4 Ultra"] {
            let info = wgpu::AdapterInfo {
                name: name.to_string(),
                vendor: 0,
                device: 0,
                device_type: wgpu::DeviceType::IntegratedGpu,
                driver: String::new(),
                driver_info: String::new(),
                backend: wgpu::Backend::Metal,
            };
            assert_eq!(
                detect_gpu_tier(&info),
                GpuPerformanceTier::High,
                "{name} should be detected as high-performance"
            );
        }
    }

    /// Test that M3 Pro/Max are detected as high-performance tier.
    #[test]
    fn gpu_tier_detects_m3_pro_max_as_high_performance() {
        for name in ["Apple M3 Pro", "Apple M3 Max"] {
            let info = wgpu::AdapterInfo {
                name: name.to_string(),
                vendor: 0,
                device: 0,
                device_type: wgpu::DeviceType::IntegratedGpu,
                driver: String::new(),
                driver_info: String::new(),
                backend: wgpu::Backend::Metal,
            };
            assert_eq!(
                detect_gpu_tier(&info),
                GpuPerformanceTier::High,
                "{name} should be detected as high-performance"
            );
        }
    }

    /// Test that base M1/M2/M3 are detected as standard tier.
    #[test]
    fn gpu_tier_detects_base_m_series_as_standard() {
        for name in ["Apple M1", "Apple M2", "Apple M3"] {
            let info = wgpu::AdapterInfo {
                name: name.to_string(),
                vendor: 0,
                device: 0,
                device_type: wgpu::DeviceType::IntegratedGpu,
                driver: String::new(),
                driver_info: String::new(),
                backend: wgpu::Backend::Metal,
            };
            assert_eq!(
                detect_gpu_tier(&info),
                GpuPerformanceTier::Standard,
                "{name} should be detected as standard"
            );
        }
    }

    /// Test that discrete GPUs are detected as high-performance.
    #[test]
    fn gpu_tier_detects_discrete_gpu_as_high_performance() {
        let info = wgpu::AdapterInfo {
            name: "NVIDIA GeForce RTX 4090".to_string(),
            vendor: 0,
            device: 0,
            device_type: wgpu::DeviceType::DiscreteGpu,
            driver: String::new(),
            driver_info: String::new(),
            backend: wgpu::Backend::Vulkan,
        };
        assert_eq!(
            detect_gpu_tier(&info),
            GpuPerformanceTier::High,
            "Discrete GPUs should be detected as high-performance"
        );
    }

    /// Test that unknown integrated GPUs are detected as standard.
    #[test]
    fn gpu_tier_detects_unknown_integrated_as_standard() {
        let info = wgpu::AdapterInfo {
            name: "Intel UHD Graphics 630".to_string(),
            vendor: 0,
            device: 0,
            device_type: wgpu::DeviceType::IntegratedGpu,
            driver: String::new(),
            driver_info: String::new(),
            backend: wgpu::Backend::Vulkan,
        };
        assert_eq!(
            detect_gpu_tier(&info),
            GpuPerformanceTier::Standard,
            "Unknown integrated GPUs should be detected as standard"
        );
    }

    /// Test that batch size is correct for each tier.
    #[test]
    fn batch_size_correct_for_each_tier() {
        assert_eq!(
            get_batch_size_for_tier(GpuPerformanceTier::High),
            HIGH_PERF_GPU_BATCH_SIZE,
            "High-performance tier should use larger batch size"
        );
        assert_eq!(
            get_batch_size_for_tier(GpuPerformanceTier::Standard),
            DEFAULT_GPU_BATCH_SIZE,
            "Standard tier should use default batch size"
        );
        assert_eq!(
            get_batch_size_for_tier(GpuPerformanceTier::Unknown),
            DEFAULT_GPU_BATCH_SIZE,
            "Unknown tier should use default batch size"
        );
    }

    #[test]
    fn gpu_batch_size_caps_for_large_sample_counts() {
        // Dec 2025: Large recordings (50k+ samples) can wedge Metal when combined with
        // a high batch size. We cap based on estimated GPU buffer sizes.
        let configured = 1024;
        let max_sample_len = 58_149; // representative from production logs

        // Helpful path uses HelpfulContribution (48 bytes) + staging, plus sample buffer.
        let bytes_per_sample_helpful = std::mem::size_of::<GpuHelpfulSample>()
            + (2 * std::mem::size_of::<HelpfulContribution>());
        let capped_helpful = cap_gpu_batch_size_by_bytes(
            configured,
            max_sample_len,
            bytes_per_sample_helpful,
            GPU_MAX_BATCH_ALLOC_BYTES,
        );
        assert!(
            capped_helpful < configured,
            "expected helpful batch size to be capped for large sample sets"
        );
        assert!(
            capped_helpful >= 1,
            "batch size must never be reduced below 1"
        );

        // Harmful path uses HarmfulContribution (16 bytes) + staging, plus sample buffer.
        let bytes_per_sample_harmful = std::mem::size_of::<GpuHelpfulSample>()
            + (2 * std::mem::size_of::<HarmfulContribution>());
        let capped_harmful = cap_gpu_batch_size_by_bytes(
            configured,
            max_sample_len,
            bytes_per_sample_harmful,
            GPU_MAX_BATCH_ALLOC_BYTES,
        );
        assert!(
            capped_harmful < configured,
            "expected harmful batch size to be capped for large sample sets"
        );
        assert!(capped_harmful >= 1);
    }

    /// Test that verbose_enabled() is cached (doesn't re-read env var each time).
    #[test]
    fn verbose_enabled_is_cached() {
        // Call twice - should return same value without re-reading env
        let first = verbose_enabled();
        let second = verbose_enabled();
        assert_eq!(first, second, "verbose_enabled() should be deterministic");
    }

    // ==================== Memory Info / Page Size Tests ====================

    /// Test parsing page size from vm_stat output - Apple Silicon (16KB pages).
    #[test]
    #[cfg(target_os = "macos")]
    fn parse_vm_stat_page_size_apple_silicon() {
        let vm_stat_output = r#"Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               15417.
Pages active:                            624277.
Pages inactive:                          601916.
Pages speculative:                        23646.
"#;
        assert_eq!(
            parse_vm_stat_page_size(vm_stat_output),
            16384,
            "Should parse 16384 byte page size for Apple Silicon"
        );
    }

    /// Test parsing page size from vm_stat output - Intel Mac (4KB pages).
    #[test]
    #[cfg(target_os = "macos")]
    fn parse_vm_stat_page_size_intel_mac() {
        let vm_stat_output = r#"Mach Virtual Memory Statistics: (page size of 4096 bytes)
Pages free:                               45123.
Pages active:                           1234567.
Pages inactive:                          876543.
Pages speculative:                        12345.
"#;
        assert_eq!(
            parse_vm_stat_page_size(vm_stat_output),
            4096,
            "Should parse 4096 byte page size for Intel Mac"
        );
    }

    /// Test parsing page size handles malformed output gracefully.
    #[test]
    #[cfg(target_os = "macos")]
    fn parse_vm_stat_page_size_malformed_output() {
        // Empty string should return architecture-appropriate default
        let result = parse_vm_stat_page_size("");
        #[cfg(target_arch = "aarch64")]
        assert_eq!(
            result, 16384,
            "Empty output should default to 16KB on ARM64"
        );
        #[cfg(not(target_arch = "aarch64"))]
        assert_eq!(result, 4096, "Empty output should default to 4KB on Intel");

        // Missing "page size of" should return default
        let malformed = "Some random output without page size info";
        let result = parse_vm_stat_page_size(malformed);
        #[cfg(target_arch = "aarch64")]
        assert_eq!(
            result, 16384,
            "Malformed output should default to 16KB on ARM64"
        );
        #[cfg(not(target_arch = "aarch64"))]
        assert_eq!(
            result, 4096,
            "Malformed output should default to 4KB on Intel"
        );
    }

    /// Test that the parser handles various page size values.
    #[test]
    #[cfg(target_os = "macos")]
    fn parse_vm_stat_page_size_various_sizes() {
        // Test 4KB pages (Intel)
        let output_4k =
            "Mach Virtual Memory Statistics: (page size of 4096 bytes)\nPages free: 100.";
        assert_eq!(parse_vm_stat_page_size(output_4k), 4096);

        // Test 16KB pages (Apple Silicon)
        let output_16k =
            "Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 100.";
        assert_eq!(parse_vm_stat_page_size(output_16k), 16384);

        // Test hypothetical larger page size (future-proofing)
        let output_64k =
            "Mach Virtual Memory Statistics: (page size of 65536 bytes)\nPages free: 100.";
        assert_eq!(parse_vm_stat_page_size(output_64k), 65536);
    }

    /// Test Linux meminfo parsing with valid input.
    #[test]
    #[cfg(target_os = "linux")]
    fn parse_meminfo_line_valid_input() {
        // Standard format from /proc/meminfo
        assert_eq!(
            parse_meminfo_line("MemTotal:       16384000 kB"),
            Some(16384000)
        );
        assert_eq!(
            parse_meminfo_line("MemAvailable:    8192000 kB"),
            Some(8192000)
        );
        // Single digit
        assert_eq!(parse_meminfo_line("MemFree:        1 kB"), Some(1));
    }

    /// Test Linux meminfo parsing with malformed input returns None (not 0).
    /// This is critical: returning None allows defaults to be preserved.
    #[test]
    #[cfg(target_os = "linux")]
    fn parse_meminfo_line_malformed_returns_none() {
        // Missing value
        assert_eq!(parse_meminfo_line("MemTotal:"), None);
        // Non-numeric value
        assert_eq!(parse_meminfo_line("MemTotal:       abc kB"), None);
        // Empty string
        assert_eq!(parse_meminfo_line(""), None);
        // Just whitespace after colon
        assert_eq!(parse_meminfo_line("MemTotal:       "), None);
    }

    /// Test that Linux get_memory_info preserves defaults when parsing fails.
    /// This prevents misleading "0.0GB" error messages.
    #[test]
    #[cfg(target_os = "linux")]
    fn linux_memory_info_preserves_defaults_on_malformed_input() {
        // This test verifies the fix by checking that parse_meminfo_line
        // returns None for malformed input, which allows get_memory_info
        // to preserve its default values instead of overwriting with 0.
        //
        // The actual get_memory_info function reads /proc/meminfo, so we
        // can't easily test it with mock data. Instead, we verify the
        // building blocks work correctly:

        // 1. Valid input should return Some(value)
        assert!(parse_meminfo_line("MemTotal:       16384000 kB").is_some());

        // 2. Malformed input should return None (not Some(0))
        assert!(parse_meminfo_line("MemTotal:").is_none());
        assert!(parse_meminfo_line("MemTotal:       abc").is_none());

        // 3. This ensures the if-let pattern in get_memory_info preserves defaults
    }

    // ==================== Activation Function Tests ====================

    #[test]
    fn test_identity_activation() {
        assert_eq!(identity_activation(1.0), 1.0);
        assert_eq!(identity_activation(-1.0), -1.0);
        assert_eq!(identity_activation(0.0), 0.0);
    }

    #[test]
    fn test_bipolar_activation() {
        assert_eq!(bipolar_activation(1.0), 1.0);
        assert_eq!(bipolar_activation(0.0001), 1.0);
        assert_eq!(bipolar_activation(0.0), -1.0);
        assert_eq!(bipolar_activation(-1.0), -1.0);
        assert_eq!(bipolar_activation(-0.0001), -1.0);
    }

    #[test]
    fn test_clipped_activation() {
        assert_eq!(clipped_activation(1.5), 1.0);
        assert_eq!(clipped_activation(0.5), 0.5);
        assert_eq!(clipped_activation(-0.5), -0.5);
        assert_eq!(clipped_activation(-1.5), -1.0);
    }

    #[test]
    fn test_absolute_activation() {
        assert_eq!(absolute_activation(1.0), 1.0);
        assert_eq!(absolute_activation(-1.0), 1.0);
        assert_eq!(absolute_activation(0.0), 0.0);
    }

    #[test]
    fn test_specs_include_new_activations() {
        let names: Vec<&str> = ACTIVATION_SPECS.iter().map(|s| s.name).collect();
        // Original activations
        assert!(names.contains(&"IDENTITY"));
        assert!(names.contains(&"BIPOLAR"));
        assert!(names.contains(&"CLIPPED"));
        assert!(names.contains(&"ABSOLUTE"));
        assert!(
            !names.contains(&"SELU"),
            "SELU should not be suggested as a new neuron activation (Issue #148: time-bounded runs)"
        );
        assert!(
            !names.contains(&"INVERSE"),
            "INVERSE (complement) should not be suggested as a new neuron activation. \
             It can be represented via IDENTITY with bias and negative incoming weights."
        );
        // New activations (v0.1.139)
        assert!(
            !names.contains(&"LeakyReLU"),
            "LeakyReLU should not be suggested as a new neuron activation"
        );
        assert!(
            names.contains(&"Mish"),
            "Mish should be included - 2 successful discoveries!"
        );
        assert!(
            !names.contains(&"Swish"),
            "Swish should not be suggested as a new neuron activation (Issue #148: time-bounded runs)"
        );
        assert!(names.contains(&"HARD_TANH"), "HARD_TANH should be included");
        assert!(
            names.contains(&"SOFTSIGN"),
            "SOFTSIGN should be included - successful discovery!"
        );
        assert!(
            names.contains(&"BENT_IDENTITY"),
            "BENT_IDENTITY should be included - successful discovery!"
        );
        assert!(names.contains(&"ArcTan"), "ArcTan should be included");
        assert!(names.contains(&"ReLU6"), "ReLU6 should be included");
        // Total count
        assert_eq!(names.len(), 15, "Should have 15 activation specs");
    }

    // ==================== Saturation Detection Tests (Issue #123) ====================

    /// Test that has_sufficient_output_variance detects saturated neurons.
    /// Issue #123: Large bias values cause neurons to output nearly-constant values,
    /// leading to massive prediction failures (e.g., predicting 4.67% when actual is ~0%).
    #[test]
    fn test_saturation_detection_rejects_constant_output() {
        // Create samples with typical activation range [-1, 1]
        let samples: Vec<HelpfulSample> = (-10..=10)
            .map(|i| HelpfulSample {
                activation: i as f32 / 10.0,
                avg_error: 0.1,
                target_value: None,
                target_activation: None,
            })
            .collect();

        // SOFTSIGN with bias=5 and incoming=0.35 (from issue #123)
        // Output ≈ (5 + 0.35×x) / (1 + |5 + 0.35×x|) ≈ 0.83 for all x
        let saturated = !has_sufficient_output_variance(
            &samples,
            0.35, // incoming_weight
            5.0,  // bias (too large!)
            softsign_activation,
        );
        assert!(
            saturated,
            "SOFTSIGN with bias=5 should be detected as saturated (constant output)"
        );

        // SOFTSIGN with bias=0 should NOT be saturated
        let not_saturated = has_sufficient_output_variance(
            &samples,
            1.0, // incoming_weight
            0.0, // bias
            softsign_activation,
        );
        assert!(
            not_saturated,
            "SOFTSIGN with bias=0 should NOT be saturated"
        );
    }

    /// Test that TANH with large bias is detected as saturated.
    #[test]
    fn test_saturation_detection_tanh_large_bias() {
        let samples: Vec<HelpfulSample> = (-10..=10)
            .map(|i| HelpfulSample {
                activation: i as f32 / 10.0,
                avg_error: 0.1,
                target_value: None,
                target_activation: None,
            })
            .collect();

        // TANH with bias=10 is heavily saturated (output ≈ 1.0 for all inputs)
        let saturated = !has_sufficient_output_variance(
            &samples,
            1.0,  // incoming_weight
            10.0, // bias (way too large!)
            |x| x.tanh(),
        );
        assert!(
            saturated,
            "TANH with bias=10 should be detected as saturated"
        );

        // TANH with moderate bias=1 should still have variance
        let not_saturated = has_sufficient_output_variance(
            &samples,
            1.0, // incoming_weight
            1.0, // bias (reasonable)
            |x| x.tanh(),
        );
        assert!(
            not_saturated,
            "TANH with bias=1 should NOT be saturated - still has output variance"
        );
    }

    /// Test that ReLU is not incorrectly flagged as saturated.
    /// ReLU naturally has "half" of samples at 0, but the other half varies.
    #[test]
    fn test_saturation_detection_relu_not_false_positive() {
        let samples: Vec<HelpfulSample> = (-10..=10)
            .map(|i| HelpfulSample {
                activation: i as f32 / 10.0,
                avg_error: 0.1,
                target_value: None,
                target_activation: None,
            })
            .collect();

        // ReLU with bias=0 should NOT be flagged as saturated
        // Half the outputs are 0, but the other half varies from 0 to 1
        let not_saturated = has_sufficient_output_variance(&samples, 1.0, 0.0, |x| x.max(0.0));
        assert!(
            not_saturated,
            "ReLU with bias=0 should NOT be flagged as saturated"
        );
    }

    /// Test that constant input samples are NOT incorrectly rejected.
    /// If input is constant, output will be constant too, but predictions are still valid.
    #[test]
    fn test_saturation_detection_allows_constant_input() {
        // All samples have the same input activation (constant input)
        let samples: Vec<HelpfulSample> = (0..20)
            .map(|_| HelpfulSample {
                activation: -1.0, // Constant input
                avg_error: 0.3,
                target_value: None,
                target_activation: None,
            })
            .collect();

        // Even with large bias, constant input should be allowed
        // because predictions are still valid when input doesn't vary
        let should_allow = has_sufficient_output_variance(
            &samples,
            1.0,
            5.0, // Large bias, but input is constant so it's OK
            |x| x.tanh(),
        );
        assert!(
            should_allow,
            "Constant input samples should NOT be rejected - predictions are valid"
        );
    }

    #[test]
    fn parse_input_index_parses_valid_ids() {
        assert_eq!(parse_input_index("input-0"), Some(0));
        assert_eq!(parse_input_index("input-1486"), Some(1486));
        assert_eq!(parse_input_index("input-001"), Some(1));
        assert_eq!(parse_input_index("hidden-1"), None);
        assert_eq!(parse_input_index("input-"), None);
    }

    #[test]
    fn shuffle_slice_is_deterministic_with_seed_and_context() {
        let mut a = (0..20).collect::<Vec<i32>>();
        let mut b = (0..20).collect::<Vec<i32>>();

        shuffle_slice(&mut a, Some(123), "ctx");
        shuffle_slice(&mut b, Some(123), "ctx");
        assert_eq!(a, b);

        // Different context should (very likely) produce a different order.
        let mut c = (0..20).collect::<Vec<i32>>();
        shuffle_slice(&mut c, Some(123), "different-ctx");
        assert_ne!(a, c);

        // And the shuffle should actually move at least something.
        assert_ne!(a, (0..20).collect::<Vec<i32>>());
    }

    #[test]
    fn neuron_analysis_counts_threshold_targets_as_completed_for_progress_reporting() -> Result<()> {
        // 29-Dec-2025: Regression test for progress reporting.
        //
        // STEP/BIPOLAR targets are intentionally skipped by add-neuron analysis (synapse analysis
        // is the supported mechanism for discrete targets), but they still appear in the focus
        // list and therefore must be counted as "completed" for progress reporting.
        //
        // Historically an early `return Ok(())` skipped the completion counter update, leaving the
        // watchdog stage stuck at "processing target ..." rather than "completed 1/1".

        let _lock = crate::watchdog::lock_for_test_serialisation();

        // Keep this aligned with integration tests: skip rather than fail when no GPU is present.
        if !crate::analysis::GpuAnalyzer::gpu_is_available() {
            eprintln!("Skipping test: no GPU available");
            return Ok(());
        }

        let _wd = crate::watchdog::Watchdog::start(crate::watchdog::WatchdogConfig {
            stall_timeout: std::time::Duration::from_secs(60),
            abort_delay: std::time::Duration::from_secs(1),
        });

        // Use a unique UUID so other parallel tests (which often use "output-0") won't
        // accidentally overwrite the watchdog stage while our watchdog is active.
        let target_uuid = "output-progress-0";

        let creature = crate::CreatureJson {
            input: 1,
            output: 1,
            neurons: vec![crate::NeuronJson {
                uuid: target_uuid.to_string(),
                neuron_type: "output".to_string(),
                squash: "STEP".to_string(),
                bias: 0.0,
            }],
            synapses: Vec::new(),
        };

        // Use an in-memory loader so the test doesn't need to write a parquet file.
        let cache = std::sync::Arc::new(RecordCache::with_loader(
            "unused.parquet",
            std::sync::Arc::new(move |_file, uuid| {
                let mut records = Vec::new();
                for obs_index in 0..20u32 {
                    match uuid {
                        "input-0" => {
                            records.push(DiscoverRecord {
                                obs_index,
                                neuron_uuid: uuid.to_string(),
                                value: None,
                                activation: (obs_index as f32 - 10.0) / 10.0, // -1.0 .. 0.9
                                errors: Vec::new(),
                            });
                        }
                        _ if uuid == target_uuid => {
                            // Provide non-empty errors so the target is considered analysable.
                            let activation = 0.0;
                            let error = if obs_index < 10 { 0.25 } else { -0.25 };
                            records.push(DiscoverRecord {
                                obs_index,
                                neuron_uuid: uuid.to_string(),
                                value: Some(activation),
                                activation,
                                errors: vec![error],
                            });
                        }
                        _ => {}
                    }
                }
                Ok(records)
            }),
        ));

        let input = crate::AnalyzeNeuronsInput {
            parquet_file: "unused.parquet".to_string(),
            creature,
            focus_neurons: vec![target_uuid.to_string()],
            max_candidates: Some(1),
            analysis_deadline_ms: None,
            random_seed: Some(123),
        };

        let _result = analyze_neurons_with_cache(&input, cache)?;

        let stage = crate::watchdog::active_stage_for_test().unwrap_or_default();
        assert!(
            stage.contains("neuron analysis → completed 1/1"),
            "expected progress to reach completed 1/1 for a threshold target, got: {stage}"
        );
        Ok(())
    }
