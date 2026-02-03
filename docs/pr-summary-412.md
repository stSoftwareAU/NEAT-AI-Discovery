# PR Summary: Plan Discovery Improvements (#412)

## Summary

Created 19 GitHub sub-issues to systematically address discovery process improvements. The issues are organised by category with clear requirements for TDD, benchmarks, and documentation updates.

## Evidence

This is a planning/meta issue that creates sub-issues rather than implementing code changes. No screenshots or benchmarks are applicable.

The following GitHub issues were created:

### Fix Broken Discovery Types (Critical)
| Issue | Title | Expected Improvement |
|-------|-------|---------------------|
| #414 | Fix remove-neuron (high error) discovery - 0% success rate | 10-15% success rate or disable |
| #415 | Fix combo-successful discovery - 0% success rate | 10-30% success rate or disable |

### Improve Existing Discovery Accuracy
| Issue | Title | Expected Improvement |
|-------|-------|---------------------|
| #413 | Improve add-synapse prediction accuracy | 15-20% success rate (from inverting predictions) |
| #416 | Investigate remove-harmful-synapse discovery | Enable testing, 15-20% success rate |
| #417 | Increase change-squash suggestion rate | 10x volume increase while maintaining 18.2% success |

### GPU/CPU Performance Improvements
| Issue | Title | Expected Improvement |
|-------|-------|---------------------|
| #418 | GPU pipeline optimisation - reduce context switches | 10-20% GPU time reduction |
| #419 | Parallel discovery module execution | 2-4x speedup on 8+ core systems |
| #420 | Streaming analysis for memory-constrained systems | 2x larger creatures in same memory |
| #427 | Add CPU SIMD fallback for GPU operations | Works on CPU-only systems |
| #429 | Early termination improvements for low-value candidates | 30-50% analysis time reduction |

### New Discovery Types
| Issue | Title | Expected Improvement |
|-------|-------|---------------------|
| #421 | Gradient-based discovery - directional improvement hints | 25-30% success rate for weight adjustments |
| #422 | Topology-aware discovery - network structure analysis | 10-15% success rate |
| #423 | Sample-weighted discovery - prioritise high-error samples | 20% improvement in candidate quality |
| #428 | Adaptive candidate confidence thresholds | 10-15% improvement in success rate |
| #431 | Activation function recommendation engine | 20% reduction in saturation/oscillation |

### Code Quality / DRY Improvements
| Issue | Title | Expected Improvement |
|-------|-------|---------------------|
| #424 | Consolidate discovery constants into central module | DRY compliance, easier tuning |
| #425 | Complete implementation.rs refactoring (Issue #185) | Reduce to <1,000 lines |
| #426 | Split implementation_tests.rs into focused test modules | No file >1,500 lines |

### Observability
| Issue | Title | Expected Improvement |
|-------|-------|---------------------|
| #430 | Add discovery module performance metrics | Data-driven module prioritisation |

## Requirements Summary

All issues include:
- **TDD requirements**: Write failing tests first
- **Test type specification**: Unit tests for functionality, benchmarks for performance
- **Documentation requirements**: Update `docs/DISCOVERY_TYPES.md` where applicable
- **Acceptance criteria**: Clear, measurable outcomes

### By Category

| Category | Issues | Requires Benchmarks | Requires Unit Tests | Requires Docs |
|----------|--------|---------------------|---------------------|---------------|
| Fix Broken Types | #414, #415 | No | Yes | Yes |
| Improve Accuracy | #413, #416, #417 | No | Yes | Yes |
| GPU/CPU Performance | #418, #419, #420, #427, #429 | Yes | Yes | Yes |
| New Discovery Types | #421, #422, #423, #428, #431 | No | Yes | Yes |
| Code Quality | #424, #425, #426 | No | Existing tests must pass | Yes |
| Observability | #430 | No | Yes | Yes |

## Test Plan

This PR creates planning issues only. No code changes were made.

- Verified all 19 GitHub issues created successfully
- Verified parent issue #412 updated with sub-issue links
- No tests modified or added (meta-issue for planning)
