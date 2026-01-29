## Summary

Added a prominent **Project Mission** section near the top of `README.md` that
clearly states the project's three guiding principles (issue #346):

1. **Improve the creature's score** — only return candidates expected to improve it.
2. **Discover improvements as fast as possible** — leverage GPU compute shaders and SIMD.
3. **Minimise changes to NEAT-AI** — reuse existing candidate types wherever possible.

Also updated the existing **Goal** section (in the reference material) to
cross-reference the new mission statement and reinforce the speed/score objectives.

## Evidence

Unable to generate screenshot: this is a Rust library with no visual interface.
The change is purely documentation (README.md) verified by compile-time tests.

## Test Plan

- Added `tests/issue_346_readme_project_goal.rs` with 6 tests that verify the
  README contains the required goal messaging:
  - `readme_contains_project_mission_section` — section header exists
  - `readme_mission_states_improve_creature_score` — score improvement goal
  - `readme_mission_states_speed_goal` — "as fast as possible" phrasing
  - `readme_mission_states_candidates_must_improve_score` — candidate quality rule
  - `readme_mission_states_reuse_candidate_types` — NEAT-AI compatibility preference
  - `readme_mission_mentions_gpu_acceleration` — GPU and SIMD mentioned
