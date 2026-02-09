//! Issue #346: Verify that the README clearly communicates the project's
//! primary goal — improving the creature's score as fast as possible — so
//! that contributors and AI agents do not need per-issue reminders.
//!
//! These tests read `README.md` at compile time and assert that the key
//! messaging required by issue #346 is present.

const README: &str = include_str!("../README.md");

// ---------------------------------------------------------------------------
// 1. The project mission must be stated prominently
// ---------------------------------------------------------------------------

#[test]
fn readme_contains_project_mission_section() {
    assert!(
        README.contains("## Project Mission"),
        "README must contain a '## Project Mission' section near the top"
    );
}

#[test]
fn readme_mission_states_improve_creature_score() {
    assert!(
        README.contains("improve the creature's score"),
        "README Project Mission must state the goal of improving the creature's score"
    );
}

#[test]
fn readme_mission_states_speed_goal() {
    // The issue asks us to make it clear we want discoveries "as fast as possible"
    assert!(
        README.contains("as fast as possible"),
        "README Project Mission must mention discovering improvements 'as fast as possible'"
    );
}

// ---------------------------------------------------------------------------
// 2. Candidate quality: only return candidates expected to improve score
// ---------------------------------------------------------------------------

#[test]
fn readme_mission_states_candidates_must_improve_score() {
    assert!(
        README
            .contains("Only return candidates which are expected to improve the creature's score"),
        "README must state that only candidates expected to improve the creature's score should be returned"
    );
}

// ---------------------------------------------------------------------------
// 3. Prefer not to change the calling programme (NEAT-AI)
// ---------------------------------------------------------------------------

#[test]
fn readme_mission_states_reuse_candidate_types() {
    assert!(
        README.contains("reuse existing candidate types")
            || README.contains("reuse the candidate types"),
        "README must mention preference to reuse existing NEAT-AI candidate types"
    );
}

// ---------------------------------------------------------------------------
// 4. GPU/SIMD acceleration is explicitly mentioned
// ---------------------------------------------------------------------------

#[test]
fn readme_mission_mentions_gpu_acceleration() {
    assert!(
        README.contains("GPU") && README.contains("SIMD"),
        "README Project Mission must mention GPU and SIMD acceleration"
    );
}
