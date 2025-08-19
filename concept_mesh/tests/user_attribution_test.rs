//! Tests for user attribution in ConceptDiffs
//!
//! This integration test validates that user attribution data is properly
//! preserved when ConceptDiffs are created, stored in a PsiArc, and replayed.

use concept_mesh::auth::{set_current_session, PersonaMode, Session, User};
use concept_mesh::diff::{
    user_context::add_user_context, ConceptDiff, ConceptDiffBuilder, UserContextExt,
};
use concept_mesh::lcn::LargeConceptNetwork;
use concept_mesh::psiarc::{PsiArc, PsiArcBuilder, PsiArcConfig, PsiArcReader};
use std::path::PathBuf;
use tempfile::tempdir;

#[test]
fn test_user_attribution_in_conceptdiff() {
    // Create a test user and session
    let user1 = User::new(
        "github",
        "user1",
        Some("user1@example.com"),
        Some("User One"),
        Some("https://example.com/avatar1.png"),
    );

    let session1 = Session::new(user1.clone(), PersonaMode::Glyphsmith);
    set_current_session(session1.clone());

    // Create a ConceptDiff with user context
    let diff1 = ConceptDiffBuilder::new(1)
        .create("concept1", "TestConcept", Default::default())
        .with_user_context()
        .build();

    // Verify user context was added
    assert!(diff1.metadata.contains_key("user_id"));
    assert_eq!(
        diff1.metadata.get("user_id").unwrap().as_str().unwrap(),
        user1.concept_id
    );

    assert!(diff1.metadata.contains_key("persona_id"));
    assert_eq!(
        diff1.metadata.get("persona_id").unwrap().as_str().unwrap(),
        PersonaMode::Glyphsmith.as_str()
    );

    // Create a second user and session
    let user2 = User::new(
        "google",
        "user2",
        Some("user2@example.com"),
        Some("User Two"),
        Some("https://example.com/avatar2.png"),
    );

    let session2 = Session::new(user2.clone(), PersonaMode::Researcher);
    set_current_session(session2.clone());

    // Create another ConceptDiff with the new user
    let diff2 = ConceptDiffBuilder::new(2)
        .create("concept2", "TestConcept", Default::default())
        .with_user_context()
        .build();

    // Verify new user context was added
    assert!(diff2.metadata.contains_key("user_id"));
    assert_eq!(
        diff2.metadata.get("user_id").unwrap().as_str().unwrap(),
        user2.concept_id
    );

    assert!(diff2.metadata.contains_key("persona_id"));
    assert_eq!(
        diff2.metadata.get("persona_id").unwrap().as_str().unwrap(),
        PersonaMode::Researcher.as_str()
    );
}

#[test]
fn test_user_attribution_with_psiarc() -> Result<(), Box<dyn std::error::Error>> {
    // Create temporary directory
    let temp_dir = tempdir()?;
    let psiarc_path = temp_dir.path().join("test.psiarc");

    // Create users
    let user1 = User::new(
        "github",
        "user1",
        Some("user1@example.com"),
        Some("User One"),
        Some("https://example.com/avatar1.png"),
    );

    let user2 = User::new(
        "google",
        "user2",
        Some("user2@example.com"),
        Some("User Two"),
        Some("https://example.com/avatar2.png"),
    );

    // Create LCN
    let lcn = LargeConceptNetwork::new();

    // Create PsiArc
    let mut config = PsiArcConfig::default();
    config.path = psiarc_path.clone();

    let mut psiarc = PsiArcBuilder::new(config).build()?;

    // Write diffs from different users
    {
        // User 1 with Glyphsmith persona
        let session1 = Session::new(user1.clone(), PersonaMode::Glyphsmith);
        set_current_session(session1);

        let diff1 = ConceptDiffBuilder::new(1)
            .create("concept1", "TestConcept", Default::default())
            .with_user_context()
            .build();

        psiarc.write_diff(&diff1)?;
        lcn.apply_diff(&diff1)?;

        // User 2 with Researcher persona
        let session2 = Session::new(user2.clone(), PersonaMode::Researcher);
        set_current_session(session2);

        let diff2 = ConceptDiffBuilder::new(2)
            .create("concept2", "TestConcept", Default::default())
            .with_user_context()
            .build();

        psiarc.write_diff(&diff2)?;
        lcn.apply_diff(&diff2)?;

        // User 1 with Memory Pruner persona
        let session3 = Session::new(user1.clone(), PersonaMode::MemoryPruner);
        set_current_session(session3);

        let diff3 = ConceptDiffBuilder::new(3)
            .create("concept3", "TestConcept", Default::default())
            .with_user_context()
            .build();

        psiarc.write_diff(&diff3)?;
        lcn.apply_diff(&diff3)?;
    }

    // Close and reopen PsiArc
    drop(psiarc);

    // Read back the diffs and verify attribution
    let reader = PsiArcReader::open(&psiarc_path)?;
    let frames = reader.read_all_frames()?;

    assert_eq!(frames.len(), 3);

    // Check first diff (User 1, Glyphsmith)
    {
        let diff = &frames[0].diff;
        assert!(diff.metadata.contains_key("user_id"));
        assert_eq!(
            diff.metadata.get("user_id").unwrap().as_str().unwrap(),
            user1.concept_id
        );

        assert!(diff.metadata.contains_key("persona_id"));
        assert_eq!(
            diff.metadata.get("persona_id").unwrap().as_str().unwrap(),
            PersonaMode::Glyphsmith.as_str()
        );
    }

    // Check second diff (User 2, Researcher)
    {
        let diff = &frames[1].diff;
        assert!(diff.metadata.contains_key("user_id"));
        assert_eq!(
            diff.metadata.get("user_id").unwrap().as_str().unwrap(),
            user2.concept_id
        );

        assert!(diff.metadata.contains_key("persona_id"));
        assert_eq!(
            diff.metadata.get("persona_id").unwrap().as_str().unwrap(),
            PersonaMode::Researcher.as_str()
        );
    }

    // Check third diff (User 1, Memory Pruner)
    {
        let diff = &frames[2].diff;
        assert!(diff.metadata.contains_key("user_id"));
        assert_eq!(
            diff.metadata.get("user_id").unwrap().as_str().unwrap(),
            user1.concept_id
        );

        assert!(diff.metadata.contains_key("persona_id"));
        assert_eq!(
            diff.metadata.get("persona_id").unwrap().as_str().unwrap(),
            PersonaMode::MemoryPruner.as_str()
        );
    }

    Ok(())
}

#[test]
fn test_multi_user_replay() -> Result<(), Box<dyn std::error::Error>> {
    // Create temporary directory
    let temp_dir = tempdir()?;
    let psiarc_path = temp_dir.path().join("multi_user.psiarc");

    // Create users
    let user1 = User::new("github", "user1", None, Some("User One"), None);
    let user2 = User::new("google", "user2", None, Some("User Two"), None);
    let user3 = User::new("discord", "user3", None, Some("User Three"), None);

    // Create original LCN
    let original_lcn = LargeConceptNetwork::new();

    // Create PsiArc
    let mut config = PsiArcConfig::default();
    config.path = psiarc_path.clone();

    let mut psiarc = PsiArcBuilder::new(config).build()?;

    // Create a complex sequence of operations from different users
    {
        // Initialize with Genesis event
        concept_mesh::initialize(&original_lcn, "TestCorpus")?;
        let genesis_diff = concept_mesh::diff::create_genesis_diff("TestCorpus");
        psiarc.write_diff(&genesis_diff)?;

        // User 1: Create a category
        let session1 = Session::new(user1.clone(), PersonaMode::CreativeAgent);
        set_current_session(session1);

        let diff1 = ConceptDiffBuilder::new(1)
            .create("category1", "Category", Default::default())
            .bind("TestCorpus", "category1", None)
            .with_user_context()
            .build();

        psiarc.write_diff(&diff1)?;
        original_lcn.apply_diff(&diff1)?;

        // User 2: Create a document
        let session2 = Session::new(user2.clone(), PersonaMode::Researcher);
        set_current_session(session2);

        let diff2 = ConceptDiffBuilder::new(2)
            .create("document1", "Document", Default::default())
            .bind("category1", "document1", None)
            .with_user_context()
            .build();

        psiarc.write_diff(&diff2)?;
        original_lcn.apply_diff(&diff2)?;

        // User 3: Add a section
        let session3 = Session::new(user3.clone(), PersonaMode::Glyphsmith);
        set_current_session(session3);

        let diff3 = ConceptDiffBuilder::new(3)
            .create("section1", "Section", Default::default())
            .bind("document1", "section1", None)
            .with_user_context()
            .build();

        psiarc.write_diff(&diff3)?;
        original_lcn.apply_diff(&diff3)?;

        // User 1: Modify the section
        let session4 = Session::new(user1.clone(), PersonaMode::MemoryPruner);
        set_current_session(session4);

        let mut props = std::collections::HashMap::new();
        props.insert("content".to_string(), serde_json::json!("Test content"));

        let diff4 = ConceptDiffBuilder::new(4)
            .update("section1", props)
            .with_user_context()
            .build();

        psiarc.write_diff(&diff4)?;
        original_lcn.apply_diff(&diff4)?;
    }

    // Close the PsiArc
    drop(psiarc);

    // Now replay the PsiArc into a new LCN
    let replay_lcn = LargeConceptNetwork::new();
    let reader = PsiArcReader::open(&psiarc_path)?;

    // Read and replay each frame
    for frame in reader.read_all_frames()? {
        replay_lcn.apply_diff(&frame.diff)?;

        // Verify user attribution in diff
        let diff = &frame.diff;
        if diff.frame_id > 0 {
            // Skip genesis frame
            assert!(diff.metadata.contains_key("user_id"));
            assert!(diff.metadata.contains_key("persona_id"));

            // Verify specific users based on frame ID
            match diff.frame_id {
                1 => {
                    assert_eq!(
                        diff.metadata.get("user_id").unwrap().as_str().unwrap(),
                        user1.concept_id
                    );
                    assert_eq!(
                        diff.metadata.get("persona_id").unwrap().as_str().unwrap(),
                        PersonaMode::CreativeAgent.as_str()
                    );
                }
                2 => {
                    assert_eq!(
                        diff.metadata.get("user_id").unwrap().as_str().unwrap(),
                        user2.concept_id
                    );
                    assert_eq!(
                        diff.metadata.get("persona_id").unwrap().as_str().unwrap(),
                        PersonaMode::Researcher.as_str()
                    );
                }
                3 => {
                    assert_eq!(
                        diff.metadata.get("user_id").unwrap().as_str().unwrap(),
                        user3.concept_id
                    );
                    assert_eq!(
                        diff.metadata.get("persona_id").unwrap().as_str().unwrap(),
                        PersonaMode::Glyphsmith.as_str()
                    );
                }
                4 => {
                    assert_eq!(
                        diff.metadata.get("user_id").unwrap().as_str().unwrap(),
                        user1.concept_id
                    );
                    assert_eq!(
                        diff.metadata.get("persona_id").unwrap().as_str().unwrap(),
                        PersonaMode::MemoryPruner.as_str()
                    );
                }
                _ => panic!("Unexpected frame ID: {}", diff.frame_id),
            }
        }
    }

    // Verify the replayed LCN matches the original
    assert_eq!(original_lcn.concepts_count(), replay_lcn.concepts_count());
    assert!(replay_lcn.has_concept("category1"));
    assert!(replay_lcn.has_concept("document1"));
    assert!(replay_lcn.has_concept("section1"));

    Ok(())
}
