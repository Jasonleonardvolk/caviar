//! ConceptDiff user context extension
//!
//! This module extends ConceptDiff with user, persona, and session information,
//! ensuring that all diffs are properly attributed to specific users and personas.

use super::{ConceptDiff, ConceptDiffBuilder};
use crate::auth::session::{get_current_session, Session};
use serde_json::json;
use tracing::debug;

/// Extension trait for ConceptDiffBuilder to add user context
pub trait UserContextExt {
    /// Add user context to the ConceptDiff
    fn with_user_context(self) -> Self;
}

impl UserContextExt for ConceptDiffBuilder {
    /// Add user context to the ConceptDiff if a session is active
    ///
    /// This adds user_id, persona_id, session_id, and phase_state to the diff's
    /// metadata, which allows operations to be properly attributed to specific
    /// users and personas.
    fn with_user_context(self) -> Self {
        if let Some(session) = get_current_session() {
            // Add user information
            let builder = self
                .with_metadata("user_id", session.user.concept_id.clone())
                .with_metadata("persona_id", session.persona.mode.as_str());

            // Add session information
            let builder = builder.with_metadata("session_id", session.id.clone());

            // Add phase state
            builder.with_metadata(
                "phase_state",
                json!({
                    "phase": session.phase_state.phase,
                    "frequency": session.phase_state.frequency
                }),
            )
        } else {
            debug!("No active session, ConceptDiff will not have user context");
            self
        }
    }
}

/// Get current session context info for inclusion in ConceptDiffs
///
/// Returns a tuple of (user_id, persona_id, session_id) if a session is active,
/// or None if no session is active.
pub fn get_context_info() -> Option<(String, String, String)> {
    get_current_session().map(|session| {
        (
            session.user.concept_id,
            session.persona.mode.as_str().to_string(),
            session.id,
        )
    })
}

/// Add user context to a ConceptDiff
///
/// This is a standalone function that can be used outside the builder pattern.
pub fn add_user_context(diff: &mut ConceptDiff) {
    if let Some((user_id, persona_id, session_id)) = get_context_info() {
        if let Ok(user_id_value) = serde_json::to_value(user_id) {
            diff.metadata.insert("user_id".to_string(), user_id_value);
        }

        if let Ok(persona_id_value) = serde_json::to_value(persona_id) {
            diff.metadata
                .insert("persona_id".to_string(), persona_id_value);
        }

        if let Ok(session_id_value) = serde_json::to_value(session_id) {
            diff.metadata
                .insert("session_id".to_string(), session_id_value);
        }

        if let Some(session) = get_current_session() {
            if let Ok(phase_state) = serde_json::to_value(json!({
                "phase": session.phase_state.phase,
                "frequency": session.phase_state.frequency
            })) {
                diff.metadata.insert("phase_state".to_string(), phase_state);
            }
        }
    } else {
        debug!("No active session, ConceptDiff will not have user context");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::PersonaMode;
    use crate::auth::User;

    #[test]
    fn test_with_user_context_no_session() {
        let diff = ConceptDiffBuilder::new(42)
            .with_metadata("test", "value")
            .with_user_context()
            .build();

        assert_eq!(
            diff.metadata.get("test").unwrap().as_str().unwrap(),
            "value"
        );
        assert!(!diff.metadata.contains_key("user_id"));
        assert!(!diff.metadata.contains_key("persona_id"));
        assert!(!diff.metadata.contains_key("session_id"));
    }

    #[test]
    fn test_add_user_context_to_diff() {
        let mut diff = ConceptDiff::new(42);

        // Initially no user context
        assert!(!diff.metadata.contains_key("user_id"));

        // Add user context (does nothing since no session is active)
        add_user_context(&mut diff);

        // Still no user context
        assert!(!diff.metadata.contains_key("user_id"));
    }
}
