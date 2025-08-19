//! Session management for Concept Mesh
//!
//! This module provides session management for the Concept Mesh,
//! tracking the current user, active persona, and session metadata.
//! Sessions are used to attribute concept operations to specific users
//! and personas.

use super::{Persona, PersonaMode, User};
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use thiserror::Error;
use tracing::{debug, info, warn};

/// Session error types
#[derive(Debug, Error)]
pub enum SessionError {
    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// JSON error
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    /// No active session
    #[error("No active session")]
    NoActiveSession,

    /// Invalid session
    #[error("Invalid session: {0}")]
    InvalidSession(String),
}

/// User session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    /// Session ID (format: ψ-YYYY-MM-DDTHH:MM:SS.sssZ)
    pub id: String,

    /// User information
    pub user: User,

    /// Active persona
    pub persona: Persona,

    /// Creation timestamp (ISO 8601)
    pub created_at: String,

    /// Last active timestamp (ISO 8601)
    pub last_active: String,

    /// Phase state information (for embedding in ConceptDiffs)
    pub phase_state: PhaseState,
}

/// Phase state information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseState {
    /// Phase value [0, 1)
    pub phase: f32,

    /// Angular frequency
    pub frequency: f32,
}

impl Default for PhaseState {
    fn default() -> Self {
        Self {
            phase: 0.0,
            frequency: 1.0,
        }
    }
}

impl Session {
    /// Create a new session for a user with a specific persona mode
    pub fn new(user: User, mode: PersonaMode) -> Self {
        let now = chrono::Utc::now();
        let timestamp = now.format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string();
        let persona = Persona::new(&user.concept_id, mode);

        Self {
            id: format!("ψ-{}", timestamp),
            user,
            persona: persona.clone(),
            created_at: now.to_rfc3339(),
            last_active: now.to_rfc3339(),
            phase_state: PhaseState {
                phase: persona.phase(),
                frequency: 1.0,
            },
        }
    }

    /// Update last active timestamp
    pub fn update_activity(&mut self) {
        self.last_active = chrono::Utc::now().to_rfc3339();
    }

    /// Update phase state
    pub fn update_phase_state(&mut self, phase: f32, frequency: f32) {
        self.phase_state.phase = phase;
        self.phase_state.frequency = frequency;
    }

    /// Check if session is recent (within the last 24 hours)
    pub fn is_recent(&self) -> bool {
        use chrono::DateTime;

        if let Ok(last_active) = DateTime::parse_from_rfc3339(&self.last_active) {
            let now = chrono::Utc::now();
            let duration = now.signed_duration_since(last_active.with_timezone(&chrono::Utc));

            duration.num_hours() < 24
        } else {
            false
        }
    }

    /// Get session display information
    pub fn display_info(&self) -> String {
        format!(
            "{} as {} ({})",
            self.user.display_name(),
            self.persona.display_name(),
            self.id
        )
    }
}

/// Session manager
pub struct SessionManager {
    /// Path to the session store directory
    path: PathBuf,

    /// Currently active session
    current: Option<Session>,
}

impl SessionManager {
    /// Create a new session manager
    pub fn new(path: impl AsRef<Path>) -> Result<Self, SessionError> {
        let path = path.as_ref().to_path_buf();
        fs::create_dir_all(&path)?;

        Ok(Self {
            path,
            current: None,
        })
    }

    /// Start a new session
    pub fn start_session(
        &mut self,
        user: User,
        mode: PersonaMode,
    ) -> Result<Session, SessionError> {
        let session = Session::new(user, mode);
        self.save_session(&session)?;
        self.current = Some(session.clone());

        // Update global session
        set_current_session(session.clone());

        info!("Started session: {}", session.display_info());
        Ok(session)
    }

    /// Save the current session
    pub fn save_session(&self, session: &Session) -> Result<(), SessionError> {
        // Save current session
        let current_path = self.path.join("current.json");
        let json = serde_json::to_string_pretty(session)?;
        fs::write(&current_path, json)?;

        // Also save to history
        let history_path = self.path.join(format!("{}.json", session.id));
        let json = serde_json::to_string_pretty(session)?;
        fs::write(&history_path, json)?;

        Ok(())
    }

    /// Load the current session
    pub fn load_session(&mut self) -> Result<Session, SessionError> {
        let current_path = self.path.join("current.json");

        if !current_path.exists() {
            return Err(SessionError::NoActiveSession);
        }

        let json = fs::read_to_string(&current_path)?;
        let session: Session = serde_json::from_str(&json)?;

        self.current = Some(session.clone());

        // Update global session
        set_current_session(session.clone());

        Ok(session)
    }

    /// Get the current session
    pub fn get_current(&self) -> Option<Session> {
        self.current.clone()
    }

    /// End the current session
    pub fn end_session(&mut self) -> Result<(), SessionError> {
        let current_path = self.path.join("current.json");

        if current_path.exists() {
            fs::remove_file(&current_path)?;
        }

        if let Some(session) = &self.current {
            info!("Ended session: {}", session.display_info());
        }

        self.current = None;

        // Update global session
        clear_current_session();

        Ok(())
    }

    /// Switch to a different persona
    pub fn switch_persona(&mut self, mode: PersonaMode) -> Result<Session, SessionError> {
        if let Some(mut session) = self.current.clone() {
            let new_persona = Persona::new(&session.user.concept_id, mode);
            session.persona = new_persona;
            session.update_activity();

            // Update phase state
            session.phase_state.phase = session.persona.phase();

            self.save_session(&session)?;
            self.current = Some(session.clone());

            // Update global session
            set_current_session(session.clone());

            info!("Switched persona: {}", session.display_info());
            Ok(session)
        } else {
            Err(SessionError::NoActiveSession)
        }
    }

    /// List recent sessions
    pub fn list_recent_sessions(&self) -> Result<Vec<Session>, SessionError> {
        let mut sessions = Vec::new();

        for entry in fs::read_dir(&self.path)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() && path.extension().map_or(false, |ext| ext == "json") {
                if let Some(filename) = path.file_stem().and_then(|s| s.to_str()) {
                    if filename != "current" && filename.starts_with("ψ-") {
                        let json = fs::read_to_string(&path)?;
                        let session: Session = serde_json::from_str(&json)?;

                        if session.is_recent() {
                            sessions.push(session);
                        }
                    }
                }
            }
        }

        // Sort by last_active (most recent first)
        sessions.sort_by(|a, b| b.last_active.cmp(&a.last_active));

        Ok(sessions)
    }

    /// Resume a previous session
    pub fn resume_session(&mut self, session_id: &str) -> Result<Session, SessionError> {
        let session_path = self.path.join(format!("{}.json", session_id));

        if !session_path.exists() {
            return Err(SessionError::InvalidSession(format!(
                "Session not found: {}",
                session_id
            )));
        }

        let json = fs::read_to_string(&session_path)?;
        let mut session: Session = serde_json::from_str(&json)?;

        // Update activity timestamp
        session.update_activity();

        self.save_session(&session)?;
        self.current = Some(session.clone());

        // Update global session
        set_current_session(session.clone());

        info!("Resumed session: {}", session.display_info());
        Ok(session)
    }
}

// Global session storage
static GLOBAL_SESSION: Lazy<Arc<Mutex<Option<Session>>>> = Lazy::new(|| Arc::new(Mutex::new(None)));

/// Get the current session
pub fn get_current_session() -> Option<Session> {
    GLOBAL_SESSION.lock().unwrap().clone()
}

/// Set the current session
pub fn set_current_session(session: Session) {
    *GLOBAL_SESSION.lock().unwrap() = Some(session);
}

/// Clear the current session
pub fn clear_current_session() {
    *GLOBAL_SESSION.lock().unwrap() = None;
}

/// Create a session manager instance
pub fn create_session_manager() -> Result<SessionManager, SessionError> {
    // Create manager in the default location
    SessionManager::new("graph/sessions")
}

/// Load or initialize the session
pub fn load_or_init_session() -> Result<Option<Session>, SessionError> {
    let mut manager = create_session_manager()?;

    match manager.load_session() {
        Ok(session) => {
            debug!("Loaded existing session: {}", session.display_info());
            Ok(Some(session))
        }
        Err(SessionError::NoActiveSession) => {
            debug!("No active session found");
            Ok(None)
        }
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::user::User;
    use tempfile::tempdir;

    #[test]
    fn test_session_creation() {
        let user = User::new(
            "github",
            "12345",
            Some("user@example.com"),
            Some("Test User"),
            Some("https://example.com/avatar.png"),
        );

        let session = Session::new(user, PersonaMode::CreativeAgent);

        assert!(session.id.starts_with("ψ-"));
        assert_eq!(session.persona.mode, PersonaMode::CreativeAgent);

        // Phase state should be set
        assert!(session.phase_state.phase >= 0.0 && session.phase_state.phase < 1.0);
        assert_eq!(session.phase_state.frequency, 1.0);
    }

    #[test]
    fn test_session_manager() -> Result<(), SessionError> {
        let temp_dir = tempdir().unwrap();
        let mut manager = SessionManager::new(temp_dir.path())?;

        // Create a user
        let user = User::new(
            "github",
            "12345",
            Some("user@example.com"),
            Some("Test User"),
            Some("https://example.com/avatar.png"),
        );

        // Start a session
        let session = manager.start_session(user.clone(), PersonaMode::Glyphsmith)?;
        assert_eq!(session.persona.mode, PersonaMode::Glyphsmith);

        // Get current session
        let current = manager.get_current().unwrap();
        assert_eq!(current.id, session.id);

        // Switch persona
        let new_session = manager.switch_persona(PersonaMode::Researcher)?;
        assert_eq!(new_session.persona.mode, PersonaMode::Researcher);

        // End session
        manager.end_session()?;
        assert!(manager.get_current().is_none());

        Ok(())
    }

    #[test]
    fn test_global_session() {
        // Initially empty
        assert!(get_current_session().is_none());

        // Create a user and session
        let user = User::new(
            "github",
            "12345",
            Some("user@example.com"),
            Some("Test User"),
            Some("https://example.com/avatar.png"),
        );

        let session = Session::new(user, PersonaMode::CreativeAgent);

        // Set global session
        set_current_session(session.clone());

        // Get global session
        let global = get_current_session().unwrap();
        assert_eq!(global.id, session.id);

        // Clear global session
        clear_current_session();
        assert!(get_current_session().is_none());
    }

    #[test]
    fn test_session_resume() -> Result<(), SessionError> {
        let temp_dir = tempdir().unwrap();
        let mut manager = SessionManager::new(temp_dir.path())?;

        // Create a user
        let user = User::new(
            "github",
            "12345",
            Some("user@example.com"),
            Some("Test User"),
            Some("https://example.com/avatar.png"),
        );

        // Start a session
        let session = manager.start_session(user.clone(), PersonaMode::Glyphsmith)?;
        let session_id = session.id.clone();

        // End the session
        manager.end_session()?;

        // Resume the session
        let resumed = manager.resume_session(&session_id)?;
        assert_eq!(resumed.id, session_id);
        assert_eq!(resumed.user.concept_id, user.concept_id);
        assert_eq!(resumed.persona.mode, PersonaMode::Glyphsmith);

        Ok(())
    }
}
