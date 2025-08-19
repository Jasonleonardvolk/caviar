//! CLI authentication for Concept Mesh
//!
//! This module provides command-line authentication and impersonation
//! functionality for the Concept Mesh. It allows users to start a session
//! from the command line without going through the OAuth flow, which is
//! useful for development and testing.

use super::{set_current_session, Persona, PersonaMode, Session, User};
use clap::{Arg, ArgMatches, Command};
use tracing::{debug, info, warn};

/// Handle CLI authentication based on command-line arguments
pub fn handle_cli_auth(matches: &ArgMatches) -> Option<Session> {
    if let Some(username) = matches.get_one::<String>("impersonate") {
        // Get persona mode (default to creative_agent if not specified)
        let mode_str = matches
            .get_one::<String>("persona")
            .map(|s| s.as_str())
            .unwrap_or("creative_agent");

        let mode = PersonaMode::from_str(mode_str);

        // Create impersonated user
        let user = User {
            concept_id: format!("USER_{}", username),
            provider: "cli".to_string(),
            provider_id: username.to_string(),
            email: Some(format!("{}@example.com", username)),
            name: Some(username.to_string()),
            avatar_url: None,
            joined_at: chrono::Utc::now().to_rfc3339(),
        };

        // Create persona
        let persona = Persona::new(&user.concept_id, mode);

        // Create session
        let session = Session::new(user, mode);

        // Set global session
        set_current_session(session.clone());

        info!(
            "Impersonating user: {} as {}",
            username,
            mode.display_name()
        );

        // Save session to file
        if let Ok(mut session_manager) = super::session::create_session_manager() {
            if let Err(e) = session_manager.save_session(&session) {
                warn!("Failed to save impersonated session: {}", e);
            }
        }

        return Some(session);
    }

    None
}

/// Add CLI authentication arguments to a command
pub fn add_cli_auth_args(app: Command) -> Command {
    app.arg(
        Arg::new("impersonate")
            .long("impersonate")
            .help("Impersonate a user for development")
            .value_name("USERNAME")
            .value_parser(clap::value_parser!(String)),
    )
    .arg(
        Arg::new("persona")
            .long("persona")
            .help("Set the active persona mode")
            .value_name("PERSONA")
            .value_parser([
                "creative_agent",
                "glyphsmith",
                "memory_pruner",
                "researcher",
            ])
            .requires("impersonate"),
    )
}

/// Check if impersonation is active from environment variables
pub fn check_env_impersonation() -> Option<Session> {
    let username = std::env::var("TORI_IMPERSONATE").ok()?;
    let mode_str = std::env::var("TORI_PERSONA").unwrap_or_else(|_| "creative_agent".to_string());

    let mode = PersonaMode::from_str(&mode_str);

    // Create impersonated user
    let user = User {
        concept_id: format!("USER_{}", username),
        provider: "cli".to_string(),
        provider_id: username.to_string(),
        email: Some(format!("{}@example.com", username)),
        name: Some(username.to_string()),
        avatar_url: None,
        joined_at: chrono::Utc::now().to_rfc3339(),
    };

    // Create session
    let session = Session::new(user, mode);

    // Set global session
    set_current_session(session.clone());

    info!(
        "Impersonating user from environment: {} as {}",
        username,
        mode.display_name()
    );

    // Save session to file
    if let Ok(mut session_manager) = super::session::create_session_manager() {
        if let Err(e) = session_manager.save_session(&session) {
            warn!("Failed to save impersonated session: {}", e);
        }
    }

    Some(session)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Command;

    #[test]
    fn test_cli_impersonation() {
        let app = Command::new("test").args(vec![Arg::new("foo").long("foo")]);

        let app = add_cli_auth_args(app);

        let matches = app
            .clone()
            .try_get_matches_from(vec![
                "test",
                "--impersonate",
                "testuser",
                "--persona",
                "glyphsmith",
            ])
            .unwrap();

        let session = handle_cli_auth(&matches).unwrap();

        assert_eq!(session.user.concept_id, "USER_testuser");
        assert_eq!(session.user.provider, "cli");
        assert_eq!(session.user.name, Some("testuser".to_string()));
        assert_eq!(session.persona.mode, PersonaMode::Glyphsmith);
    }

    #[test]
    fn test_cli_impersonation_default_persona() {
        let app = Command::new("test");
        let app = add_cli_auth_args(app);

        let matches = app
            .clone()
            .try_get_matches_from(vec!["test", "--impersonate", "testuser"])
            .unwrap();

        let session = handle_cli_auth(&matches).unwrap();

        assert_eq!(session.user.concept_id, "USER_testuser");
        assert_eq!(session.persona.mode, PersonaMode::CreativeAgent);
    }

    #[test]
    fn test_no_impersonation() {
        let app = Command::new("test");
        let app = add_cli_auth_args(app);

        let matches = app.clone().try_get_matches_from(vec!["test"]).unwrap();

        let session = handle_cli_auth(&matches);

        assert!(session.is_none());
    }
}
