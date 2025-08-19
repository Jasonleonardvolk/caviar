//! Authentication command handlers for Tauri
//!
//! This module provides Tauri command handlers for user authentication,
//! including OAuth flows and persona selection.

use crate::auth::{
    oauth::{OAuthClient, OAuthCredentials, OAuthProvider, OAuthError, load_oauth_credentials},
    user::{User, UserStore, create_user_store},
    persona::{Persona, PersonaMode, create_persona_store},
    session::{Session, SessionManager, create_session_manager, set_current_session},
};
use serde::{Serialize, Deserialize};
use std::process::Command;
use tracing::{debug, info, warn, error};

/// Response for OAuth login requests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoginResponse {
    /// User information
    pub user: User,
    
    /// Success flag
    pub success: bool,
    
    /// Error message (if any)
    pub error: Option<String>,
    
    /// URL to open in browser (if required)
    pub auth_url: Option<String>,
}

/// Response for persona selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonaResponse {
    /// Session information
    pub session: Session,
    
    /// Success flag
    pub success: bool,
    
    /// Error message (if any)
    pub error: Option<String>,
}

/// All available persona modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonaModes {
    /// List of persona modes
    pub modes: Vec<PersonaModeInfo>,
}

/// Information about a persona mode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonaModeInfo {
    /// Mode ID
    pub id: String,
    
    /// Display name
    pub name: String,
    
    /// Description
    pub description: String,
}

/// Start OAuth login process
///
/// This generates an OAuth URL for the specified provider and returns it
/// to the UI, which can then open it in a browser or WebView.
#[tauri::command]
pub async fn login_oauth(provider: String) -> Result<LoginResponse, String> {
    // Create OAuth client
    let client = OAuthClient::new();
    
    // Load OAuth credentials
    let credentials = match load_oauth_credentials(&provider) {
        Ok(creds) => creds,
        Err(e) => {
            error!("Failed to load OAuth credentials: {}", e);
            return Ok(LoginResponse {
                user: User::new("unknown", "unknown", None, None, None),
                success: false,
                error: Some(format!("Failed to load OAuth credentials: {}", e)),
                auth_url: None,
            });
        }
    };
    
    // Generate authorization URL
    let auth_url = match client.authorization_url(&credentials) {
        Ok(url) => url,
        Err(e) => {
            error!("Failed to generate authorization URL: {}", e);
            return Ok(LoginResponse {
                user: User::new("unknown", "unknown", None, None, None),
                success: false,
                error: Some(format!("Failed to generate authorization URL: {}", e)),
                auth_url: None,
            });
        }
    };
    
    // Return the URL to the UI
    Ok(LoginResponse {
        user: User::new("unknown", "unknown", None, None, None),
        success: true,
        error: None,
        auth_url: Some(auth_url),
    })
}

/// Handle OAuth callback
///
/// This is called when the OAuth provider redirects back to the app
/// with an authorization code. It exchanges the code for a token,
/// fetches user information, and creates a user in the store.
#[tauri::command]
pub async fn handle_oauth_callback(
    provider: String,
    code: String,
) -> Result<LoginResponse, String> {
    // Create OAuth client
    let client = OAuthClient::new();
    
    // Load OAuth credentials
    let credentials = match load_oauth_credentials(&provider) {
        Ok(creds) => creds,
        Err(e) => {
            error!("Failed to load OAuth credentials: {}", e);
            return Ok(LoginResponse {
                user: User::new("unknown", "unknown", None, None, None),
                success: false,
                error: Some(format!("Failed to load OAuth credentials: {}", e)),
                auth_url: None,
            });
        }
    };
    
    // Exchange code for tokens and user info
    let oauth_user = match client.exchange_code(&credentials, &code).await {
        Ok(user) => user,
        Err(e) => {
            error!("Failed to exchange OAuth code: {}", e);
            return Ok(LoginResponse {
                user: User::new("unknown", "unknown", None, None, None),
                success: false,
                error: Some(format!("Failed to exchange OAuth code: {}", e)),
                auth_url: None,
            });
        }
    };
    
    // Create or update user in store
    let user = User::new(
        provider.as_str(),
        &oauth_user.provider_id,
        oauth_user.email.as_deref(),
        oauth_user.name.as_deref(),
        oauth_user.avatar_url.as_deref(),
    );
    
    // Save user to store
    match create_user_store() {
        Ok(mut store) => {
            if let Err(e) = store.save_user(&user) {
                warn!("Failed to save user to store: {}", e);
            }
        },
        Err(e) => {
            warn!("Failed to create user store: {}", e);
        }
    }
    
    // Return success with user info
    Ok(LoginResponse {
        user,
        success: true,
        error: None,
        auth_url: None,
    })
}

/// Get available persona modes
///
/// This returns a list of all available persona modes for the UI to display.
#[tauri::command]
pub fn get_persona_modes() -> PersonaModes {
    let modes = PersonaMode::all();
    let mode_infos = modes.into_iter().map(|mode| {
        PersonaModeInfo {
            id: mode.as_str().to_string(),
            name: mode.display_name(),
            description: mode.description(),
        }
    }).collect();
    
    PersonaModes {
        modes: mode_infos,
    }
}

/// Select a persona
///
/// This creates a new session with the specified user and persona mode.
#[tauri::command]
pub fn select_persona(user_id: String, persona_mode: String) -> Result<PersonaResponse, String> {
    // Get user from store
    let user = match create_user_store() {
        Ok(mut store) => {
            match store.get_user(&user_id) {
                Ok(user) => user,
                Err(e) => {
                    error!("Failed to get user from store: {}", e);
                    return Err(format!("Failed to get user from store: {}", e));
                }
            }
        },
        Err(e) => {
            error!("Failed to create user store: {}", e);
            return Err(format!("Failed to create user store: {}", e));
        }
    };
    
    // Parse persona mode
    let mode = PersonaMode::from_str(&persona_mode);
    
    // Create session
    let session = match create_session_manager() {
        Ok(mut manager) => {
            match manager.start_session(user, mode) {
                Ok(session) => session,
                Err(e) => {
                    error!("Failed to start session: {}", e);
                    return Err(format!("Failed to start session: {}", e));
                }
            }
        },
        Err(e) => {
            error!("Failed to create session manager: {}", e);
            return Err(format!("Failed to create session manager: {}", e));
        }
    };
    
    // Return success with session info
    Ok(PersonaResponse {
        session,
        success: true,
        error: None,
    })
}

/// Get current session
///
/// This returns the current active session, if any.
#[tauri::command]
pub fn get_current_session() -> Option<Session> {
    crate::auth::session::get_current_session()
}

/// Switch to a different persona
///
/// This changes the active persona for the current session.
#[tauri::command]
pub fn switch_persona(persona_mode: String) -> Result<PersonaResponse, String> {
    // Parse persona mode
    let mode = PersonaMode::from_str(&persona_mode);
    
    // Switch persona
    let session = match create_session_manager() {
        Ok(mut manager) => {
            match manager.switch_persona(mode) {
                Ok(session) => session,
                Err(e) => {
                    error!("Failed to switch persona: {}", e);
                    return Err(format!("Failed to switch persona: {}", e));
                }
            }
        },
        Err(e) => {
            error!("Failed to create session manager: {}", e);
            return Err(format!("Failed to create session manager: {}", e));
        }
    };
    
    // Return success with session info
    Ok(PersonaResponse {
        session,
        success: true,
        error: None,
    })
}

/// Logout
///
/// This ends the current session.
#[tauri::command]
pub fn logout() -> Result<(), String> {
    // End session
    match create_session_manager() {
        Ok(mut manager) => {
            if let Err(e) = manager.end_session() {
                error!("Failed to end session: {}", e);
                return Err(format!("Failed to end session: {}", e));
            }
        },
        Err(e) => {
            error!("Failed to create session manager: {}", e);
            return Err(format!("Failed to create session manager: {}", e));
        }
    }
    
    Ok(())
}
