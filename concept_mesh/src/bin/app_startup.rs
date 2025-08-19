//! Application Startup Flow
//!
//! This module implements the startup flow for the Concept Mesh application,
//! including OAuth configuration, session restoration, and agent pack loading.

use concept_mesh_rs::agents::persona_packs::{initialize_registry, update_packs_for_current_persona};
use concept_mesh_rs::auth::{
    cli::{handle_cli_auth, parse_cli_args},
    oauth::{init_oauth_providers, load_oauth_credentials, OAuthClient},
    session::{create_session_manager, get_current_session, SessionManager},
    user::{create_user_store, UserStore},
};
use dotenv::from_filename;
use std::path::PathBuf;
use tracing::{error, info, warn};

/// Result type for startup functions
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Application startup configuration
pub struct StartupConfig {
    /// Base directory for application data
    pub base_dir: PathBuf,

    /// User store directory
    pub user_store_dir: PathBuf,

    /// Agent packs directory
    pub agent_packs_dir: PathBuf,

    /// OAuth configuration file
    pub oauth_config_file: PathBuf,

    /// Check for existing session on startup
    pub restore_session: bool,

    /// Parse CLI authentication arguments
    pub enable_cli_auth: bool,
}

impl Default for StartupConfig {
    fn default() -> Self {
        let base_dir = dirs::data_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("concept-mesh");

        Self {
            user_store_dir: base_dir.join("users"),
            agent_packs_dir: base_dir.join("agent_packs"),
            oauth_config_file: PathBuf::from(".env.oauth"),
            base_dir,
            restore_session: true,
            enable_cli_auth: true,
        }
    }
}

/// Initialize the application
pub fn initialize_app(config: StartupConfig) -> Result<()> {
    info!("Initializing Concept Mesh application");

    // Ensure base directory exists
    std::fs::create_dir_all(&config.base_dir)?;

    // Initialize OAuth providers
    initialize_oauth(&config.oauth_config_file)?;

    // Initialize user store
    initialize_user_store(&config.user_store_dir)?;

    // Initialize agent packs
    initialize_agent_packs(&config.agent_packs_dir)?;

    // Check for CLI auth arguments
    if config.enable_cli_auth {
        handle_cli_authentication()?;
    }

    // Restore previous session if enabled
    if config.restore_session {
        restore_previous_session()?;
    }

    // Update agent packs for current persona (if any)
    update_agent_packs()?;

    info!("Application initialization complete");
    Ok(())
}

/// Initialize OAuth providers
fn initialize_oauth(config_file: &PathBuf) -> Result<()> {
    info!(
        "Initializing OAuth providers from {}",
        config_file.display()
    );

    // Load OAuth environment variables
    if config_file.exists() {
        from_filename(config_file)?;
        info!("Loaded OAuth configuration from {}", config_file.display());
    } else {
        warn!(
            "OAuth configuration file not found: {}",
            config_file.display()
        );
        warn!("OAuth functionality will be limited or unavailable");
    }

    // Initialize OAuth providers
    init_oauth_providers()?;

    info!("OAuth providers initialized");
    Ok(())
}

/// Initialize user store
fn initialize_user_store(user_store_dir: &PathBuf) -> Result<()> {
    info!("Initializing user store in {}", user_store_dir.display());

    // Ensure directory exists
    std::fs::create_dir_all(user_store_dir)?;

    // Set user store directory
    std::env::set_var(
        "CONCEPT_MESH_USER_STORE_DIR",
        user_store_dir.to_string_lossy().to_string(),
    );

    // Create user store to initialize it
    let _store = create_user_store()?;

    info!("User store initialized");
    Ok(())
}

/// Initialize agent packs
fn initialize_agent_packs(agent_packs_dir: &PathBuf) -> Result<()> {
    info!("Initializing agent packs in {}", agent_packs_dir.display());

    // Initialize registry with the agent packs directory
    initialize_registry(agent_packs_dir)?;

    info!("Agent packs initialized");
    Ok(())
}

/// Handle CLI authentication
fn handle_cli_authentication() -> Result<()> {
    info!("Checking for CLI authentication arguments");

    // Parse CLI arguments
    let args = parse_cli_args();

    // Check if impersonation is requested
    if args.impersonate.is_some() {
        // Handle CLI authentication
        let session = handle_cli_auth(args)?;

        info!(
            "CLI authentication successful: user={}, persona={}",
            session.user.name.as_deref().unwrap_or("Unknown"),
            session.persona.mode.as_str()
        );
    }

    Ok(())
}

/// Restore previous session
fn restore_previous_session() -> Result<()> {
    info!("Attempting to restore previous session");

    // Create session manager
    let mut session_manager = create_session_manager()?;

    // Try to restore previous session
    match session_manager.restore_previous_session() {
        Ok(Some(session)) => {
            info!(
                "Restored previous session: user={}, persona={}",
                session.user.name.as_deref().unwrap_or("Unknown"),
                session.persona.mode.as_str()
            );
            Ok(())
        }
        Ok(None) => {
            info!("No previous session to restore");
            Ok(())
        }
        Err(e) => {
            warn!("Failed to restore previous session: {}", e);
            Err(e.into())
        }
    }
}

/// Update agent packs for current persona
fn update_agent_packs() -> Result<()> {
    // Check if there is a current session
    if let Some(session) = get_current_session() {
        info!(
            "Updating agent packs for persona: {}",
            session.persona.mode.as_str()
        );
        update_packs_for_current_persona()?;
    } else {
        info!("No active session, skipping agent pack update");
    }

    Ok(())
}

/// Main function for standalone execution
pub fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    // Default configuration
    let config = StartupConfig::default();

    // Initialize the application
    initialize_app(config)?;

    // Print current session info
    if let Some(session) = get_current_session() {
        println!("Active session:");
        println!(
            "  User: {}",
            session
                .user
                .name
                .unwrap_or_else(|| session.user.concept_id.clone())
        );
        println!("  Persona: {}", session.persona.mode.as_str());
        println!("  Session ID: {}", session.id);
    } else {
        println!("No active session");
    }

    Ok(())
}

/// Integration with Tauri application
#[cfg(feature = "tauri")]
pub fn tauri_startup() -> Result<()> {
    use std::path::Path;

    // Determine base directory based on Tauri app directory
    let app_dir = tauri::api::path::app_dir().unwrap_or_else(|| PathBuf::from("."));
    let base_dir = app_dir.join("data");

    // Create custom configuration
    let config = StartupConfig {
        base_dir: base_dir.clone(),
        user_store_dir: base_dir.join("users"),
        agent_packs_dir: base_dir.join("agent_packs"),
        oauth_config_file: Path::new(".env.oauth").to_path_buf(),
        restore_session: true,
        enable_cli_auth: true,
    };

    // Initialize the application
    initialize_app(config)
}

