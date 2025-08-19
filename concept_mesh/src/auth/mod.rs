//! Authentication and identity module for Concept Mesh
//!
//! This module provides OAuth-based authentication, user identity management,
//! persona selection, and session tracking for the Concept Mesh. It integrates
//! with the ConceptDiff system to ensure all operations are properly attributed
//! to specific users and personas.

pub mod cli;
pub mod oauth;
pub mod persona;
pub mod session;
pub mod user;

// Re-export key components
pub use cli::handle_cli_auth;
pub use oauth::{OAuthCredentials, OAuthProvider, OAuthUser};
pub use persona::{Persona, PersonaMode};
pub use session::{get_current_session, set_current_session, Session, SessionManager};
pub use user::{User, UserStore};
