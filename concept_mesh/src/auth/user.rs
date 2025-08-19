//! User identity management for Concept Mesh
//!
//! This module provides the User struct and UserStore for managing
//! user identities within the Concept Mesh. Each user is represented
//! as a concept in the graph with provider-specific information.

use crate::diff::{ConceptDiff, ConceptDiffBuilder};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;
use tracing::{debug, info, warn};

/// User identity error types
#[derive(Debug, Error)]
pub enum UserError {
    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// JSON error
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    /// User not found
    #[error("User not found: {0}")]
    NotFound(String),

    /// User already exists
    #[error("User already exists: {0}")]
    AlreadyExists(String),
}

/// User identity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    /// Concept ID
    pub concept_id: String,

    /// Provider (e.g., "github", "google")
    pub provider: String,

    /// Provider-specific ID
    pub provider_id: String,

    /// Email address (if available)
    pub email: Option<String>,

    /// User name (if available)
    pub name: Option<String>,

    /// Avatar URL (if available)
    pub avatar_url: Option<String>,

    /// Joined timestamp (ISO 8601)
    pub joined_at: String,
}

impl User {
    /// Create a new user from provider information
    pub fn new(
        provider: &str,
        provider_id: &str,
        email: Option<&str>,
        name: Option<&str>,
        avatar_url: Option<&str>,
    ) -> Self {
        // Create a hash of provider + provider_id for the concept_id
        let hash_input = format!("{}:{}", provider, provider_id);
        let mut hasher = Sha256::new();
        hasher.update(hash_input.as_bytes());
        let hash = format!("{:x}", hasher.finalize());
        let hash_short = &hash[0..7];

        Self {
            concept_id: format!("USER_{}", hash_short),
            provider: provider.to_string(),
            provider_id: provider_id.to_string(),
            email: email.map(|s| s.to_string()),
            name: name.map(|s| s.to_string()),
            avatar_url: avatar_url.map(|s| s.to_string()),
            joined_at: chrono::Utc::now().to_rfc3339(),
        }
    }

    /// Convert user to a ConceptDiff that creates the user concept
    pub fn to_concept_diff(&self) -> ConceptDiff {
        let mut attributes = HashMap::new();
        attributes.insert("provider".to_string(), self.provider.clone());
        attributes.insert("provider_id".to_string(), self.provider_id.clone());

        if let Some(ref email) = self.email {
            attributes.insert("email".to_string(), email.clone());
        }

        if let Some(ref name) = self.name {
            attributes.insert("name".to_string(), name.clone());
        }

        if let Some(ref avatar_url) = self.avatar_url {
            attributes.insert("avatar_url".to_string(), avatar_url.clone());
        }

        attributes.insert("joined_at".to_string(), self.joined_at.clone());

        ConceptDiffBuilder::new(0) // Frame ID will be assigned later
            .create_with_attributes(self.concept_id.clone(), "User".to_string(), attributes)
            .build()
    }

    /// Get user display name
    pub fn display_name(&self) -> String {
        self.name
            .clone()
            .or_else(|| {
                self.email
                    .clone()
                    .map(|e| e.split('@').next().unwrap_or("").to_string())
            })
            .unwrap_or_else(|| format!("User {}", &self.concept_id[5..]))
    }
}

/// Store for user identities
pub struct UserStore {
    /// Path to the user store directory
    path: PathBuf,

    /// In-memory cache of user_id to User
    cache: HashMap<String, User>,

    /// Provider ID to concept ID index
    provider_index: HashMap<String, String>,
}

impl UserStore {
    /// Create a new user store
    pub fn new(path: impl AsRef<Path>) -> Result<Self, UserError> {
        let path = path.as_ref().to_path_buf();
        fs::create_dir_all(&path)?;

        let mut store = Self {
            path,
            cache: HashMap::new(),
            provider_index: HashMap::new(),
        };

        // Load index if it exists
        store.load_index()?;

        Ok(store)
    }

    /// Save a user to the store
    pub fn save_user(&mut self, user: &User) -> Result<(), UserError> {
        let user_path = self.path.join(format!("{}.json", user.concept_id));
        let json = serde_json::to_string_pretty(user)?;
        fs::write(&user_path, json)?;

        // Update index
        let key = format!("{}:{}", user.provider, user.provider_id);
        self.provider_index.insert(key, user.concept_id.clone());
        self.save_index()?;

        // Update cache
        self.cache.insert(user.concept_id.clone(), user.clone());

        info!("Saved user {} ({})", user.concept_id, user.display_name());
        Ok(())
    }

    /// Get a user by concept ID
    pub fn get_user(&mut self, concept_id: &str) -> Result<User, UserError> {
        // Check cache first
        if let Some(user) = self.cache.get(concept_id) {
            return Ok(user.clone());
        }

        // Load from disk
        let user_path = self.path.join(format!("{}.json", concept_id));
        if !user_path.exists() {
            return Err(UserError::NotFound(concept_id.to_string()));
        }

        let json = fs::read_to_string(&user_path)?;
        let user: User = serde_json::from_str(&json)?;

        // Update cache
        self.cache.insert(user.concept_id.clone(), user.clone());

        Ok(user)
    }

    /// Get a user by provider information
    pub fn get_user_by_provider(
        &mut self,
        provider: &str,
        provider_id: &str,
    ) -> Result<User, UserError> {
        let key = format!("{}:{}", provider, provider_id);

        if let Some(concept_id) = self.provider_index.get(&key) {
            return self.get_user(concept_id);
        }

        Err(UserError::NotFound(format!(
            "Provider {}:{}",
            provider, provider_id
        )))
    }

    /// Check if a user exists by provider information
    pub fn user_exists(&self, provider: &str, provider_id: &str) -> bool {
        let key = format!("{}:{}", provider, provider_id);
        self.provider_index.contains_key(&key)
    }

    /// List all users
    pub fn list_users(&mut self) -> Result<Vec<User>, UserError> {
        let mut users = Vec::new();

        for entry in fs::read_dir(&self.path)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() && path.extension().map_or(false, |ext| ext == "json") {
                if let Some(filename) = path.file_stem().and_then(|s| s.to_str()) {
                    if filename != "index" {
                        match self.get_user(filename) {
                            Ok(user) => users.push(user),
                            Err(e) => warn!("Failed to load user {}: {}", filename, e),
                        }
                    }
                }
            }
        }

        Ok(users)
    }

    /// Delete a user
    pub fn delete_user(&mut self, concept_id: &str) -> Result<(), UserError> {
        // Get user first to update index
        let user = self.get_user(concept_id)?;

        // Remove from filesystem
        let user_path = self.path.join(format!("{}.json", concept_id));
        fs::remove_file(user_path)?;

        // Remove from cache
        self.cache.remove(concept_id);

        // Remove from index
        let key = format!("{}:{}", user.provider, user.provider_id);
        self.provider_index.remove(&key);
        self.save_index()?;

        info!("Deleted user {} ({})", concept_id, user.display_name());
        Ok(())
    }

    /// Load the provider index from disk
    fn load_index(&mut self) -> Result<(), UserError> {
        let index_path = self.path.join("index.json");

        if !index_path.exists() {
            return Ok(());
        }

        let json = fs::read_to_string(index_path)?;
        self.provider_index = serde_json::from_str(&json)?;

        Ok(())
    }

    /// Save the provider index to disk
    fn save_index(&self) -> Result<(), UserError> {
        let index_path = self.path.join("index.json");
        let json = serde_json::to_string_pretty(&self.provider_index)?;
        fs::write(index_path, json)?;

        Ok(())
    }
}

/// Create a user store instance
pub fn create_user_store() -> Result<UserStore, UserError> {
    // Create store in the default location
    UserStore::new("graph/users")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_user_creation() {
        let user = User::new(
            "github",
            "12345",
            Some("user@example.com"),
            Some("Test User"),
            Some("https://example.com/avatar.png"),
        );

        assert!(user.concept_id.starts_with("USER_"));
        assert_eq!(user.provider, "github");
        assert_eq!(user.provider_id, "12345");
        assert_eq!(user.email, Some("user@example.com".to_string()));
        assert_eq!(user.name, Some("Test User".to_string()));
        assert_eq!(
            user.avatar_url,
            Some("https://example.com/avatar.png".to_string())
        );
    }

    #[test]
    fn test_user_to_concept_diff() {
        let user = User::new(
            "github",
            "12345",
            Some("user@example.com"),
            Some("Test User"),
            Some("https://example.com/avatar.png"),
        );

        let diff = user.to_concept_diff();

        assert_eq!(diff.frame_id, 0);
        assert!(diff.operations.len() == 1);

        if let crate::diff::Operation::Create {
            id,
            concept_type,
            attributes,
        } = &diff.operations[0]
        {
            assert_eq!(id, &user.concept_id);
            assert_eq!(concept_type, "User");
            assert_eq!(attributes.get("provider").unwrap(), "github");
            assert_eq!(attributes.get("provider_id").unwrap(), "12345");
            assert_eq!(attributes.get("email").unwrap(), "user@example.com");
            assert_eq!(attributes.get("name").unwrap(), "Test User");
            assert_eq!(
                attributes.get("avatar_url").unwrap(),
                "https://example.com/avatar.png"
            );
        } else {
            panic!("Expected Create operation");
        }
    }

    #[test]
    fn test_user_store() -> Result<(), UserError> {
        let temp_dir = tempdir().unwrap();
        let mut store = UserStore::new(temp_dir.path())?;

        // Create and save a user
        let user = User::new(
            "github",
            "12345",
            Some("user@example.com"),
            Some("Test User"),
            Some("https://example.com/avatar.png"),
        );

        store.save_user(&user)?;

        // Get the user by concept_id
        let retrieved_user = store.get_user(&user.concept_id)?;
        assert_eq!(retrieved_user.provider, user.provider);
        assert_eq!(retrieved_user.provider_id, user.provider_id);

        // Get by provider
        let provider_user = store.get_user_by_provider("github", "12345")?;
        assert_eq!(provider_user.concept_id, user.concept_id);

        // Check existence
        assert!(store.user_exists("github", "12345"));
        assert!(!store.user_exists("github", "54321"));

        // List users
        let users = store.list_users()?;
        assert_eq!(users.len(), 1);

        // Delete user
        store.delete_user(&user.concept_id)?;
        assert!(!store.user_exists("github", "12345"));

        Ok(())
    }
}
