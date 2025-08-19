//! Persona Memory
//!
//! This module provides functionality for storing and retrieving persona-specific 
//! preferences and memory, enabling personalized experiences based on user behavior.

use crate::auth::persona::PersonaMode;
use crate::auth::session::{get_current_session, Session};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn, error};

/// Persona preferences data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonaPreferences {
    /// Persona ID
    pub persona_id: String,
    
    /// User ID
    pub user_id: String,
    
    /// Favorite concepts
    #[serde(default)]
    pub favorite_concepts: Vec<String>,
    
    /// Frequently used concepts (with usage count)
    #[serde(default)]
    pub concept_usage: HashMap<String, u32>,
    
    /// Preferred phase state
    #[serde(default)]
    pub preferred_phase: Option<f64>,
    
    /// Preferred frequency
    #[serde(default)]
    pub preferred_frequency: Option<f64>,
    
    /// View preferences
    #[serde(default)]
    pub view_preferences: HashMap<String, serde_json::Value>,
    
    /// Agent pack preferences
    #[serde(default)]
    pub agent_pack_preferences: HashMap<String, bool>,
    
    /// Custom settings
    #[serde(default)]
    pub custom_settings: HashMap<String, serde_json::Value>,
    
    /// Last used timestamp
    #[serde(default)]
    pub last_used: Option<String>,
    
    /// Creation timestamp
    #[serde(default)]
    pub created_at: Option<String>,
}

impl PersonaPreferences {
    /// Create new persona preferences
    pub fn new(user_id: &str, persona_id: &str) -> Self {
        let now = chrono::Utc::now().to_rfc3339();
        
        Self {
            persona_id: persona_id.to_string(),
            user_id: user_id.to_string(),
            favorite_concepts: Vec::new(),
            concept_usage: HashMap::new(),
            preferred_phase: None,
            preferred_frequency: None,
            view_preferences: HashMap::new(),
            agent_pack_preferences: HashMap::new(),
            custom_settings: HashMap::new(),
            last_used: Some(now.clone()),
            created_at: Some(now),
        }
    }
    
    /// Add a concept to favorites
    pub fn add_favorite(&mut self, concept_id: &str) {
        if !self.favorite_concepts.contains(&concept_id.to_string()) {
            self.favorite_concepts.push(concept_id.to_string());
        }
    }
    
    /// Remove a concept from favorites
    pub fn remove_favorite(&mut self, concept_id: &str) {
        self.favorite_concepts.retain(|id| id != concept_id);
    }
    
    /// Record usage of a concept
    pub fn record_concept_usage(&mut self, concept_id: &str) {
        let count = self.concept_usage.entry(concept_id.to_string()).or_insert(0);
        *count += 1;
    }
    
    /// Set preferred phase state
    pub fn set_preferred_phase(&mut self, phase: f64) {
        self.preferred_phase = Some(phase);
    }
    
    /// Set preferred frequency
    pub fn set_preferred_frequency(&mut self, frequency: f64) {
        self.preferred_frequency = Some(frequency);
    }
    
    /// Set a view preference
    pub fn set_view_preference(&mut self, key: &str, value: serde_json::Value) {
        self.view_preferences.insert(key.to_string(), value);
    }
    
    /// Get a view preference
    pub fn get_view_preference(&self, key: &str) -> Option<&serde_json::Value> {
        self.view_preferences.get(key)
    }
    
    /// Set agent pack preference
    pub fn set_agent_pack_preference(&mut self, pack_id: &str, enabled: bool) {
        self.agent_pack_preferences.insert(pack_id.to_string(), enabled);
    }
    
    /// Check if agent pack is preferred
    pub fn is_agent_pack_preferred(&self, pack_id: &str) -> bool {
        self.agent_pack_preferences.get(pack_id).copied().unwrap_or(false)
    }
    
    /// Set a custom setting
    pub fn set_custom_setting(&mut self, key: &str, value: serde_json::Value) {
        self.custom_settings.insert(key.to_string(), value);
    }
    
    /// Get a custom setting
    pub fn get_custom_setting(&self, key: &str) -> Option<&serde_json::Value> {
        self.custom_settings.get(key)
    }
    
    /// Update last used timestamp
    pub fn update_last_used(&mut self) {
        self.last_used = Some(chrono::Utc::now().to_rfc3339());
    }
    
    /// Get most used concepts
    pub fn get_most_used_concepts(&self, limit: usize) -> Vec<(String, u32)> {
        let mut concepts: Vec<(String, u32)> = self.concept_usage.iter()
            .map(|(id, count)| (id.clone(), *count))
            .collect();
            
        concepts.sort_by(|a, b| b.1.cmp(&a.1));
        concepts.truncate(limit);
        
        concepts
    }
}

/// Persona memory store
pub struct PersonaMemoryStore {
    /// Base directory for persona memory
    base_dir: PathBuf,
    
    /// Cached preferences
    preferences_cache: HashMap<String, PersonaPreferences>,
}

impl PersonaMemoryStore {
    /// Create a new persona memory store
    pub fn new(base_dir: impl AsRef<Path>) -> Self {
        let base_dir = base_dir.as_ref().to_path_buf();
        
        Self {
            base_dir,
            preferences_cache: HashMap::new(),
        }
    }
    
    /// Initialize the store
    pub fn initialize(&mut self) -> Result<(), String> {
        // Ensure base directory exists
        if !self.base_dir.exists() {
            fs::create_dir_all(&self.base_dir)
                .map_err(|e| format!("Failed to create persona memory directory: {}", e))?;
        }
        
        info!("Initialized persona memory store in {}", self.base_dir.display());
        Ok(())
    }
    
    /// Get preferences file path for a persona
    fn get_preferences_path(&self, user_id: &str, persona_id: &str) -> PathBuf {
        let user_dir = self.base_dir.join(user_id);
        let persona_dir = user_dir.join(persona_id);
        persona_dir.join("preferences.json")
    }
    
    /// Get preferences for a persona
    pub fn get_preferences(&mut self, user_id: &str, persona_id: &str) -> Result<PersonaPreferences, String> {
        // Check cache first
        let cache_key = format!("{}:{}", user_id, persona_id);
        
        if let Some(prefs) = self.preferences_cache.get(&cache_key) {
            return Ok(prefs.clone());
        }
        
        // Load from file
        let path = self.get_preferences_path(user_id, persona_id);
        
        if path.exists() {
            let content = fs::read_to_string(&path)
                .map_err(|e| format!("Failed to read preferences file: {}", e))?;
                
            let preferences: PersonaPreferences = serde_json::from_str(&content)
                .map_err(|e| format!("Failed to parse preferences file: {}", e))?;
                
            // Cache preferences
            self.preferences_cache.insert(cache_key, preferences.clone());
            
            Ok(preferences)
        } else {
            // Create new preferences
            let preferences = PersonaPreferences::new(user_id, persona_id);
            
            // Save to file
            self.save_preferences(&preferences)?;
            
            // Cache preferences
            self.preferences_cache.insert(cache_key, preferences.clone());
            
            Ok(preferences)
        }
    }
    
    /// Save preferences for a persona
    pub fn save_preferences(&self, preferences: &PersonaPreferences) -> Result<(), String> {
        let path = self.get_preferences_path(&preferences.user_id, &preferences.persona_id);
        
        // Ensure directory exists
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create directory: {}", e))?;
        }
        
        // Serialize preferences
        let content = serde_json::to_string_pretty(preferences)
            .map_err(|e| format!("Failed to serialize preferences: {}", e))?;
            
        // Write to file
        fs::write(&path, content)
            .map_err(|e| format!("Failed to write preferences file: {}", e))?;
            
        debug!("Saved preferences for {}:{} to {}", 
              preferences.user_id, 
              preferences.persona_id, 
              path.display());
              
        Ok(())
    }
    
    /// Update preferences for a persona
    pub fn update_preferences<F>(&mut self, user_id: &str, persona_id: &str, updater: F) -> Result<PersonaPreferences, String>
    where
        F: FnOnce(&mut PersonaPreferences),
    {
        // Get current preferences
        let mut preferences = self.get_preferences(user_id, persona_id)?;
        
        // Update preferences
        updater(&mut preferences);
        
        // Update last used timestamp
        preferences.update_last_used();
        
        // Save preferences
        self.save_preferences(&preferences)?;
        
        // Update cache
        let cache_key = format!("{}:{}", user_id, persona_id);
        self.preferences_cache.insert(cache_key, preferences.clone());
        
        Ok(preferences)
    }
    
    /// Record concept usage for the current persona
    pub fn record_concept_usage(&mut self, concept_id: &str) -> Result<(), String> {
        if let Some(session) = get_current_session() {
            self.update_preferences(
                &session.user.concept_id,
                session.persona.mode.as_str(),
                |prefs| {
                    prefs.record_concept_usage(concept_id);
                }
            )?;
        }
        
        Ok(())
    }
    
    /// Add a concept to favorites for the current persona
    pub fn add_favorite(&mut self, concept_id: &str) -> Result<(), String> {
        if let Some(session) = get_current_session() {
            self.update_preferences(
                &session.user.concept_id,
                session.persona.mode.as_str(),
                |prefs| {
                    prefs.add_favorite(concept_id);
                }
            )?;
        }
        
        Ok(())
    }
    
    /// Remove a concept from favorites for the current persona
    pub fn remove_favorite(&mut self, concept_id: &str) -> Result<(), String> {
        if let Some(session) = get_current_session() {
            self.update_preferences(
                &session.user.concept_id,
                session.persona.mode.as_str(),
                |prefs| {
                    prefs.remove_favorite(concept_id);
                }
            )?;
        }
        
        Ok(())
    }
    
    /// Get favorites for the current persona
    pub fn get_favorites(&mut self) -> Result<Vec<String>, String> {
        if let Some(session) = get_current_session() {
            let prefs = self.get_preferences(&session.user.concept_id, session.persona.mode.as_str())?;
            Ok(prefs.favorite_concepts)
        } else {
            Ok(Vec::new())
        }
    }
    
    /// Get most used concepts for the current persona
    pub fn get_most_used_concepts(&mut self, limit: usize) -> Result<Vec<(String, u32)>, String> {
        if let Some(session) = get_current_session() {
            let prefs = self.get_preferences(&session.user.concept_id, session.persona.mode.as_str())?;
            Ok(prefs.get_most_used_concepts(limit))
        } else {
            Ok(Vec::new())
        }
    }
    
    /// Set phase state preference for the current persona
    pub fn set_phase_preference(&mut self, phase: f64, frequency: f64) -> Result<(), String> {
        if let Some(session) = get_current_session() {
            self.update_preferences(
                &session.user.concept_id,
                session.persona.mode.as_str(),
                |prefs| {
                    prefs.set_preferred_phase(phase);
                    prefs.set_preferred_frequency(frequency);
                }
            )?;
        }
        
        Ok(())
    }
    
    /// Get phase state preference for the current persona
    pub fn get_phase_preference(&mut self) -> Result<Option<(f64, f64)>, String> {
        if let Some(session) = get_current_session() {
            let prefs = self.get_preferences(&session.user.concept_id, session.persona.mode.as_str())?;
            
            match (prefs.preferred_phase, prefs.preferred_frequency) {
                (Some(phase), Some(frequency)) => Ok(Some((phase, frequency))),
                _ => Ok(None),
            }
        } else {
            Ok(None)
        }
    }
    
    /// Get preferences for all personas of a user
    pub fn get_all_preferences_for_user(&self, user_id: &str) -> Result<HashMap<String, PersonaPreferences>, String> {
        let user_dir = self.base_dir.join(user_id);
        let mut result = HashMap::new();
        
        if !user_dir.exists() {
            return Ok(result);
        }
        
        // Iterate through persona directories
        if let Ok(entries) = fs::read_dir(user_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                
                if path.is_dir() {
                    let persona_id = path.file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .to_string();
                        
                    let prefs_path = path.join("preferences.json");
                    
                    if prefs_path.exists() {
                        if let Ok(content) = fs::read_to_string(&prefs_path) {
                            if let Ok(prefs) = serde_json::from_str::<PersonaPreferences>(&content) {
                                result.insert(persona_id, prefs);
                            }
                        }
                    }
                }
            }
        }
        
        Ok(result)
    }
}

/// Global persona memory store
static mut PERSONA_MEMORY_STORE: Option<PersonaMemoryStore> = None;

/// Initialize the global persona memory store
pub fn initialize_memory_store(base_dir: impl AsRef<Path>) -> Result<(), String> {
    let mut store = PersonaMemoryStore::new(base_dir);
    store.initialize()?;
    
    // Store in global
    unsafe {
        PERSONA_MEMORY_STORE = Some(store);
    }
    
    Ok(())
}

/// Get the global persona memory store
pub fn get_memory_store() -> Result<&'static mut PersonaMemoryStore, String> {
    unsafe {
        PERSONA_MEMORY_STORE.as_mut().ok_or_else(|| "Persona memory store not initialized".to_string())
    }
}

/// Get preferences for the current persona
pub fn get_current_preferences() -> Result<PersonaPreferences, String> {
    if let Some(session) = get_current_session() {
        get_memory_store()?.get_preferences(&session.user.concept_id, session.persona.mode.as_str())
    } else {
        Err("No active session".to_string())
    }
}

/// Generate a usage heatmap for visualization
pub fn generate_usage_heatmap(user_id: &str, persona_id: &str) -> Result<HashMap<String, f64>, String> {
    // Get preferences
    let prefs = get_memory_store()?.get_preferences(user_id, persona_id)?;
    
    // Calculate total usage
    let total_usage: u32 = prefs.concept_usage.values().sum();
    
    if total_usage == 0 {
        return Ok(HashMap::new());
    }
    
    // Generate normalized heatmap
    let mut heatmap = HashMap::new();
    
    for (concept_id, count) in &prefs.concept_usage {
        let normalized_value = *count as f64 / total_usage as f64;
        heatmap.insert(concept_id.clone(), normalized_value);
    }
    
    Ok(heatmap)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::{User, PersonaMode, Session, set_current_session};
    use tempfile::tempdir;
    
    #[test]
    fn test_persona_preferences() {
        let mut prefs = PersonaPreferences::new("user1", "glyphsmith");
        
        // Test adding favorites
        prefs.add_favorite("concept1");
        prefs.add_favorite("concept2");
        assert_eq!(prefs.favorite_concepts, vec!["concept1", "concept2"]);
        
        // Test removing favorites
        prefs.remove_favorite("concept1");
        assert_eq!(prefs.favorite_concepts, vec!["concept2"]);
        
        // Test recording usage
        prefs.record_concept_usage("concept3");
        prefs.record_concept_usage("concept3");
        prefs.record_concept_usage("concept4");
        
        assert_eq!(prefs.concept_usage.get("concept3"), Some(&2));
        assert_eq!(prefs.concept_usage.get("concept4"), Some(&1));
        
        // Test most used concepts
        let most_used = prefs.get_most_used_concepts(2);
        assert_eq!(most_used[0].0, "concept3");
        assert_eq!(most_used[0].1, 2);
        assert_eq!(most_used[1].0, "concept4");
        assert_eq!(most_used[1].1, 1);
    }
    
    #[test]
    fn test_persona_memory_store() -> Result<(), String> {
        // Create temporary directory
        let temp_dir = tempdir().map_err(|e| format!("Failed to create temp dir: {}", e))?;
        
        // Create store
        let mut store = PersonaMemoryStore::new(temp_dir.path());
        store.initialize()?;
        
        // Create a test user and session
        let user = User::new(
            "github",
            "test_user",
            Some("test@example.com"),
            Some("Test User"),
            None,
        );
        
        // Get preferences (should create new preferences)
        let prefs = store.get_preferences(&user.concept_id, "glyphsmith")?;
        assert_eq!(prefs.user_id, user.concept_id);
        assert_eq!(prefs.persona_id, "glyphsmith");
        assert!(prefs.favorite_concepts.is_empty());
        
        // Update preferences
        store.update_preferences(&user.concept_id, "glyphsmith", |prefs| {
            prefs.add_favorite("concept1");
            prefs.record_concept_usage("concept1");
            prefs.set_preferred_phase(0.5);
            prefs.set_preferred_frequency(1.0);
        })?;
        
        // Get preferences again (should load from file)
        let prefs = store.get_preferences(&user.concept_id, "glyphsmith")?;
        assert_eq!(prefs.favorite_concepts, vec!["concept1"]);
        assert_eq!(prefs.concept_usage.get("concept1"), Some(&1));
        assert_eq!(prefs.preferred_phase, Some(0.5));
        assert_eq!(prefs.preferred_frequency, Some(1.0));
        
        Ok(())
    }
}
