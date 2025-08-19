//! Persona management for Concept Mesh
//!
//! This module provides the PersonaMode enum and Persona struct for
//! representing different operational modes in the Concept Mesh. Each user
//! can have multiple personas, which affect how they interact with the system.

use rand::{thread_rng, Rng};
use serde::{Deserialize, Serialize};

/// User persona mode
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum PersonaMode {
    /// Creative Agent - focused on design and synthesis
    CreativeAgent,

    /// Glyphsmith - focused on ELFIN compiler and visual expressiveness
    Glyphsmith,

    /// Memory Pruner - focused on timeline reflection and psiarc editing
    MemoryPruner,

    /// Researcher - focused on AV/hologram ingestion and psiarc emission
    Researcher,

    /// Custom persona with a unique name
    Custom(String),
}

impl PersonaMode {
    /// Convert persona mode to string representation
    pub fn as_str(&self) -> &str {
        match self {
            Self::CreativeAgent => "creative_agent",
            Self::Glyphsmith => "glyphsmith",
            Self::MemoryPruner => "memory_pruner",
            Self::Researcher => "researcher",
            Self::Custom(s) => s,
        }
    }

    /// Parse persona mode from string
    pub fn from_str(s: &str) -> Self {
        match s {
            "creative_agent" => Self::CreativeAgent,
            "glyphsmith" => Self::Glyphsmith,
            "memory_pruner" => Self::MemoryPruner,
            "researcher" => Self::Researcher,
            custom => Self::Custom(custom.to_string()),
        }
    }

    /// Get all predefined persona modes
    pub fn all() -> Vec<Self> {
        vec![
            Self::CreativeAgent,
            Self::Glyphsmith,
            Self::MemoryPruner,
            Self::Researcher,
        ]
    }

    /// Get display name for the persona mode
    pub fn display_name(&self) -> String {
        match self {
            Self::CreativeAgent => "Creative Agent".to_string(),
            Self::Glyphsmith => "Glyphsmith".to_string(),
            Self::MemoryPruner => "Memory Pruner".to_string(),
            Self::Researcher => "Researcher".to_string(),
            Self::Custom(s) => s.clone(),
        }
    }

    /// Get description for the persona mode
    pub fn description(&self) -> String {
        match self {
            Self::CreativeAgent => "Design, synthesis, and creative exploration".to_string(),
            Self::Glyphsmith => {
                "ELFIN compiler, visual expressiveness, and glyph editing".to_string()
            }
            Self::MemoryPruner => {
                "Timeline reflection, psiarc editing, and memory organization".to_string()
            }
            Self::Researcher => {
                "AV/hologram ingestion, psiarc emission, and knowledge acquisition".to_string()
            }
            Self::Custom(_) => "Custom persona mode".to_string(),
        }
    }
}

/// User persona
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Persona {
    /// User ID this persona belongs to
    pub user_id: String,

    /// Persona mode
    pub mode: PersonaMode,

    /// Creation timestamp (ISO 8601)
    pub created_at: String,

    /// Last used timestamp (ISO 8601)
    pub last_used: String,

    /// Phase seed for oscillator initialization
    pub phase_seed: u64,

    /// Active agent pack IDs
    pub active_pack_ids: Vec<String>,
}

impl Persona {
    /// Create a new persona
    pub fn new(user_id: &str, mode: PersonaMode) -> Self {
        let mut rng = thread_rng();
        let now = chrono::Utc::now().to_rfc3339();

        Self {
            user_id: user_id.to_string(),
            mode,
            created_at: now.clone(),
            last_used: now,
            phase_seed: rng.gen(),
            active_pack_ids: Vec::new(),
        }
    }

    /// Update last used timestamp
    pub fn update_last_used(&mut self) {
        self.last_used = chrono::Utc::now().to_rfc3339();
    }

    /// Add an agent pack to this persona
    pub fn add_agent_pack(&mut self, pack_id: &str) {
        if !self.active_pack_ids.contains(&pack_id.to_string()) {
            self.active_pack_ids.push(pack_id.to_string());
        }
    }

    /// Remove an agent pack from this persona
    pub fn remove_agent_pack(&mut self, pack_id: &str) {
        self.active_pack_ids.retain(|id| id != pack_id);
    }

    /// Get display name for this persona
    pub fn display_name(&self) -> String {
        self.mode.display_name()
    }

    /// Get phase value in the range [0, 1) based on the seed
    pub fn phase(&self) -> f32 {
        // Convert the phase seed to a value in [0, 1)
        (self.phase_seed as f64 / u64::MAX as f64) as f32
    }

    /// Get angular phase in radians [0, 2π)
    pub fn angular_phase(&self) -> f32 {
        use std::f32::consts::TAU;
        self.phase() * TAU
    }
}

/// Store for user personas
pub struct PersonaStore {
    /// Path to the persona store directory
    path: std::path::PathBuf,

    /// In-memory cache of (user_id, mode) to Persona
    cache: std::collections::HashMap<(String, String), Persona>,
}

impl PersonaStore {
    /// Create a new persona store
    pub fn new(path: impl AsRef<std::path::Path>) -> Result<Self, std::io::Error> {
        let path = path.as_ref().to_path_buf();
        std::fs::create_dir_all(&path)?;

        Ok(Self {
            path,
            cache: std::collections::HashMap::new(),
        })
    }

    /// Save a persona to the store
    pub fn save_persona(&mut self, persona: &Persona) -> Result<(), std::io::Error> {
        let file_name = format!("{}_{}.json", persona.user_id, persona.mode.as_str());
        let persona_path = self.path.join(file_name);

        let json = serde_json::to_string_pretty(persona)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        std::fs::write(&persona_path, json)?;

        // Update cache
        let key = (persona.user_id.clone(), persona.mode.as_str().to_string());
        self.cache.insert(key, persona.clone());

        Ok(())
    }

    /// Get a persona by user ID and mode
    pub fn get_persona(
        &mut self,
        user_id: &str,
        mode: &PersonaMode,
    ) -> Result<Persona, std::io::Error> {
        let key = (user_id.to_string(), mode.as_str().to_string());

        // Check cache first
        if let Some(persona) = self.cache.get(&key) {
            return Ok(persona.clone());
        }

        // Load from disk
        let file_name = format!("{}_{}.json", user_id, mode.as_str());
        let persona_path = self.path.join(&file_name);

        if !persona_path.exists() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Persona not found: {}_{}", user_id, mode.as_str()),
            ));
        }

        let json = std::fs::read_to_string(persona_path)?;
        let persona: Persona = serde_json::from_str(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        // Update cache
        self.cache.insert(key, persona.clone());

        Ok(persona)
    }

    /// List all personas for a user
    pub fn list_personas_for_user(
        &mut self,
        user_id: &str,
    ) -> Result<Vec<Persona>, std::io::Error> {
        let mut personas = Vec::new();
        let prefix = format!("{}_", user_id);

        for entry in std::fs::read_dir(&self.path)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() && path.extension().map_or(false, |ext| ext == "json") {
                if let Some(file_name) = path.file_stem().and_then(|s| s.to_str()) {
                    if file_name.starts_with(&prefix) {
                        // Extract the mode from the filename
                        let mode_str = file_name[prefix.len()..].to_string();
                        let mode = PersonaMode::from_str(&mode_str);

                        match self.get_persona(user_id, &mode) {
                            Ok(persona) => personas.push(persona),
                            Err(e) => eprintln!("Failed to load persona {}: {}", file_name, e),
                        }
                    }
                }
            }
        }

        Ok(personas)
    }

    /// Delete a persona
    pub fn delete_persona(
        &mut self,
        user_id: &str,
        mode: &PersonaMode,
    ) -> Result<(), std::io::Error> {
        let file_name = format!("{}_{}.json", user_id, mode.as_str());
        let persona_path = self.path.join(&file_name);

        if persona_path.exists() {
            std::fs::remove_file(persona_path)?;
        }

        // Remove from cache
        let key = (user_id.to_string(), mode.as_str().to_string());
        self.cache.remove(&key);

        Ok(())
    }

    /// Get or create a persona
    pub fn get_or_create_persona(
        &mut self,
        user_id: &str,
        mode: PersonaMode,
    ) -> Result<Persona, std::io::Error> {
        match self.get_persona(user_id, &mode) {
            Ok(persona) => Ok(persona),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                let persona = Persona::new(user_id, mode);
                self.save_persona(&persona)?;
                Ok(persona)
            }
            Err(e) => Err(e),
        }
    }
}

/// Create a persona store instance
pub fn create_persona_store() -> Result<PersonaStore, std::io::Error> {
    // Create store in the default location
    PersonaStore::new("graph/personas")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_persona_creation() {
        let persona = Persona::new("USER_12345", PersonaMode::Glyphsmith);

        assert_eq!(persona.user_id, "USER_12345");
        assert_eq!(persona.mode, PersonaMode::Glyphsmith);
        assert!(persona.active_pack_ids.is_empty());

        // Phase should be in [0, 1)
        let phase = persona.phase();
        assert!(phase >= 0.0 && phase < 1.0);
    }

    #[test]
    fn test_persona_store() -> Result<(), std::io::Error> {
        let temp_dir = tempdir().unwrap();
        let mut store = PersonaStore::new(temp_dir.path())?;

        // Create and save a persona
        let mut persona = Persona::new("USER_12345", PersonaMode::Glyphsmith);
        persona.add_agent_pack("pack1");

        store.save_persona(&persona)?;

        // Get the persona
        let retrieved = store.get_persona("USER_12345", &PersonaMode::Glyphsmith)?;
        assert_eq!(retrieved.user_id, persona.user_id);
        assert_eq!(retrieved.mode, persona.mode);
        assert_eq!(retrieved.active_pack_ids, vec!["pack1"]);

        // List personas
        let personas = store.list_personas_for_user("USER_12345")?;
        assert_eq!(personas.len(), 1);

        // Delete persona
        store.delete_persona("USER_12345", &PersonaMode::Glyphsmith)?;

        // Check if deleted
        let result = store.get_persona("USER_12345", &PersonaMode::Glyphsmith);
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_persona_mode_conversions() {
        let mode = PersonaMode::Glyphsmith;
        let str_mode = mode.as_str();
        let parsed_mode = PersonaMode::from_str(str_mode);

        assert_eq!(mode, parsed_mode);

        // Test custom mode
        let custom = PersonaMode::Custom("TestMode".to_string());
        let str_custom = custom.as_str();
        let parsed_custom = PersonaMode::from_str(str_custom);

        if let PersonaMode::Custom(name) = parsed_custom {
            assert_eq!(name, "TestMode");
        } else {
            panic!("Expected Custom mode");
        }
    }

    #[test]
    fn test_get_or_create_persona() -> Result<(), std::io::Error> {
        let temp_dir = tempdir().unwrap();
        let mut store = PersonaStore::new(temp_dir.path())?;

        // Get or create a new persona
        let persona = store.get_or_create_persona("USER_12345", PersonaMode::Researcher)?;
        assert_eq!(persona.user_id, "USER_12345");
        assert_eq!(persona.mode, PersonaMode::Researcher);

        // Get the same persona again
        let retrieved = store.get_or_create_persona("USER_12345", PersonaMode::Researcher)?;
        assert_eq!(retrieved.user_id, persona.user_id);
        assert_eq!(retrieved.mode, persona.mode);
        assert_eq!(retrieved.phase_seed, persona.phase_seed);

        Ok(())
    }
}
