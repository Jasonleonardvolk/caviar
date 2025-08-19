//! Persona-based Agent Pack Loading
//!
//! This module provides functionality for dynamically loading agent packs
//! based on the active persona. Each persona can have different agent
//! packs associated with it, which are automatically loaded when the
//! persona is activated.

use crate::auth::session::{get_current_session, Session};
use crate::auth::persona::PersonaMode;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

/// Agent pack type
#[derive(Debug, Clone)]
pub struct AgentPack {
    /// Pack ID
    pub id: String,
    
    /// Pack name
    pub name: String,
    
    /// Pack description
    pub description: String,
    
    /// Pack version
    pub version: String,
    
    /// Pack path
    pub path: PathBuf,
    
    /// Pack agents
    pub agents: Vec<String>,
    
    /// Pack is active
    pub active: bool,
}

/// Agent pack registry
pub struct AgentPackRegistry {
    /// All available packs
    packs: HashMap<String, AgentPack>,
    
    /// Active packs
    active_packs: Vec<String>,
    
    /// Pack directory
    pack_dir: PathBuf,
}

impl AgentPackRegistry {
    /// Create a new agent pack registry
    pub fn new(pack_dir: impl AsRef<Path>) -> Self {
        let pack_dir = pack_dir.as_ref().to_path_buf();
        
        Self {
            packs: HashMap::new(),
            active_packs: Vec::new(),
            pack_dir,
        }
    }
    
    /// Initialize the registry
    pub fn initialize(&mut self) -> Result<(), String> {
        info!("Initializing agent pack registry from {}", self.pack_dir.display());
        
        // Scan for agent packs
        self.scan_packs()?;
        
        // Load default packs
        self.load_default_packs()?;
        
        Ok(())
    }
    
    /// Scan for agent packs
    pub fn scan_packs(&mut self) -> Result<(), String> {
        // Ensure pack directory exists
        if !self.pack_dir.exists() {
            std::fs::create_dir_all(&self.pack_dir)
                .map_err(|e| format!("Failed to create pack directory: {}", e))?;
        }
        
        // Scan for pack manifests
        for entry in std::fs::read_dir(&self.pack_dir)
            .map_err(|e| format!("Failed to read pack directory: {}", e))? 
        {
            let entry = entry.map_err(|e| format!("Failed to read directory entry: {}", e))?;
            let path = entry.path();
            
            if path.is_dir() {
                let manifest_path = path.join("manifest.json");
                if manifest_path.exists() {
                    // Load pack manifest
                    self.load_pack_manifest(&manifest_path)?;
                }
            }
        }
        
        info!("Discovered {} agent packs", self.packs.len());
        Ok(())
    }
    
    /// Load a pack manifest
    fn load_pack_manifest(&mut self, path: &Path) -> Result<(), String> {
        #[derive(serde::Deserialize)]
        struct PackManifest {
            id: String,
            name: String,
            description: String,
            version: String,
            agents: Vec<String>,
        }
        
        // Read and parse manifest
        let manifest_content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read manifest file: {}", e))?;
            
        let manifest: PackManifest = serde_json::from_str(&manifest_content)
            .map_err(|e| format!("Failed to parse manifest file: {}", e))?;
            
        // Create agent pack
        let pack = AgentPack {
            id: manifest.id.clone(),
            name: manifest.name,
            description: manifest.description,
            version: manifest.version,
            path: path.parent().unwrap().to_path_buf(),
            agents: manifest.agents,
            active: false,
        };
        
        // Add to registry
        self.packs.insert(manifest.id, pack);
        
        debug!("Loaded agent pack: {}", path.display());
        Ok(())
    }
    
    /// Load default packs
    fn load_default_packs(&mut self) -> Result<(), String> {
        // Load core pack if available
        if let Some(pack) = self.packs.get_mut("core") {
            self.activate_pack(&pack.id)?;
        }
        
        Ok(())
    }
    
    /// Activate a pack
    pub fn activate_pack(&mut self, pack_id: &str) -> Result<(), String> {
        // Ensure pack exists
        if !self.packs.contains_key(pack_id) {
            return Err(format!("Pack not found: {}", pack_id));
        }
        
        // Skip if already active
        if self.active_packs.contains(&pack_id.to_string()) {
            return Ok(());
        }
        
        // Activate pack
        if let Some(pack) = self.packs.get_mut(pack_id) {
            pack.active = true;
            
            // Add to active packs
            self.active_packs.push(pack_id.to_string());
            
            info!("Activated agent pack: {}", pack_id);
        }
        
        Ok(())
    }
    
    /// Deactivate a pack
    pub fn deactivate_pack(&mut self, pack_id: &str) -> Result<(), String> {
        // Ensure pack exists
        if !self.packs.contains_key(pack_id) {
            return Err(format!("Pack not found: {}", pack_id));
        }
        
        // Skip if not active
        if !self.active_packs.contains(&pack_id.to_string()) {
            return Ok(());
        }
        
        // Deactivate pack
        if let Some(pack) = self.packs.get_mut(pack_id) {
            pack.active = false;
            
            // Remove from active packs
            self.active_packs.retain(|id| id != pack_id);
            
            info!("Deactivated agent pack: {}", pack_id);
        }
        
        Ok(())
    }
    
    /// Get all packs
    pub fn get_packs(&self) -> Vec<&AgentPack> {
        self.packs.values().collect()
    }
    
    /// Get active packs
    pub fn get_active_packs(&self) -> Vec<&AgentPack> {
        self.active_packs.iter()
            .filter_map(|id| self.packs.get(id))
            .collect()
    }
    
    /// Get a specific pack
    pub fn get_pack(&self, pack_id: &str) -> Option<&AgentPack> {
        self.packs.get(pack_id)
    }
    
    /// Update packs based on current persona
    pub fn update_for_persona(&mut self) -> Result<(), String> {
        // Get current session
        if let Some(session) = get_current_session() {
            self.update_for_specific_persona(&session.persona.mode)
        } else {
            Ok(()) // No session, no change
        }
    }
    
    /// Update packs for a specific persona
    pub fn update_for_specific_persona(&mut self, mode: &PersonaMode) -> Result<(), String> {
        // Get packs for this persona
        let packs = get_packs_for_persona(mode);
        
        // First deactivate all packs except core
        let current_active = self.active_packs.clone();
        for pack_id in current_active {
            if pack_id != "core" {
                self.deactivate_pack(&pack_id)?;
            }
        }
        
        // Then activate all packs for this persona
        for pack_id in packs {
            if self.packs.contains_key(&pack_id) {
                self.activate_pack(&pack_id)?;
            } else {
                warn!("Persona pack not found: {}", pack_id);
            }
        }
        
        info!("Updated agent packs for persona: {}", mode.as_str());
        Ok(())
    }
}

/// Global agent pack registry
pub static mut AGENT_PACK_REGISTRY: Option<AgentPackRegistry> = None;

/// Initialize the global agent pack registry
pub fn initialize_registry(pack_dir: impl AsRef<Path>) -> Result<(), String> {
    let mut registry = AgentPackRegistry::new(pack_dir);
    registry.initialize()?;
    
    // Store in global
    unsafe {
        AGENT_PACK_REGISTRY = Some(registry);
    }
    
    Ok(())
}

/// Get the global agent pack registry
pub fn get_registry() -> Result<&'static mut AgentPackRegistry, String> {
    unsafe {
        AGENT_PACK_REGISTRY.as_mut().ok_or_else(|| "Agent pack registry not initialized".to_string())
    }
}

/// Update agent packs for the current persona
pub fn update_packs_for_current_persona() -> Result<(), String> {
    get_registry()?.update_for_persona()
}

/// Get packs for a specific persona
pub fn get_packs_for_persona(mode: &PersonaMode) -> Vec<String> {
    match mode {
        PersonaMode::CreativeAgent => vec![
            "core".to_string(),
            "creative_tools".to_string(),
            "synthesis".to_string(),
        ],
        PersonaMode::Glyphsmith => vec![
            "core".to_string(),
            "elfin_compiler".to_string(),
            "glyph_tools".to_string(),
        ],
        PersonaMode::MemoryPruner => vec![
            "core".to_string(),
            "memory_tools".to_string(),
            "psiarc_editor".to_string(),
        ],
        PersonaMode::Researcher => vec![
            "core".to_string(),
            "av_toolkit".to_string(),
            "ingest_tools".to_string(),
        ],
        PersonaMode::Custom(name) => {
            // Custom personas get core + any packs that match their name
            let mut packs = vec!["core".to_string()];
            if let Some(custom_pack) = name.to_lowercase().replace(' ', "_").into() {
                packs.push(custom_pack);
            }
            packs
        },
    }
}

/// Create example agent pack manifests
pub fn create_example_packs(pack_dir: impl AsRef<Path>) -> Result<(), String> {
    let pack_dir = pack_dir.as_ref().to_path_buf();
    
    // Ensure directory exists
    std::fs::create_dir_all(&pack_dir)
        .map_err(|e| format!("Failed to create pack directory: {}", e))?;
    
    // Create core pack
    let core_dir = pack_dir.join("core");
    std::fs::create_dir_all(&core_dir)
        .map_err(|e| format!("Failed to create core pack directory: {}", e))?;
        
    let core_manifest = serde_json::json!({
        "id": "core",
        "name": "Core Agent Pack",
        "description": "Core agents for the Concept Mesh",
        "version": "1.0.0",
        "agents": [
            "orchestrator",
            "planner",
            "diff_generator",
        ]
    });
    
    std::fs::write(
        core_dir.join("manifest.json"),
        serde_json::to_string_pretty(&core_manifest)
            .map_err(|e| format!("Failed to serialize core manifest: {}", e))?,
    ).map_err(|e| format!("Failed to write core manifest: {}", e))?;
    
    // Create creative tools pack
    let creative_dir = pack_dir.join("creative_tools");
    std::fs::create_dir_all(&creative_dir)
        .map_err(|e| format!("Failed to create creative tools pack directory: {}", e))?;
        
    let creative_manifest = serde_json::json!({
        "id": "creative_tools",
        "name": "Creative Tools",
        "description": "Tools for creative synthesis and design",
        "version": "1.0.0",
        "agents": [
            "design_agent",
            "synthesis_agent",
        ]
    });
    
    std::fs::write(
        creative_dir.join("manifest.json"),
        serde_json::to_string_pretty(&creative_manifest)
            .map_err(|e| format!("Failed to serialize creative manifest: {}", e))?,
    ).map_err(|e| format!("Failed to write creative manifest: {}", e))?;
    
    // Create ELFIN compiler pack
    let elfin_dir = pack_dir.join("elfin_compiler");
    std::fs::create_dir_all(&elfin_dir)
        .map_err(|e| format!("Failed to create ELFIN compiler pack directory: {}", e))?;
        
    let elfin_manifest = serde_json::json!({
        "id": "elfin_compiler",
        "name": "ELFIN Compiler",
        "description": "ELFIN language compiler and tools",
        "version": "1.0.0",
        "agents": [
            "elfin_compiler",
            "elfin_linter",
        ]
    });
    
    std::fs::write(
        elfin_dir.join("manifest.json"),
        serde_json::to_string_pretty(&elfin_manifest)
            .map_err(|e| format!("Failed to serialize ELFIN manifest: {}", e))?,
    ).map_err(|e| format!("Failed to write ELFIN manifest: {}", e))?;
    
    // Create AV toolkit pack
    let av_dir = pack_dir.join("av_toolkit");
    std::fs::create_dir_all(&av_dir)
        .map_err(|e| format!("Failed to create AV toolkit pack directory: {}", e))?;
        
    let av_manifest = serde_json::json!({
        "id": "av_toolkit",
        "name": "AV Toolkit",
        "description": "Audio and video processing tools",
        "version": "1.0.0",
        "agents": [
            "audio_processor",
            "video_processor",
        ]
    });
    
    std::fs::write(
        av_dir.join("manifest.json"),
        serde_json::to_string_pretty(&av_manifest)
            .map_err(|e| format!("Failed to serialize AV manifest: {}", e))?,
    ).map_err(|e| format!("Failed to write AV manifest: {}", e))?;
    
    info!("Created example agent packs in {}", pack_dir.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::{User, PersonaMode, Session, set_current_session};
    use tempfile::tempdir;
    
    #[test]
    fn test_agent_pack_registry() -> Result<(), String> {
        // Create temporary directory for packs
        let temp_dir = tempdir().map_err(|e| format!("Failed to create temp dir: {}", e))?;
        
        // Create example packs
        create_example_packs(&temp_dir)?;
        
        // Create registry
        let mut registry = AgentPackRegistry::new(&temp_dir);
        registry.initialize()?;
        
        // Check packs were loaded
        assert!(registry.packs.contains_key("core"));
        assert!(registry.packs.contains_key("creative_tools"));
        assert!(registry.packs.contains_key("elfin_compiler"));
        assert!(registry.packs.contains_key("av_toolkit"));
        
        // Check core pack is active by default
        assert!(registry.active_packs.contains(&"core".to_string()));
        
        // Test activating packs for a persona
        registry.update_for_specific_persona(&PersonaMode::Glyphsmith)?;
        
        // Check expected packs are active
        assert!(registry.active_packs.contains(&"core".to_string()));
        assert!(registry.active_packs.contains(&"elfin_compiler".to_string()));
        assert!(registry.active_packs.contains(&"glyph_tools".to_string()) == false); // This one doesn't exist
        
        Ok(())
    }
    
    #[test]
    fn test_update_for_current_persona() -> Result<(), String> {
        // Create temporary directory for packs
        let temp_dir = tempdir().map_err(|e| format!("Failed to create temp dir: {}", e))?;
        
        // Create example packs
        create_example_packs(&temp_dir)?;
        
        // Set up a session
        let user = User::new(
            "test",
            "test_user",
            Some("test@example.com"),
            Some("Test User"),
            None,
        );
        
        let session = Session::new(user, PersonaMode::Researcher);
        set_current_session(session);
        
        // Initialize global registry
        initialize_registry(&temp_dir)?;
        
        // Update for current persona
        update_packs_for_current_persona()?;
        
        // Check expected packs are active
        let registry = get_registry()?;
        assert!(registry.active_packs.contains(&"core".to_string()));
        assert!(registry.active_packs.contains(&"av_toolkit".to_string()));
        
        Ok(())
    }
}
