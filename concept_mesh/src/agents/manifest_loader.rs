//! Agent Pack Manifest Loader
//!
//! This module provides functionality for loading and validating agent pack
//! manifests, which define pack metadata, capabilities, dependencies, and
//! compatibility with different personas.

use std::path::{Path, PathBuf};
use std::fs;
use std::collections::HashMap;
use std::io;
use serde::{Serialize, Deserialize};
use crate::auth::persona::PersonaMode;
use tracing::{info, warn, error, debug};

/// Agent pack manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackManifest {
    /// Pack ID (unique identifier)
    pub id: String,
    
    /// Pack name (display name)
    pub name: String,
    
    /// Pack description
    pub description: String,
    
    /// Pack version
    pub version: String,
    
    /// Agents included in the pack
    pub agents: Vec<String>,
    
    /// Capabilities provided by the pack
    #[serde(default)]
    pub capabilities: Vec<String>,
    
    /// Dependencies required by the pack
    #[serde(default)]
    pub dependencies: Vec<String>,
    
    /// Compatibility with specific personas
    #[serde(default)]
    pub persona_compatibility: Vec<String>,
    
    /// Auto-suggest this pack when relevant
    #[serde(default)]
    pub auto_suggest: bool,
    
    /// Pack author
    #[serde(default)]
    pub author: Option<String>,
    
    /// Pack website
    #[serde(default)]
    pub website: Option<String>,
    
    /// Pack license
    #[serde(default)]
    pub license: Option<String>,
    
    /// Additional metadata
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl PackManifest {
    /// Check if the pack is compatible with a persona
    pub fn is_compatible_with(&self, persona: &PersonaMode) -> bool {
        // If no compatibility list is specified, assume compatible with all
        if self.persona_compatibility.is_empty() {
            return true;
        }
        
        // Check if the persona mode is in the compatibility list
        self.persona_compatibility.iter().any(|p| {
            p == persona.as_str() || p == "*"
        })
    }
    
    /// Get the path to the agent binary or script
    pub fn get_agent_path(&self, base_dir: &Path, agent_name: &str) -> Option<PathBuf> {
        if !self.agents.contains(&agent_name.to_string()) {
            return None;
        }
        
        let agent_path = base_dir.join("agents").join(agent_name);
        
        // Check for various extensions
        let extensions = ["", ".exe", ".js", ".py", ".wasm"];
        
        for ext in extensions {
            let path_with_ext = format!("{}{}", agent_path.to_string_lossy(), ext);
            let path = PathBuf::from(&path_with_ext);
            if path.exists() {
                return Some(path);
            }
        }
        
        None
    }
    
    /// Check if all dependencies are satisfied
    pub fn check_dependencies(&self, available_dependencies: &[String]) -> (bool, Vec<String>) {
        let mut missing = Vec::new();
        
        for dep in &self.dependencies {
            if !available_dependencies.contains(dep) {
                missing.push(dep.clone());
            }
        }
        
        (missing.is_empty(), missing)
    }
}

/// Result of manifest loading
#[derive(Debug)]
pub enum ManifestLoadResult {
    /// Successfully loaded manifest
    Success(PackManifest),
    
    /// Invalid manifest format
    InvalidFormat(String),
    
    /// File not found
    NotFound(String),
    
    /// IO error
    IoError(io::Error),
}

/// Load a manifest from a file
pub fn load_manifest(path: &Path) -> ManifestLoadResult {
    debug!("Loading manifest from {}", path.display());
    
    // Check if file exists
    if !path.exists() {
        return ManifestLoadResult::NotFound(format!("Manifest file not found: {}", path.display()));
    }
    
    // Read file content
    let content = match fs::read_to_string(path) {
        Ok(content) => content,
        Err(e) => return ManifestLoadResult::IoError(e),
    };
    
    // Parse JSON
    match serde_json::from_str::<PackManifest>(&content) {
        Ok(manifest) => {
            debug!("Successfully loaded manifest for pack '{}'", manifest.id);
            ManifestLoadResult::Success(manifest)
        },
        Err(e) => ManifestLoadResult::InvalidFormat(format!("Invalid manifest format: {}", e)),
    }
}

/// Load all manifests from a directory
pub fn load_manifests_from_dir(dir: &Path) -> HashMap<String, PackManifest> {
    let mut manifests = HashMap::new();
    
    // Check if directory exists
    if !dir.exists() || !dir.is_dir() {
        warn!("Agent pack directory does not exist or is not a directory: {}", dir.display());
        return manifests;
    }
    
    // Iterate through subdirectories
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            
            if path.is_dir() {
                let manifest_path = path.join("manifest.json");
                
                if manifest_path.exists() {
                    match load_manifest(&manifest_path) {
                        ManifestLoadResult::Success(manifest) => {
                            manifests.insert(manifest.id.clone(), manifest);
                        },
                        ManifestLoadResult::InvalidFormat(err) => {
                            warn!("Invalid manifest at {}: {}", manifest_path.display(), err);
                        },
                        ManifestLoadResult::NotFound(_) => {
                            // Should not happen since we checked existence
                        },
                        ManifestLoadResult::IoError(err) => {
                            warn!("Error reading manifest at {}: {}", manifest_path.display(), err);
                        },
                    }
                }
            }
        }
    }
    
    info!("Loaded {} agent pack manifests", manifests.len());
    manifests
}

/// Pack installation result
#[derive(Debug)]
pub enum InstallResult {
    /// Successfully installed
    Success(PackManifest),
    
    /// Invalid pack format
    InvalidFormat(String),
    
    /// Pack already exists
    AlreadyExists(String),
    
    /// IO error during installation
    IoError(io::Error),
    
    /// Missing dependencies
    MissingDependencies(Vec<String>),
}

/// Install a pack from a ZIP file
pub fn install_pack_from_zip(zip_path: &Path, packs_dir: &Path, dependencies: &[String]) -> InstallResult {
    info!("Installing agent pack from {}", zip_path.display());
    
    // Create temporary directory
    let temp_dir = match tempfile::tempdir() {
        Ok(dir) => dir,
        Err(e) => return InstallResult::IoError(io::Error::new(io::ErrorKind::Other, e)),
    };
    
    // Extract ZIP file
    let file = match fs::File::open(zip_path) {
        Ok(file) => file,
        Err(e) => return InstallResult::IoError(e),
    };
    
    let mut archive = match zip::ZipArchive::new(file) {
        Ok(archive) => archive,
        Err(e) => return InstallResult::InvalidFormat(format!("Invalid ZIP file: {}", e)),
    };
    
    // Extract all files
    for i in 0..archive.len() {
        let mut file = match archive.by_index(i) {
            Ok(file) => file,
            Err(e) => return InstallResult::InvalidFormat(format!("Error accessing ZIP entry: {}", e)),
        };
        
        let outpath = temp_dir.path().join(file.name());
        
        if file.name().ends_with('/') {
            // Create directory
            fs::create_dir_all(&outpath).map_err(InstallResult::IoError)?;
        } else {
            // Create parent directory if needed
            if let Some(parent) = outpath.parent() {
                fs::create_dir_all(parent).map_err(InstallResult::IoError)?;
            }
            
            // Extract file
            let mut outfile = fs::File::create(&outpath).map_err(InstallResult::IoError)?;
            io::copy(&mut file, &mut outfile).map_err(InstallResult::IoError)?;
        }
    }
    
    // Look for manifest
    let manifest_path = temp_dir.path().join("manifest.json");
    
    if !manifest_path.exists() {
        return InstallResult::InvalidFormat("manifest.json not found in ZIP file".to_string());
    }
    
    // Load manifest
    let manifest = match load_manifest(&manifest_path) {
        ManifestLoadResult::Success(manifest) => manifest,
        ManifestLoadResult::InvalidFormat(err) => return InstallResult::InvalidFormat(err),
        ManifestLoadResult::NotFound(_) => unreachable!(), // We already checked existence
        ManifestLoadResult::IoError(err) => return InstallResult::IoError(err),
    };
    
    // Check dependencies
    let (deps_satisfied, missing_deps) = manifest.check_dependencies(dependencies);
    
    if !deps_satisfied {
        return InstallResult::MissingDependencies(missing_deps);
    }
    
    // Check if pack already exists
    let pack_dir = packs_dir.join(&manifest.id);
    
    if pack_dir.exists() {
        return InstallResult::AlreadyExists(format!("Pack '{}' already exists", manifest.id));
    }
    
    // Copy files to pack directory
    fs::create_dir_all(&pack_dir).map_err(InstallResult::IoError)?;
    
    // Copy all files except manifest.json (we'll create it separately)
    if let Ok(entries) = fs::read_dir(temp_dir.path()) {
        for entry in entries.flatten() {
            let path = entry.path();
            let file_name = path.file_name().unwrap();
            
            if file_name != "manifest.json" {
                let dest_path = pack_dir.join(file_name);
                
                if path.is_dir() {
                    // Copy directory recursively
                    copy_dir_all(&path, &dest_path).map_err(InstallResult::IoError)?;
                } else {
                    // Copy file
                    fs::copy(&path, &dest_path).map_err(InstallResult::IoError)?;
                }
            }
        }
    }
    
    // Write manifest
    let manifest_content = serde_json::to_string_pretty(&manifest)
        .map_err(|e| InstallResult::InvalidFormat(format!("Error serializing manifest: {}", e)))?;
        
    fs::write(pack_dir.join("manifest.json"), manifest_content).map_err(InstallResult::IoError)?;
    
    info!("Successfully installed agent pack '{}'", manifest.id);
    InstallResult::Success(manifest)
}

/// Recursively copy a directory
fn copy_dir_all(src: &Path, dst: &Path) -> io::Result<()> {
    fs::create_dir_all(dst)?;
    
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        let path = entry.path();
        
        let filename = match path.file_name() {
            Some(name) => name,
            None => continue,
        };
        
        let dst_path = dst.join(filename);
        
        if ty.is_dir() {
            copy_dir_all(&path, &dst_path)?;
        } else {
            fs::copy(&path, &dst_path)?;
        }
    }
    
    Ok(())
}

/// CLI command for installing a pack
pub fn install_pack(zip_path: &str, packs_dir: &str) -> Result<(), String> {
    let zip_path = PathBuf::from(zip_path);
    let packs_dir = PathBuf::from(packs_dir);
    
    // Ensure packs directory exists
    if !packs_dir.exists() {
        fs::create_dir_all(&packs_dir)
            .map_err(|e| format!("Failed to create packs directory: {}", e))?;
    }
    
    // Get available dependencies
    let dependencies = get_available_dependencies(&packs_dir)?;
    
    // Install pack
    match install_pack_from_zip(&zip_path, &packs_dir, &dependencies) {
        InstallResult::Success(manifest) => {
            println!("Successfully installed pack '{}'", manifest.id);
            println!("  Name: {}", manifest.name);
            println!("  Description: {}", manifest.description);
            println!("  Version: {}", manifest.version);
            println!("  Agents: {}", manifest.agents.join(", "));
            Ok(())
        },
        InstallResult::InvalidFormat(err) => {
            Err(format!("Invalid pack format: {}", err))
        },
        InstallResult::AlreadyExists(err) => {
            Err(format!("Pack already exists: {}", err))
        },
        InstallResult::IoError(err) => {
            Err(format!("IO error: {}", err))
        },
        InstallResult::MissingDependencies(deps) => {
            Err(format!("Missing dependencies: {}", deps.join(", ")))
        },
    }
}

/// Get available dependencies from installed packs
fn get_available_dependencies(packs_dir: &Path) -> Result<Vec<String>, String> {
    let mut dependencies = Vec::new();
    
    // Add system dependencies
    dependencies.push("core".to_string());
    
    // Add dependencies from installed packs
    let manifests = load_manifests_from_dir(packs_dir);
    
    for manifest in manifests.values() {
        // Add pack ID as a dependency
        dependencies.push(manifest.id.clone());
        
        // Add capabilities as dependencies
        for capability in &manifest.capabilities {
            dependencies.push(capability.clone());
        }
    }
    
    Ok(dependencies)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[test]
    fn test_manifest_loading() {
        // Create a temporary directory
        let temp_dir = tempdir().unwrap();
        let manifest_path = temp_dir.path().join("manifest.json");
        
        // Create a sample manifest
        let manifest = r#"{
            "id": "test_pack",
            "name": "Test Pack",
            "description": "A test pack",
            "version": "1.0.0",
            "agents": ["agent1", "agent2"],
            "capabilities": ["capability1", "capability2"],
            "dependencies": ["core"],
            "persona_compatibility": ["glyphsmith", "researcher"],
            "auto_suggest": true,
            "author": "Test Author"
        }"#;
        
        fs::write(&manifest_path, manifest).unwrap();
        
        // Load the manifest
        match load_manifest(&manifest_path) {
            ManifestLoadResult::Success(manifest) => {
                assert_eq!(manifest.id, "test_pack");
                assert_eq!(manifest.name, "Test Pack");
                assert_eq!(manifest.agents, vec!["agent1", "agent2"]);
                assert_eq!(manifest.capabilities, vec!["capability1", "capability2"]);
                assert_eq!(manifest.dependencies, vec!["core"]);
                assert_eq!(manifest.persona_compatibility, vec!["glyphsmith", "researcher"]);
                assert_eq!(manifest.auto_suggest, true);
                assert_eq!(manifest.author, Some("Test Author".to_string()));
            },
            _ => panic!("Failed to load manifest"),
        }
    }
    
    #[test]
    fn test_manifest_compatibility() {
        let manifest = PackManifest {
            id: "test_pack".to_string(),
            name: "Test Pack".to_string(),
            description: "A test pack".to_string(),
            version: "1.0.0".to_string(),
            agents: vec!["agent1".to_string(), "agent2".to_string()],
            capabilities: vec!["capability1".to_string(), "capability2".to_string()],
            dependencies: vec!["core".to_string()],
            persona_compatibility: vec!["glyphsmith".to_string(), "researcher".to_string()],
            auto_suggest: true,
            author: Some("Test Author".to_string()),
            website: None,
            license: None,
            metadata: HashMap::new(),
        };
        
        // Test compatibility
        assert!(manifest.is_compatible_with(&PersonaMode::Glyphsmith));
        assert!(manifest.is_compatible_with(&PersonaMode::Researcher));
        assert!(!manifest.is_compatible_with(&PersonaMode::CreativeAgent));
        assert!(!manifest.is_compatible_with(&PersonaMode::MemoryPruner));
        
        // Test with wildcard
        let manifest_wildcard = PackManifest {
            persona_compatibility: vec!["*".to_string()],
            ..manifest.clone()
        };
        
        assert!(manifest_wildcard.is_compatible_with(&PersonaMode::CreativeAgent));
        assert!(manifest_wildcard.is_compatible_with(&PersonaMode::Glyphsmith));
        assert!(manifest_wildcard.is_compatible_with(&PersonaMode::MemoryPruner));
        assert!(manifest_wildcard.is_compatible_with(&PersonaMode::Researcher));
        
        // Test with empty compatibility (should be compatible with all)
        let manifest_empty = PackManifest {
            persona_compatibility: vec![],
            ..manifest.clone()
        };
        
        assert!(manifest_empty.is_compatible_with(&PersonaMode::CreativeAgent));
        assert!(manifest_empty.is_compatible_with(&PersonaMode::Glyphsmith));
        assert!(manifest_empty.is_compatible_with(&PersonaMode::MemoryPruner));
        assert!(manifest_empty.is_compatible_with(&PersonaMode::Researcher));
    }
}
