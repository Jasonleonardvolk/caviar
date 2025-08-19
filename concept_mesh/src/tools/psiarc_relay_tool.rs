//! PsiArc Relay Tool
//!
//! This module provides tools for replaying PsiArc files while preserving privacy,
//! allowing collaborative sharing of concept operations without exposing sensitive
//! user information.

use crate::auth::{User, PersonaMode, Session};
use crate::diff::{ConceptDiff, ConceptDiffBuilder, UserContextExt};
use crate::psiarc::{PsiArc, PsiArcReader, PsiArcBuilder, PsiArcConfig, PsiArcFrame};
use std::path::{Path, PathBuf};
use tracing::{info, warn, error};
use std::collections::HashMap;
use std::io::Result as IoResult;

/// PsiArc relay configuration
#[derive(Debug, Clone)]
pub struct RelayConfig {
    /// Source PsiArc file
    pub source_path: PathBuf,
    
    /// Destination PsiArc file
    pub dest_path: PathBuf,
    
    /// Whether to anonymize user attribution
    pub anonymize: bool,
    
    /// Whether to preserve persona information even when anonymizing
    pub preserve_personas: bool,
    
    /// Whether to preserve phase states
    pub preserve_phase: bool,
    
    /// Custom metadata to add to relayed diffs
    pub custom_metadata: HashMap<String, serde_json::Value>,
}

impl Default for RelayConfig {
    fn default() -> Self {
        Self {
            source_path: PathBuf::new(),
            dest_path: PathBuf::new(),
            anonymize: true,
            preserve_personas: true,
            preserve_phase: true,
            custom_metadata: HashMap::new(),
        }
    }
}

/// Process a PsiArc file, optionally anonymizing attribution data
pub fn relay_psiarc(config: &RelayConfig) -> IoResult<()> {
    info!("Relaying PsiArc from {} to {}", 
         config.source_path.display(), 
         config.dest_path.display());
    
    // Open source PsiArc
    let reader = PsiArcReader::open(&config.source_path)?;
    
    // Create destination PsiArc
    let mut psiarc_config = PsiArcConfig::default();
    psiarc_config.path = config.dest_path.clone();
    let mut dest_psiarc = PsiArcBuilder::new(psiarc_config).build()?;
    
    // Read all frames
    let frames = reader.read_all_frames()?;
    info!("Found {} frames to relay", frames.len());
    
    // Process each frame
    for frame in frames {
        // Process the diff
        let processed_diff = process_diff(&frame.diff, config);
        
        // Write to destination
        dest_psiarc.write_diff(&processed_diff)?;
    }
    
    info!("PsiArc relay complete. Wrote {} frames", frames.len());
    Ok(())
}

/// Process a single diff according to relay configuration
fn process_diff(diff: &ConceptDiff, config: &RelayConfig) -> ConceptDiff {
    // Clone the diff to modify it
    let mut processed_diff = diff.clone();
    
    // Apply anonymization if configured
    if config.anonymize {
        anonymize_diff(&mut processed_diff, config.preserve_personas);
    }
    
    // Remove phase state if not preserving
    if !config.preserve_phase && processed_diff.metadata.contains_key("phase_state") {
        processed_diff.metadata.remove("phase_state");
    }
    
    // Add custom metadata
    for (key, value) in &config.custom_metadata {
        processed_diff.metadata.insert(key.clone(), value.clone());
    }
    
    // Add relay marker
    processed_diff.metadata.insert(
        "relayed".to_string(), 
        serde_json::Value::Bool(true)
    );
    
    processed_diff
}

/// Anonymize user information in a diff
fn anonymize_diff(diff: &mut ConceptDiff, preserve_personas: bool) {
    // Remove user_id
    diff.metadata.remove("user_id");
    
    // Remove session_id
    diff.metadata.remove("session_id");
    
    // Optionally remove persona_id
    if !preserve_personas {
        diff.metadata.remove("persona_id");
    }
    
    // Remove any other personally identifiable information
    diff.metadata.remove("email");
    diff.metadata.remove("name");
    diff.metadata.remove("avatar_url");
    
    // Add anonymization marker
    diff.metadata.insert(
        "anonymized".to_string(), 
        serde_json::Value::Bool(true)
    );
}

/// Relay CLI options
#[derive(Debug, Clone)]
pub struct RelayOptions {
    /// Source PsiArc file
    pub source: String,
    
    /// Destination PsiArc file
    pub destination: Option<String>,
    
    /// Whether to anonymize user attribution
    pub anonymize: bool,
    
    /// Whether to preserve persona information
    pub preserve_personas: bool,
    
    /// Whether to preserve phase states
    pub preserve_phase: bool,
}

/// Run the relay tool with CLI options
pub fn run_relay(options: RelayOptions) -> IoResult<()> {
    // Set up default destination if not provided
    let source_path = PathBuf::from(&options.source);
    let dest_path = match &options.destination {
        Some(dest) => PathBuf::from(dest),
        None => {
            let stem = source_path.file_stem().unwrap_or_default();
            let extension = source_path.extension().unwrap_or_default();
            let mut new_name = stem.to_os_string();
            new_name.push("_relayed.");
            new_name.push(extension);
            source_path.with_file_name(new_name)
        }
    };
    
    // Create relay configuration
    let config = RelayConfig {
        source_path,
        dest_path,
        anonymize: options.anonymize,
        preserve_personas: options.preserve_personas,
        preserve_phase: options.preserve_phase,
        custom_metadata: HashMap::new(),
    };
    
    // Run relay
    relay_psiarc(&config)
}

/// Command-line tool for PsiArc relay
#[cfg(feature = "cli")]
pub mod cli {
    use super::*;
    use clap::{Parser, ArgAction};
    
    #[derive(Parser, Debug)]
    #[clap(author, version, about = "Relay PsiArc files with privacy controls")]
    pub struct Cli {
        /// Source PsiArc file
        #[clap(value_name = "SOURCE")]
        pub source: String,
        
        /// Destination PsiArc file
        #[clap(value_name = "DESTINATION")]
        pub destination: Option<String>,
        
        /// Don't anonymize user information
        #[clap(short, long, action = ArgAction::SetFalse)]
        pub anonymize: bool,
        
        /// Don't preserve persona information
        #[clap(short = 'P', long, action = ArgAction::SetFalse)]
        pub preserve_personas: bool,
        
        /// Don't preserve phase state information
        #[clap(short = 'p', long, action = ArgAction::SetFalse)]
        pub preserve_phase: bool,
    }
    
    /// Run the CLI tool
    pub fn run() -> IoResult<()> {
        let cli = Cli::parse();
        
        let options = RelayOptions {
            source: cli.source,
            destination: cli.destination,
            anonymize: cli.anonymize,
            preserve_personas: cli.preserve_personas,
            preserve_phase: cli.preserve_phase,
        };
        
        run_relay(options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::{User, PersonaMode, Session, set_current_session};
    use tempfile::tempdir;
    
    #[test]
    fn test_relay_anonymization() -> IoResult<()> {
        // Create temporary directory
        let temp_dir = tempdir()?;
        let source_path = temp_dir.path().join("source.psiarc");
        let dest_path = temp_dir.path().join("dest.psiarc");
        
        // Create a test user and session
        let user = User::new(
            "github",
            "test_user",
            Some("test@example.com"),
            Some("Test User"),
            None,
        );
        
        let session = Session::new(user, PersonaMode::Glyphsmith);
        set_current_session(session);
        
        // Create source PsiArc with user context
        let mut source_config = PsiArcConfig::default();
        source_config.path = source_path.clone();
        let mut source_psiarc = PsiArcBuilder::new(source_config).build()?;
        
        // Add a diff with user context
        let diff = ConceptDiffBuilder::new(1)
            .create("test_concept", "TestConcept", Default::default())
            .with_user_context()
            .build();
            
        source_psiarc.write_diff(&diff)?;
        
        // Close source PsiArc
        drop(source_psiarc);
        
        // Create relay config with anonymization
        let relay_config = RelayConfig {
            source_path,
            dest_path: dest_path.clone(),
            anonymize: true,
            preserve_personas: true,
            preserve_phase: true,
            custom_metadata: HashMap::new(),
        };
        
        // Relay the PsiArc
        relay_psiarc(&relay_config)?;
        
        // Open destination PsiArc and check anonymization
        let reader = PsiArcReader::open(&dest_path)?;
        let frames = reader.read_all_frames()?;
        
        assert_eq!(frames.len(), 1);
        
        let relayed_diff = &frames[0].diff;
        
        // User ID should be removed
        assert!(!relayed_diff.metadata.contains_key("user_id"));
        
        // Session ID should be removed
        assert!(!relayed_diff.metadata.contains_key("session_id"));
        
        // Persona ID should be preserved
        assert!(relayed_diff.metadata.contains_key("persona_id"));
        
        // Anonymization marker should be added
        assert!(relayed_diff.metadata.contains_key("anonymized"));
        assert!(relayed_diff.metadata.get("anonymized").unwrap().as_bool().unwrap());
        
        // Relay marker should be added
        assert!(relayed_diff.metadata.contains_key("relayed"));
        assert!(relayed_diff.metadata.get("relayed").unwrap().as_bool().unwrap());
        
        Ok(())
    }
}
