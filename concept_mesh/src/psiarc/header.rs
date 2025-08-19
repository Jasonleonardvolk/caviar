//! PsiArc Header Metadata
//!
//! This module provides functionality for working with PsiArc header metadata,
//! which stores information about the archive's contents, creator, and other
//! metadata to enable better organization and searchability.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use tracing::{debug, error, info, warn};

/// PsiArc header metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PsiarcHeader {
    /// Archive version
    pub version: String,

    /// User who created the archive
    pub created_by: Option<String>,

    /// Persona used to create the archive
    pub persona: Option<String>,

    /// Source document or origin
    pub source: Option<String>,

    /// Source document ID
    pub document_id: Option<String>,

    /// Number of phase clusters
    pub phase_cluster_count: Option<u32>,

    /// Total number of ConceptDiffs
    pub total_concept_diffs: Option<u32>,

    /// Archive creation timestamp
    pub creation_timestamp: String,

    /// Archive title
    pub title: Option<String>,

    /// Archive description
    pub description: Option<String>,

    /// Archive tags
    pub tags: Vec<String>,

    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl Default for PsiarcHeader {
    fn default() -> Self {
        Self {
            version: "1.0".to_string(),
            created_by: None,
            persona: None,
            source: None,
            document_id: None,
            phase_cluster_count: None,
            total_concept_diffs: None,
            creation_timestamp: chrono::Utc::now().to_rfc3339(),
            title: None,
            description: None,
            tags: Vec::new(),
            metadata: HashMap::new(),
        }
    }
}

impl PsiarcHeader {
    /// Create a new PsiArc header
    pub fn new() -> Self {
        Self::default()
    }

    /// Set user information from current session
    pub fn with_current_session(mut self) -> Self {
        if let Some(session) = crate::auth::session::get_current_session() {
            self.created_by = Some(session.user.concept_id.clone());
            self.persona = Some(session.persona.mode.as_str().to_string());
        }
        self
    }

    /// Set source document
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Set document ID
    pub fn with_document_id(mut self, document_id: impl Into<String>) -> Self {
        self.document_id = Some(document_id.into());
        self
    }

    /// Set phase cluster count
    pub fn with_phase_cluster_count(mut self, count: u32) -> Self {
        self.phase_cluster_count = Some(count);
        self
    }

    /// Set total ConceptDiff count
    pub fn with_total_concept_diffs(mut self, count: u32) -> Self {
        self.total_concept_diffs = Some(count);
        self
    }

    /// Set title
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Set description
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Set tags
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Serialize) -> Self {
        if let Ok(json_value) = serde_json::to_value(value) {
            self.metadata.insert(key.into(), json_value);
        }
        self
    }

    /// Save header to a file
    pub fn save_to_file(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)
    }

    /// Load header from a file
    pub fn load_from_file(path: impl AsRef<Path>) -> io::Result<Self> {
        let json = fs::read_to_string(path)?;
        let header = serde_json::from_str(&json)?;
        Ok(header)
    }

    /// Generate a sidecar metadata file path for a PsiArc
    pub fn sidecar_path(psiarc_path: impl AsRef<Path>) -> PathBuf {
        let path = psiarc_path.as_ref();
        let stem = path.file_stem().unwrap_or_default();
        let mut new_name = stem.to_os_string();
        new_name.push(".meta.json");
        path.with_file_name(new_name)
    }

    /// Save as a sidecar metadata file for a PsiArc
    pub fn save_as_sidecar(&self, psiarc_path: impl AsRef<Path>) -> io::Result<PathBuf> {
        let sidecar_path = Self::sidecar_path(&psiarc_path);
        self.save_to_file(&sidecar_path)?;
        Ok(sidecar_path)
    }

    /// Load from a sidecar metadata file for a PsiArc
    pub fn load_from_sidecar(psiarc_path: impl AsRef<Path>) -> io::Result<Self> {
        let sidecar_path = Self::sidecar_path(&psiarc_path);
        Self::load_from_file(sidecar_path)
    }

    /// Create a header from a ConceptDiff with source information
    pub fn from_concept_diff(diff: &crate::diff::ConceptDiff) -> Self {
        let mut header = Self::new().with_current_session();

        // Extract metadata from the diff
        if let Some(source_title) = diff.metadata.get("source_title") {
            if let Some(title_str) = source_title.as_str() {
                header = header.with_title(title_str);
                header = header.with_source(title_str);
            }
        }

        if let Some(document_id) = diff.metadata.get("document_id") {
            if let Some(id_str) = document_id.as_str() {
                header = header.with_document_id(id_str);
            }
        }

        if let Some(tags) = diff.metadata.get("tags") {
            if let Some(tags_array) = tags.as_array() {
                let tag_strings: Vec<String> = tags_array
                    .iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect();

                if !tag_strings.is_empty() {
                    header = header.with_tags(tag_strings);
                }
            }
        }

        header
    }
}

/// List metadata for all PsiArc files in a directory
pub fn list_psiarc_metadata(dir: impl AsRef<Path>) -> io::Result<Vec<(PathBuf, PsiarcHeader)>> {
    let mut results = Vec::new();

    let dir_path = dir.as_ref();
    if !dir_path.exists() || !dir_path.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "Directory not found",
        ));
    }

    for entry in fs::read_dir(dir_path)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() {
            if let Some(ext) = path.extension() {
                if ext == "psiarc" {
                    // Check for sidecar metadata file
                    let sidecar_path = PsiarcHeader::sidecar_path(&path);

                    if sidecar_path.exists() {
                        // Load metadata from sidecar
                        if let Ok(header) = PsiarcHeader::load_from_file(&sidecar_path) {
                            results.push((path.clone(), header));
                        }
                    } else {
                        // Create default metadata
                        let header = PsiarcHeader::new().with_title(
                            path.file_stem()
                                .unwrap_or_default()
                                .to_string_lossy()
                                .to_string(),
                        );

                        results.push((path.clone(), header));
                    }
                }
            }
        }
    }

    Ok(results)
}

/// CLI command to list PsiArc files with metadata
pub fn psiarc_list_command(dir: impl AsRef<Path>, use_color: bool) -> io::Result<()> {
    // Enable or disable colors
    colored::control::set_override(!use_color);

    use colored::*;

    let archives = list_psiarc_metadata(dir)?;

    if archives.is_empty() {
        println!("No PsiArc files found.");
        return Ok(());
    }

    println!(
        "{} Found {} PsiArc archives",
        "🗃️".bright_magenta(),
        archives.len()
    );
    println!();

    for (path, header) in archives {
        let filename = path.file_name().unwrap_or_default().to_string_lossy();

        println!("{} {}", "📄".bright_blue(), filename.bright_white().bold());

        if let Some(title) = &header.title {
            println!("   Title: {}", title.bright_yellow());
        }

        if let Some(source) = &header.source {
            println!("   Source: {}", source.bright_green());
        }

        if let Some(created_by) = &header.created_by {
            println!("   Created by: {}", created_by.bright_blue());
        }

        if let Some(persona) = &header.persona {
            println!("   Persona: {}", persona.bright_cyan());
        }

        if let Some(diff_count) = header.total_concept_diffs {
            println!("   ConceptDiffs: {}", diff_count.to_string().bright_white());
        }

        if !header.tags.is_empty() {
            let tags_str = header
                .tags
                .iter()
                .map(|t| t.bright_green().to_string())
                .collect::<Vec<_>>()
                .join(", ");

            println!("   Tags: {}", tags_str);
        }

        println!();
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::{set_current_session, PersonaMode, Session, User};
    use tempfile::tempdir;

    #[test]
    fn test_psiarc_header() {
        // Create a test header
        let header = PsiarcHeader::new()
            .with_title("Test PsiArc")
            .with_source("test_document.pdf")
            .with_document_id("doc_123")
            .with_phase_cluster_count(3)
            .with_total_concept_diffs(42)
            .with_tags(vec!["test".to_string(), "document".to_string()])
            .with_metadata("custom_field", "custom_value");

        // Verify fields
        assert_eq!(header.title, Some("Test PsiArc".to_string()));
        assert_eq!(header.source, Some("test_document.pdf".to_string()));
        assert_eq!(header.document_id, Some("doc_123".to_string()));
        assert_eq!(header.phase_cluster_count, Some(3));
        assert_eq!(header.total_concept_diffs, Some(42));
        assert_eq!(header.tags, vec!["test", "document"]);
        assert!(header.metadata.contains_key("custom_field"));
    }

    #[test]
    fn test_save_load_header() -> io::Result<()> {
        // Create a temporary directory
        let temp_dir = tempdir()?;
        let header_path = temp_dir.path().join("test.meta.json");

        // Create a test header
        let header = PsiarcHeader::new()
            .with_title("Test PsiArc")
            .with_source("test_document.pdf");

        // Save to file
        header.save_to_file(&header_path)?;

        // Load from file
        let loaded_header = PsiarcHeader::load_from_file(&header_path)?;

        // Verify fields
        assert_eq!(loaded_header.title, Some("Test PsiArc".to_string()));
        assert_eq!(loaded_header.source, Some("test_document.pdf".to_string()));

        Ok(())
    }

    #[test]
    fn test_sidecar_path() {
        let psiarc_path = Path::new("test.psiarc");
        let sidecar_path = PsiarcHeader::sidecar_path(psiarc_path);

        assert_eq!(sidecar_path, Path::new("test.meta.json"));
    }

    #[test]
    fn test_with_current_session() {
        // Create a test user and session
        let user = User::new(
            "test",
            "test_user",
            Some("test@example.com"),
            Some("Test User"),
            None,
        );

        let session = Session::new(user, PersonaMode::Researcher);
        set_current_session(session);

        // Create header with current session
        let header = PsiarcHeader::new().with_current_session();

        // Verify fields
        assert_eq!(header.created_by, Some("USER_test_test_user".to_string()));
        assert_eq!(header.persona, Some("researcher".to_string()));
    }
}
