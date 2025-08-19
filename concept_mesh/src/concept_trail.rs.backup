//! Concept Trail
//!
//! This module provides functionality for tracking and navigating through
//! concept trails - time-ordered sequences of concepts created by a specific user,
//! from a specific source, or with specific attributes.

use crate::diff::ConceptDiff;
use crate::lcn::LargeConceptNetwork;
use crate::psiarc::{PsiArcReader, PsiarcHeader};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};
use tracing::{debug, info, warn, error};

/// A concept trail entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptTrailEntry {
    /// Concept ID
    pub concept_id: String,
    
    /// Concept type
    pub concept_type: Option<String>,
    
    /// Concept title
    pub title: Option<String>,
    
    /// User ID
    pub user_id: Option<String>,
    
    /// Persona ID
    pub persona_id: Option<String>,
    
    /// Source document ID
    pub document_id: Option<String>,
    
    /// Source title
    pub source_title: Option<String>,
    
    /// Source path
    pub source_path: Option<String>,
    
    /// Tags
    pub tags: Vec<String>,
    
    /// Created timestamp
    pub created_at: Option<DateTime<Utc>>,
    
    /// PsiArc ID
    pub psiarc_id: Option<String>,
    
    /// Frame ID in PsiArc
    pub frame_id: Option<u64>,
    
    /// Preview content
    pub preview: Option<String>,
}

impl ConceptTrailEntry {
    /// Create a new concept trail entry from a ConceptDiff
    pub fn from_diff(diff: &ConceptDiff) -> Option<Self> {
        // Extract concept ID and properties from the first creation operation
        let (concept_id, concept_type, properties) = diff.ops.iter()
            .find_map(|op| {
                if let crate::diff::Op::Create { id, concept_type, properties } = op {
                    Some((id.clone(), concept_type.clone(), properties.clone()))
                } else {
                    None
                }
            })?;
        
        // Extract metadata from diff
        let user_id = diff.metadata.get("user_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
            
        let persona_id = diff.metadata.get("persona_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
            
        let document_id = diff.metadata.get("document_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .or_else(|| {
                properties.get("document_id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
            });
            
        let source_title = diff.metadata.get("source_title")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .or_else(|| {
                properties.get("source_title")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
            });
            
        let source_path = diff.metadata.get("imported_from")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .or_else(|| {
                properties.get("source_path")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
            });
            
        // Extract tags
        let tags = diff.metadata.get("tags")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_else(|| {
                properties.get("tags")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(|s| s.to_string()))
                            .collect()
                    })
                    .unwrap_or_default()
            });
            
        // Extract created timestamp
        let created_at = diff.metadata.get("ingested_at")
            .and_then(|v| v.as_str())
            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&Utc));
            
        // Extract title
        let title = properties.get("title")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
            
        // Extract psiarc info from ingest origin
        let (psiarc_id, frame_id) = diff.metadata.get("concept_ingest_origin")
            .and_then(|v| v.as_object())
            .map(|obj| {
                let psiarc = obj.get("psiarc_id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                    
                let offset = obj.get("offset")
                    .and_then(|v| v.as_u64());
                    
                (psiarc, offset)
            })
            .unwrap_or((None, None));
            
        // Create preview from content or description
        let preview = properties.get("content")
            .and_then(|v| v.as_str())
            .map(|s| {
                if s.len() > 200 {
                    format!("{}...", &s[0..200])
                } else {
                    s.to_string()
                }
            })
            .or_else(|| {
                properties.get("description")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
            });
            
        Some(Self {
            concept_id,
            concept_type: Some(concept_type),
            title,
            user_id,
            persona_id,
            document_id,
            source_title,
            source_path,
            tags,
            created_at,
            psiarc_id,
            frame_id,
            preview,
        })
    }
}

/// A concept trail
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptTrail {
    /// Trail name
    pub name: String,
    
    /// Trail description
    pub description: Option<String>,
    
    /// Entries in the trail
    pub entries: Vec<ConceptTrailEntry>,
    
    /// Filter criteria
    pub filter: ConceptTrailFilter,
    
    /// Trail creation timestamp
    pub created_at: DateTime<Utc>,
}

/// Filter criteria for concept trails
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptTrailFilter {
    /// User ID to filter by
    pub user_id: Option<String>,
    
    /// Persona ID to filter by
    pub persona_id: Option<String>,
    
    /// Document ID to filter by
    pub document_id: Option<String>,
    
    /// Source title to filter by
    pub source_title: Option<String>,
    
    /// Tags to filter by (AND logic)
    pub tags: Vec<String>,
    
    /// Date range to filter by
    pub date_range: Option<(DateTime<Utc>, DateTime<Utc>)>,
    
    /// Custom filter function name
    pub custom_filter: Option<String>,
}

impl Default for ConceptTrailFilter {
    fn default() -> Self {
        Self {
            user_id: None,
            persona_id: None,
            document_id: None,
            source_title: None,
            tags: Vec::new(),
            date_range: None,
            custom_filter: None,
        }
    }
}

impl ConceptTrailFilter {
    /// Create a new filter for a specific user
    pub fn for_user(user_id: impl Into<String>) -> Self {
        Self {
            user_id: Some(user_id.into()),
            ..Default::default()
        }
    }
    
    /// Create a new filter for a specific persona
    pub fn for_persona(persona_id: impl Into<String>) -> Self {
        Self {
            persona_id: Some(persona_id.into()),
            ..Default::default()
        }
    }
    
    /// Create a new filter for a specific document
    pub fn for_document(document_id: impl Into<String>) -> Self {
        Self {
            document_id: Some(document_id.into()),
            ..Default::default()
        }
    }
    
    /// Create a new filter for a specific source
    pub fn for_source(source_title: impl Into<String>) -> Self {
        Self {
            source_title: Some(source_title.into()),
            ..Default::default()
        }
    }
    
    /// Add tags to the filter (AND logic)
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }
    
    /// Add a date range to the filter
    pub fn with_date_range(mut self, start: DateTime<Utc>, end: DateTime<Utc>) -> Self {
        self.date_range = Some((start, end));
        self
    }
    
    /// Check if an entry matches this filter
    pub fn matches(&self, entry: &ConceptTrailEntry) -> bool {
        // Check user ID
        if let Some(user_id) = &self.user_id {
            if entry.user_id.as_ref().map_or(true, |id| id != user_id) {
                return false;
            }
        }
        
        // Check persona ID
        if let Some(persona_id) = &self.persona_id {
            if entry.persona_id.as_ref().map_or(true, |id| id != persona_id) {
                return false;
            }
        }
        
        // Check document ID
        if let Some(document_id) = &self.document_id {
            if entry.document_id.as_ref().map_or(true, |id| id != document_id) {
                return false;
            }
        }
        
        // Check source title
        if let Some(source_title) = &self.source_title {
            if entry.source_title.as_ref().map_or(true, |title| title != source_title) {
                return false;
            }
        }
        
        // Check tags (AND logic)
        if !self.tags.is_empty() {
            let entry_tags: HashSet<_> = entry.tags.iter().collect();
            for tag in &self.tags {
                if !entry_tags.contains(tag) {
                    return false;
                }
            }
        }
        
        // Check date range
        if let Some((start, end)) = &self.date_range {
            if let Some(created_at) = &entry.created_at {
                if created_at < start || created_at > end {
                    return false;
                }
            } else {
                return false;
            }
        }
        
        true
    }
}

impl ConceptTrail {
    /// Create a new concept trail
    pub fn new(name: impl Into<String>, filter: ConceptTrailFilter) -> Self {
        Self {
            name: name.into(),
            description: None,
            entries: Vec::new(),
            filter,
            created_at: Utc::now(),
        }
    }
    
    /// Add an entry to the trail
    pub fn add_entry(&mut self, entry: ConceptTrailEntry) {
        // Only add if it matches the filter
        if self.filter.matches(&entry) {
            self.entries.push(entry);
        }
    }
    
    /// Sort entries by creation time
    pub fn sort_by_time(&mut self) {
        self.entries.sort_by(|a, b| {
            match (&a.created_at, &b.created_at) {
                (Some(a_time), Some(b_time)) => a_time.cmp(b_time),
                (Some(_), None) => std::cmp::Ordering::Less,
                (None, Some(_)) => std::cmp::Ordering::Greater,
                (None, None) => a.concept_id.cmp(&b.concept_id),
            }
        });
    }
    
    /// Get the most recent entries
    pub fn recent_entries(&self, limit: usize) -> Vec<&ConceptTrailEntry> {
        let mut entries: Vec<_> = self.entries.iter().collect();
        
        entries.sort_by(|a, b| {
            match (&b.created_at, &a.created_at) {
                (Some(b_time), Some(a_time)) => b_time.cmp(a_time),
                (Some(_), None) => std::cmp::Ordering::Less,
                (None, Some(_)) => std::cmp::Ordering::Greater,
                (None, None) => b.concept_id.cmp(&a.concept_id),
            }
        });
        
        entries.truncate(limit);
        entries
    }
    
    /// Filter entries by concept type
    pub fn filter_by_type(&self, concept_type: &str) -> Vec<&ConceptTrailEntry> {
        self.entries.iter()
            .filter(|entry| entry.concept_type.as_ref().map_or(false, |t| t == concept_type))
            .collect()
    }
    
    /// Set description
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
    
    /// Save the trail to a file
    pub fn save_to_file(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)
    }
    
    /// Load a trail from a file
    pub fn load_from_file(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let trail = serde_json::from_str(&json)?;
        Ok(trail)
    }
}

/// Create a concept trail from a PsiArc
pub fn create_trail_from_psiarc(
    psiarc_path: impl AsRef<Path>,
    name: Option<String>,
    filter: Option<ConceptTrailFilter>,
) -> std::io::Result<ConceptTrail> {
    let path = psiarc_path.as_ref();
    
    // Determine a name for the trail
    let trail_name = name.unwrap_or_else(|| {
        path.file_stem()
            .map(|stem| stem.to_string_lossy().to_string())
            .unwrap_or_else(|| "Unknown Trail".to_string())
    });
    
    // Create a filter if none provided
    let filter = filter.unwrap_or_default();
    
    // Create a new trail
    let mut trail = ConceptTrail::new(trail_name, filter);
    
    // Load the PsiArc header for description
    if let Ok(header) = crate::psiarc::header::PsiarcHeader::load_from_sidecar(path) {
        if let Some(title) = header.title {
            trail.description = Some(format!("Trail from PsiArc: {}", title));
        }
    }
    
    // Open the PsiArc
    let reader = PsiArcReader::open(path)?;
    
    // Read all frames
    let frames = reader.read_all_frames()?;
    
    // Process each frame
    for frame in frames {
        if let Some(entry) = ConceptTrailEntry::from_diff(&frame.diff) {
            trail.add_entry(entry);
        }
    }
    
    // Sort entries by time
    trail.sort_by_time();
    
    Ok(trail)
}

/// Create a trail from multiple PsiArcs
pub fn create_trail_from_multiple_psiarcs(
    psiarc_paths: &[PathBuf],
    name: impl Into<String>,
    filter: Option<ConceptTrailFilter>,
) -> std::io::Result<ConceptTrail> {
    let trail_name = name.into();
    
    // Create a filter if none provided
    let filter = filter.unwrap_or_default();
    
    // Create a new trail
    let mut trail = ConceptTrail::new(trail_name, filter);
    
    // Process each PsiArc
    for path in psiarc_paths {
        // Skip if file doesn't exist
        if !path.exists() {
            warn!("PsiArc file not found: {}", path.display());
            continue;
        }
        
        // Open the PsiArc
        match PsiArcReader::open(path) {
            Ok(reader) => {
                // Read all frames
                match reader.read_all_frames() {
                    Ok(frames) => {
                        // Process each frame
                        for frame in frames {
                            if let Some(entry) = ConceptTrailEntry::from_diff(&frame.diff) {
                                trail.add_entry(entry);
                            }
                        }
                    },
                    Err(e) => {
                        warn!("Failed to read frames from {}: {}", path.display(), e);
                    }
                }
            },
            Err(e) => {
                warn!("Failed to open PsiArc {}: {}", path.display(), e);
            }
        }
    }
    
    // Sort entries by time
    trail.sort_by_time();
    
    Ok(trail)
}

/// Create a concept trail from a directory of PsiArcs
pub fn create_trail_from_directory(
    directory: impl AsRef<Path>,
    name: impl Into<String>,
    filter: Option<ConceptTrailFilter>,
    recursive: bool,
) -> std::io::Result<ConceptTrail> {
    let dir_path = directory.as_ref();
    
    // Collect all PsiArc files
    let mut psiarc_paths = Vec::new();
    
    if recursive {
        // Recursive walk
        for entry in walkdir::WalkDir::new(dir_path) {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };
            
            let path = entry.path();
            
            if path.is_file() && path.extension().map_or(false, |ext| ext == "psiarc") {
                psiarc_paths.push(path.to_path_buf());
            }
        }
    } else {
        // Non-recursive
        if let Ok(entries) = std::fs::read_dir(dir_path) {
            for entry in entries.flatten() {
                let path = entry.path();
                
                if path.is_file() && path.extension().map_or(false, |ext| ext == "psiarc") {
                    psiarc_paths.push(path);
                }
            }
        }
    }
    
    // Create trail from collected files
    create_trail_from_multiple_psiarcs(&psiarc_paths, name, filter)
}

/// Concept trail manager
pub struct ConceptTrailManager {
    /// Base directory for trails
    base_dir: PathBuf,
    
    /// Loaded trails
    trails: HashMap<String, ConceptTrail>,
}

impl ConceptTrailManager {
    /// Create a new concept trail manager
    pub fn new(base_dir: impl AsRef<Path>) -> Self {
        let base_dir = base_dir.as_ref().to_path_buf();
        
        // Ensure directory exists
        if !base_dir.exists() {
            std::fs::create_dir_all(&base_dir).ok();
        }
        
        Self {
            base_dir,
            trails: HashMap::new(),
        }
    }
    
    /// Initialize the manager
    pub fn initialize(&mut self) -> std::io::Result<()> {
        // Load all trails
        self.load_all_trails()?;
        
        Ok(())
    }
    
    /// Load all trails from the base directory
    pub fn load_all_trails(&mut self) -> std::io::Result<()> {
        // Clear existing trails
        self.trails.clear();
        
        // Read all files in the base directory
        if let Ok(entries) = std::fs::read_dir(&self.base_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                
                if path.is_file() && path.extension().map_or(false, |ext| ext == "trail") {
                    if let Ok(trail) = ConceptTrail::load_from_file(&path) {
                        self.trails.insert(trail.name.clone(), trail);
                    }
                }
            }
        }
        
        info!("Loaded {} concept trails", self.trails.len());
        
        Ok(())
    }
    
    /// Save a trail
    pub fn save_trail(&mut self, trail: &ConceptTrail) -> std::io::Result<PathBuf> {
        let file_name = format!("{}.trail", trail.name.replace(" ", "_").to_lowercase());
        let path = self.base_dir.join(file_name);
        
        trail.save_to_file(&path)?;
        self.trails.insert(trail.name.clone(), trail.clone());
        
        Ok(path)
    }
    
    /// Get a trail by name
    pub fn get_trail(&self, name: &str) -> Option<&ConceptTrail> {
        self.trails.get(name)
    }
    
    /// List all trails
    pub fn list_trails(&self) -> Vec<&ConceptTrail> {
        self.trails.values().collect()
    }
    
    /// Create a new trail
    pub fn create_trail(&mut self, trail: ConceptTrail) -> std::io::Result<PathBuf> {
        self.save_trail(&trail)
    }
    
    /// Delete a trail
    pub fn delete_trail(&mut self, name: &str) -> std::io::Result<()> {
        if let Some(trail) = self.trails.remove(name) {
            let file_name = format!("{}.trail", trail.name.replace(" ", "_").to_lowercase());
            let path = self.base_dir.join(file_name);
            
            if path.exists() {
                std::fs::remove_file(path)?;
            }
        }
        
        Ok(())
    }
}

/// CLI command to create a trail from a source
pub fn create_trail_command(
    source: impl AsRef<Path>, 
    name: Option<String>,
    output: Option<impl AsRef<Path>>,
    filter_user: Option<String>,
    filter_persona: Option<String>,
    filter_tags: Vec<String>,
) -> std::io::Result<PathBuf> {
    let source_path = source.as_ref();
    
    // Create filter
    let mut filter = ConceptTrailFilter::default();
    
    if let Some(user) = filter_user {
        filter.user_id = Some(user);
    }
    
    if let Some(persona) = filter_persona {
        filter.persona_id = Some(persona);
    }
    
    if !filter_tags.is_empty() {
        filter.tags = filter_tags;
    }
    
    // Create trail
    let trail = if source_path.is_dir() {
        create_trail_from_directory(source_path, name.clone().unwrap_or_else(|| "Directory Trail".to_string()), Some(filter), true)?
    } else {
        create_trail_from_psiarc(source_path, name.clone(), Some(filter))?
    };
    
    // Determine output path
    let output_path = if let Some(output_path) = output {
        output_path.as_ref().to_path_buf()
    } else {
        let file_name = format!("{}.trail", trail.name.replace(" ", "_").to_lowercase());
        std::env::current_dir()?.join(file_name)
    };
    
    // Save trail
    trail.save_to_file(&output_path)?;
    
    // Print summary
    println!("Created trail '{}' with {} entries", trail.name, trail.entries.len());
    println!("Saved to {}", output_path.display());
    
    Ok(output_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::{User, PersonaMode, Session, set_current_session};
    use tempfile::tempdir;
    
    #[test]
    fn test_concept_trail_filter() {
        // Create an entry
        let entry = ConceptTrailEntry {
            concept_id: "test_concept".to_string(),
            concept_type: Some("Document".to_string()),
            title: Some("Test Document".to_string()),
            user_id: Some("user1".to_string()),
            persona_id: Some("researcher".to_string()),
            document_id: Some("doc1".to_string()),
            source_title: Some("Source Document".to_string()),
            source_path: Some("documents/test.pdf".to_string()),
            tags: vec!["test".to_string(), "document".to_string()],
            created_at: Some(Utc::now()),
            psiarc_id: None,
            frame_id: None,
            preview: None,
        };
        
        // Test user filter
        let user_filter = ConceptTrailFilter::for_user("user1");
        assert!(user_filter.matches(&entry));
        
        let wrong_user_filter = ConceptTrailFilter::for_user("user2");
        assert!(!wrong_user_filter.matches(&entry));
        
        // Test persona filter
        let persona_filter = ConceptTrailFilter::for_persona("researcher");
        assert!(persona_filter.matches(&entry));
        
        let wrong_persona_filter = ConceptTrailFilter::for_persona("glyphsmith");
        assert!(!wrong_persona_filter.matches(&entry));
        
        // Test document filter
        let document_filter = ConceptTrailFilter::for_document("doc1");
        assert!(document_filter.matches(&entry));
        
        // Test source filter
        let source_filter = ConceptTrailFilter::for_source("Source Document");
        assert!(source_filter.matches(&entry));
        
        // Test tag filter
        let tag_filter = ConceptTrailFilter::default().with_tags(vec!["test".to_string()]);
        assert!(tag_filter.matches(&entry));
        
        let multi_tag_filter = ConceptTrailFilter::default().with_tags(vec!["test".to_string(), "document".to_string()]);
        assert!(multi_tag_filter.matches(&entry));
        
        let wrong_tag_filter = ConceptTrailFilter::default().with_tags(vec!["wrong_tag".to_string()]);
        assert!(!wrong_tag_filter.matches(&entry));
        
        // Test combined filter
        let combined_filter = ConceptTrailFilter {
            user_id: Some("user1".to_string()),
            persona_id: Some("researcher".to_string()),
            document_id: None,
            source_title: None,
            tags: vec!["test".to_string()],
            date_range: None,
            custom_filter: None,
        };
        
        assert!(combined_filter.matches(&entry));
        
        // Test incompatible combined filter
        let incompatible_filter = ConceptTrailFilter {
            user_id: Some("user1".to_string()),
            persona_id: Some("glyphsmith".to_string()), // Different persona
            document_id: None,
            source_title: None,
            tags: vec![],
            date_range: None,
            custom_filter: None,
        };
        
        assert!(!incompatible_filter.matches(&entry));
    }
    
    #[test]
    fn test_concept_trail() -> std::io::Result<()> {
        // Create entries
        let entry1 = ConceptTrailEntry {
            concept_id: "concept1".to_string(),
            concept_type: Some("Document".to_string()),
            title: Some("Document 1".to_string()),
            user_id: Some("user1".to_string()),
            persona_id: Some("researcher".to_string()),
            document_id: Some("doc1".to_string()),
            source_title: Some("Source 1".to_string()),
            source_path: Some("documents/doc1.pdf".to_string()),
            tags: vec!["test".to_string(), "document".to_string()],
            created_at: Some(Utc::now() - chrono::Duration::days(1)),
            psiarc_id: None,
            frame_id: None,
            preview: None,
        };
        
        let entry2 = ConceptTrailEntry {
            concept_id: "concept2".to_string(),
            concept_type: Some("Section".to_string()),
            title: Some("Section 1".to_string()),
            user_id: Some("user1".to_string()),
            persona_id: Some("researcher".to_string()),
            document_id: Some("doc1".to_string()),
            source_title: Some("Source 1".to_string()),
            source_path: Some("documents/doc1.pdf".to_string()),
            tags: vec!["test".to_string(), "section".to_string()],
            created_at: Some(Utc::now()),
            psiarc_id: None,
            frame_id: None,
            preview: None,
        };
        
        // Create a trail
        let mut trail = ConceptTrail::new("Test Trail", ConceptTrailFilter::default());
        
        // Add entries
        trail.add_entry(entry1.clone());
        trail.add_entry(entry2.clone());
        
        // Sort by time
        trail.sort_by_time();
        
        // Verify entries are sorted
        assert_eq!(trail.entries.len(), 2);
        assert_eq!(trail.entries[0].concept_id, "concept1");
        assert_eq!(trail.entries[1].concept_id, "concept2");
        
        // Test recent entries
        let recent = trail.recent_entries(1);
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0].concept_id, "concept2");
        
        // Test filter by type
        let documents = trail.filter_by_type("Document");
        assert_eq!(documents.len(), 1);
        assert_eq!(documents[0].concept_id, "concept1");
        
        let sections = trail.filter_by_type("Section");
        assert_eq!(sections.len(), 1);
        assert_eq!(sections[0].concept_id, "concept2");
        
        // Create a temporary directory
        let
