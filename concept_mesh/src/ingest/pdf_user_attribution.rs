//! PDF Ingest User Attribution
//!
//! This module extends the PDF ingest system to include user and persona attribution,
//! ensuring that all ingested content is properly linked to the user who uploaded it
//! and the persona that was active during ingestion.
//! 
//! It also adds rich metadata about the source document, including titles, tags, 
//! file origins, and document identifiers for complete provenance tracking.

use crate::auth::session::{get_current_session, Session};
use crate::diff::{ConceptDiff, ConceptDiffBuilder, UserContextExt};
use crate::ingest::pdf::PdfIngestOptions;
use serde_json::json;
use std::path::Path;
use std::path::PathBuf;
use std::io::Read;
use sha2::{Sha256, Digest};
use tracing::{info, warn, debug};

/// Ingest source information structure
#[derive(Debug, Clone)]
pub struct IngestSourceInfo {
    /// Source title (e.g., document title)
    pub source_title: String,
    
    /// Source path (where the document was imported from)
    pub source_path: PathBuf,
    
    /// Document identifier (SHA-256 hash of file path)
    pub document_id: String,
    
    /// Tags for the document
    pub tags: Vec<String>,
    
    /// PsiArc identifier (if known)
    pub psiarc_id: Option<String>,
    
    /// Offset within PsiArc (if known)
    pub offset: Option<u64>,
}

impl IngestSourceInfo {
    /// Create new source info from a file path and title
    pub fn new(path: impl AsRef<Path>, title: impl Into<String>) -> Self {
        let path_buf = path.as_ref().to_path_buf();
        let title_str = title.into();
        
        Self {
            source_title: title_str,
            source_path: path_buf.clone(),
            document_id: compute_document_id(&path_buf),
            tags: Vec::new(),
            psiarc_id: None,
            offset: None,
        }
    }
    
    /// Add tags to the source info
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }
    
    /// Set PsiArc origin information
    pub fn with_psiarc_origin(mut self, psiarc_id: impl Into<String>, offset: u64) -> Self {
        self.psiarc_id = Some(psiarc_id.into());
        self.offset = Some(offset);
        self
    }
    
    /// Convert to metadata map for ConceptDiff
    pub fn to_metadata(&self) -> serde_json::Map<String, serde_json::Value> {
        let mut metadata = serde_json::Map::new();
        
        metadata.insert("source_title".to_string(), json!(self.source_title));
        metadata.insert("imported_from".to_string(), json!(self.source_path.to_string_lossy().to_string()));
        metadata.insert("document_id".to_string(), json!(self.document_id));
        
        if !self.tags.is_empty() {
            metadata.insert("tags".to_string(), json!(self.tags));
        }
        
        if let (Some(psiarc_id), Some(offset)) = (&self.psiarc_id, self.offset) {
            metadata.insert("concept_ingest_origin".to_string(), json!({
                "psiarc_id": psiarc_id,
                "offset": offset
            }));
        }
        
        metadata
    }
}

/// Compute document ID from file path (SHA-256 hash)
fn compute_document_id(path: &Path) -> String {
    // Try to read the file to hash its contents
    if path.exists() {
        if let Ok(mut file) = std::fs::File::open(path) {
            let mut hasher = Sha256::new();
            let mut buffer = Vec::new();
            
            // Read file content
            if file.read_to_end(&mut buffer).is_ok() {
                // Hash the content
                hasher.update(&buffer);
                let result = hasher.finalize();
                return format!("{:x}", result);
            }
        }
    }
    
    // Fallback to hashing the path string if file can't be read
    let mut hasher = Sha256::new();
    hasher.update(path.to_string_lossy().as_bytes());
    let result = hasher.finalize();
    format!("{:x}", result)
}

/// Extension trait for PDF ingest
pub trait UserAttributionExt {
    /// Add user attribution to PDF ingest options
    fn with_user_attribution(self) -> Self;
    
    /// Add source information to PDF ingest options
    fn with_source_info(self, source_info: &IngestSourceInfo) -> Self;
}

impl UserAttributionExt for PdfIngestOptions {
    /// Add user attribution to PDF ingest options if a session is active
    fn with_user_attribution(mut self) -> Self {
        if let Some(session) = get_current_session() {
            // Add user and persona IDs to metadata
            self.metadata.insert("user_id".to_string(), session.user.concept_id.clone());
            self.metadata.insert("persona_id".to_string(), session.persona.mode.as_str().to_string());
            self.metadata.insert("session_id".to_string(), session.id.clone());
            
            // Add ingestion timestamp
            self.metadata.insert("ingested_at".to_string(), chrono::Utc::now().to_rfc3339());
            
            info!("PDF ingest attributed to user {} ({})", 
                 session.user.concept_id, 
                 session.persona.mode.as_str());
        } else {
            warn!("No active session, PDF ingest will not have user attribution");
        }
        
        self
    }
    
    /// Add source information to PDF ingest options
    fn with_source_info(mut self, source_info: &IngestSourceInfo) -> Self {
        // Add source metadata
        let source_metadata = source_info.to_metadata();
        
        for (key, value) in source_metadata {
            self.metadata.insert(key, value);
        }
        
        // If title is not set in options, use source title
        if self.title.is_none() {
            self.title = Some(source_info.source_title.clone());
        }
        
        debug!("Added source information to PDF ingest options: title={}, path={}", 
              source_info.source_title, 
              source_info.source_path.display());
              
        self
    }
}

/// Create a document concept diff with user attribution and source information
pub fn create_document_diff(
    concept_id: &str,
    title: &str,
    source_path: &Path,
    metadata: serde_json::Map<String, serde_json::Value>,
) -> ConceptDiff {
    let mut builder = ConceptDiffBuilder::new(0);
    
    // Create document node
    let mut attributes = serde_json::Map::new();
    attributes.insert("title".to_string(), json!(title));
    attributes.insert("source_path".to_string(), json!(source_path.to_string_lossy()));
    
    // Add all metadata
    for (key, value) in metadata {
        attributes.insert(key, value);
    }
    
    // Create document concept with user context
    builder = builder.create(concept_id, "Document", attributes);
    
    // Add user context if available
    builder.with_user_context().build()
}

/// Create a document concept diff with enhanced attribution
pub fn create_attributed_document(
    concept_id: &str,
    source_info: &IngestSourceInfo,
    additional_metadata: Option<serde_json::Map<String, serde_json::Value>>,
) -> ConceptDiff {
    let mut builder = ConceptDiffBuilder::new(0);
    
    // Create document node
    let mut attributes = serde_json::Map::new();
    attributes.insert("title".to_string(), json!(source_info.source_title.clone()));
    attributes.insert("source_path".to_string(), json!(source_info.source_path.to_string_lossy().to_string()));
    
    // Add source metadata
    let source_metadata = source_info.to_metadata();
    for (key, value) in source_metadata {
        attributes.insert(key, value);
    }
    
    // Add additional metadata if provided
    if let Some(metadata) = additional_metadata {
        for (key, value) in metadata {
            attributes.insert(key, value);
        }
    }
    
    // Create document concept with user context
    builder = builder.create(concept_id, "Document", attributes);
    
    // Add user context if available
    builder.with_user_context().build()
}

/// Register a document section with user attribution and source information
pub fn register_document_section(
    document_id: &str,
    section_id: &str,
    section_type: &str,
    content: &str,
    metadata: serde_json::Map<String, serde_json::Value>,
) -> ConceptDiff {
    let mut builder = ConceptDiffBuilder::new(0);
    
    // Create section node
    let mut attributes = serde_json::Map::new();
    attributes.insert("content".to_string(), json!(content));
    
    // Add all metadata
    for (key, value) in metadata {
        attributes.insert(key, value);
    }
    
    // Create section concept
    builder = builder.create(section_id, section_type, attributes);
    
    // Link to document
    builder = builder.bind(document_id, section_id, None);
    
    // Add user context if available
    builder.with_user_context().build()
}

/// Register a document section with source information
pub fn register_attributed_section(
    document_id: &str,
    section_id: &str,
    section_type: &str,
    content: &str,
    source_info: &IngestSourceInfo,
    additional_metadata: Option<serde_json::Map<String, serde_json::Value>>,
) -> ConceptDiff {
    let mut builder = ConceptDiffBuilder::new(0);
    
    // Create section node
    let mut attributes = serde_json::Map::new();
    attributes.insert("content".to_string(), json!(content));
    
    // Add document reference
    attributes.insert("document_id".to_string(), json!(document_id));
    attributes.insert("source_title".to_string(), json!(source_info.source_title.clone()));
    
    // Add source metadata
    let source_metadata = source_info.to_metadata();
    for (key, value) in &source_metadata {
        // Skip document_id since we already added it
        if key != "document_id" {
            attributes.insert(key.clone(), value.clone());
        }
    }
    
    // Add additional metadata if provided
    if let Some(metadata) = additional_metadata {
        for (key, value) in metadata {
            attributes.insert(key, value);
        }
    }
    
    // Create section concept
    builder = builder.create(section_id, section_type, attributes);
    
    // Link to document
    builder = builder.bind(document_id, section_id, None);
    
    // Add user context if available
    builder.with_user_context().build()
}

/// Integration function to enable user attribution in the PDF ingest pipeline
pub fn enable_user_attribution() {
    info!("Enabling user attribution for PDF ingest");
    // This function would be called during system initialization
    // In a real implementation, it might register hooks or modify the ingest pipeline
}

/// Extract tags from document content
pub fn extract_tags_from_content(content: &str) -> Vec<String> {
    // This is a placeholder for a more sophisticated tag extraction algorithm
    // In a real implementation, this would use NLP techniques to extract meaningful tags
    
    // For now, just extract some common keywords
    let common_keywords = [
        "research", "science", "technology", "engineering", "mathematics",
        "philosophy", "psychology", "biology", "physics", "chemistry",
        "computer", "software", "hardware", "network", "internet",
        "data", "information", "knowledge", "learning", "education",
        "business", "economics", "finance", "marketing", "management",
        "health", "medicine", "disease", "treatment", "prevention",
        "art", "music", "literature", "culture", "history",
        "politics", "government", "policy", "law", "regulation",
        "environment", "climate", "energy", "sustainability", "conservation",
        "society", "community", "family", "religion", "ethics",
        "systems", "complexity", "chaos", "emergence", "adaptation",
        "feedback", "recursion", "iteration", "simulation", "modeling"
    ];
    
    // Convert content to lowercase for case-insensitive matching
    let content_lower = content.to_lowercase();
    
    // Extract tags that appear in the content
    let mut tags = Vec::new();
    
    for keyword in common_keywords.iter() {
        if content_lower.contains(keyword) {
            tags.push(keyword.to_string());
        }
    }
    
    // Limit to top 10 tags
    if tags.len() > 10 {
        tags.truncate(10);
    }
    
    tags
}

/// Auto-detect title from PDF content
pub fn auto_detect_title(content: &str) -> Option<String> {
    // This is a placeholder for a more sophisticated title detection algorithm
    // In a real implementation, this would use NLP techniques to extract the document title
    
    // For now, just use the first non-empty line
    content.lines()
        .map(|line| line.trim())
        .filter(|line| !line.is_empty())
        .next()
        .map(|line| {
            // Limit title length
            if line.len() > 100 {
                line[0..100].to_string()
            } else {
                line.to_string()
            }
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::{User, PersonaMode, Session, set_current_session};
    use std::path::PathBuf;
    
    #[test]
    fn test_pdf_ingest_with_attribution() {
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
        
        // Create ingest options with attribution
        let mut options = PdfIngestOptions {
            title: Some("Test Document".to_string()),
            metadata: serde_json::Map::new(),
        };
        
        options = options.with_user_attribution();
        
        // Verify attribution
        assert!(options.metadata.contains_key("user_id"));
        assert!(options.metadata.contains_key("persona_id"));
        assert!(options.metadata.contains_key("session_id"));
        assert!(options.metadata.contains_key("ingested_at"));
        
        // Create document diff
        let doc_id = "doc_123";
        let source_path = PathBuf::from("test.pdf");
        let diff = create_document_diff(doc_id, "Test Document", &source_path, options.metadata);
        
        // Verify diff has user context
        assert!(diff.metadata.contains_key("user_id"));
        assert!(diff.metadata.contains_key("persona_id"));
        assert!(diff.metadata.contains_key("session_id"));
    }
    
    #[test]
    fn test_source_info() {
        // Create source info
        let path = PathBuf::from("test_document.pdf");
        let source_info = IngestSourceInfo::new(&path, "Test Document")
            .with_tags(vec!["research".to_string(), "systems".to_string()])
            .with_psiarc_origin("psiarc_123", 42);
        
        // Verify source info
        assert_eq!(source_info.source_title, "Test Document");
        assert_eq!(source_info.source_path, path);
        assert!(!source_info.document_id.is_empty());
        assert_eq!(source_info.tags, vec!["research", "systems"]);
        assert_eq!(source_info.psiarc_id, Some("psiarc_123".to_string()));
        assert_eq!(source_info.offset, Some(42));
        
        // Verify metadata conversion
        let metadata = source_info.to_metadata();
        assert_eq!(metadata.get("source_title").unwrap().as_str().unwrap(), "Test Document");
        assert_eq!(metadata.get("imported_from").unwrap().as_str().unwrap(), "test_document.pdf");
        assert!(metadata.get("document_id").is_some());
        assert!(metadata.get("tags").is_some());
        assert!(metadata.get("concept_ingest_origin").is_some());
    }
    
    #[test]
    fn test_pdf_ingest_with_source_info() {
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
        
        // Create source info
        let path = PathBuf::from("research_paper.pdf");
        let source_info = IngestSourceInfo::new(&path, "Research Paper")
            .with_tags(vec!["research".to_string(), "systems".to_string()]);
        
        // Create ingest options with attribution and source info
        let options = PdfIngestOptions {
            title: None, // Let it use source title
            metadata: serde_json::Map::new(),
        }
        .with_user_attribution()
        .with_source_info(&source_info);
        
        // Verify options
        assert_eq!(options.title, Some("Research Paper".to_string()));
        assert!(options.metadata.contains_key("user_id"));
        assert!(options.metadata.contains_key("source_title"));
        assert!(options.metadata.contains_key("imported_from"));
        assert!(options.metadata.contains_key("document_id"));
        assert!(options.metadata.contains_key("tags"));
        
        // Create document with attributed source
        let doc_id = "doc_456";
        let diff = create_attributed_document(doc_id, &source_info, None);
        
        // Verify document attributes
        if let Some(create_op) = diff.ops.first() {
            if let crate::diff::Op::Create { properties, .. } = create_op {
                assert_eq!(properties.get("title").unwrap().as_str().unwrap(), "Research Paper");
                assert_eq!(properties.get("source_title").unwrap().as_str().unwrap(), "Research Paper");
                assert!(properties.get("document_id").is_some());
                assert!(properties.get("tags").is_some());
            } else {
                panic!("Expected Create operation");
            }
        } else {
            panic!("No operations in diff");
        }
    }
    
    #[test]
    fn test_extract_tags() {
        let content = "This document discusses complex systems theory and the role of feedback loops in creating emergent behavior. The chaos that arises from simple rules is a key aspect of systems thinking.";
        
        let tags = extract_tags_from_content(content);
        
        assert!(tags.contains(&"systems".to_string()));
        assert!(tags.contains(&"feedback".to_string()));
        assert!(tags.contains(&"chaos".to_string()));
        assert!(tags.contains(&"complexity".to_string()));
        assert!(tags.contains(&"emergence".to_string()));
    }
}
