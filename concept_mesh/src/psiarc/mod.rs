//! PsiArc Module
//!
use std::path::PathBuf;

// Export the header module
pub mod header;

// Import public types
pub use self::header::PsiarcHeader;

/// Options for creating a PsiArc
#[derive(Debug, Clone)]
pub struct PsiarcOptions {
    /// Path to the PsiArc file
    pub path: std::path::PathBuf,

    /// Whether to create a metadata sidecar file
    pub create_sidecar: bool,

    /// Whether to compress the PsiArc
    pub compress: bool,
}

impl Default for PsiarcOptions {
    fn default() -> Self {
        Self {
            path: std::path::PathBuf::new(),
            create_sidecar: true,
            compress: true,
        }
    }
}

/// A PsiArc is a file that stores ConceptDiffs for later replay.
pub struct PsiArc {
    /// Path to the PsiArc file
    pub path: std::path::PathBuf,

    /// Header metadata
    pub header: PsiarcHeader,

    /// Options
    pub options: PsiarcOptions,
}

/// Builder for creating a PsiArc
pub struct PsiarcBuilder {
    /// Options for the PsiArc
    pub options: PsiarcOptions,

    /// Header metadata
    pub header: PsiarcHeader,
}

impl PsiarcBuilder {
    /// Create a new PsiarcBuilder with the given options
    pub fn new(options: PsiarcOptions) -> Self {
        Self {
            options,
            header: PsiarcHeader::new().with_current_session(),
        }
    }

    /// Set the title of the PsiArc
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.header = self.header.with_title(title);
        self
    }

    /// Set the source of the PsiArc
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.header = self.header.with_source(source);
        self
    }

    /// Set the document ID of the PsiArc
    pub fn with_document_id(mut self, document_id: impl Into<String>) -> Self {
        self.header = self.header.with_document_id(document_id);
        self
    }

    /// Set the tags of the PsiArc
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.header = self.header.with_tags(tags);
        self
    }

    /// Build the PsiArc
    pub fn build(self) -> std::io::Result<PsiArc> {
        // Create a new PsiArc
        let psiarc = PsiArc {
            path: self.options.path.clone(),
            header: self.header,
            options: self.options,
        };

        // Save the header as a sidecar if enabled
        if psiarc.options.create_sidecar {
            psiarc.header.save_as_sidecar(&psiarc.path)?;
        }

        Ok(psiarc)
    }
}

impl PsiArc {
    /// Create a new PsiArc
    pub fn new(path: impl Into<std::path::PathBuf>) -> Self {
        Self {
            path: path.into(),
            header: PsiarcHeader::new().with_current_session(),
            options: PsiarcOptions::default(),
        }
    }

    /// Write a ConceptDiff to the PsiArc
    pub fn write_diff(&mut self, diff: &crate::diff::ConceptDiff) -> std::io::Result<()> {
        // This is a placeholder implementation
        // In a real implementation, this would write the diff to the file

        // Update the header with information from the diff
        if self.header.title.is_none() {
            if let Some(source_title) = diff.metadata.get("source_title") {
                if let Some(title_str) = source_title.as_str() {
                    self.header = self.header.with_title(title_str);
                }
            }
        }

        // Update the diff count
        let current_count = self.header.total_concept_diffs.unwrap_or(0);
        self.header = self.header.with_total_concept_diffs(current_count + 1);

        // Update the sidecar if enabled
        if self.options.create_sidecar {
            self.header.save_as_sidecar(&self.path)?;
        }

        Ok(())
    }
}

/// PsiArc frame containing a ConceptDiff
#[derive(Debug, Clone)]
pub struct PsiArcFrame {
    /// Frame ID
    pub id: u64,

    /// ConceptDiff
    pub diff: crate::diff::ConceptDiff,

    /// Timestamp
    pub timestamp: std::time::SystemTime,
}

/// PsiArc reader for reading ConceptDiffs from a PsiArc
pub struct PsiArcReader {
    /// Path to the PsiArc file
    pub path: std::path::PathBuf,

    /// Header metadata
    pub header: PsiarcHeader,
}

impl PsiArcReader {
    /// Open a PsiArc file for reading
    pub fn open(path: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        let path_buf = path.as_ref().to_path_buf();

        // Try to load the header from the sidecar
        let header = match PsiarcHeader::load_from_sidecar(&path_buf) {
            Ok(h) => h,
            Err(_) => {
                // If no sidecar exists, create a default header
                PsiarcHeader::new().with_title(
                    path_buf
                        .file_stem()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .to_string(),
                )
            }
        };

        Ok(Self {
            path: path_buf,
            header,
        })
    }

    /// Read all frames from the PsiArc
    pub fn read_all_frames(&self) -> std::io::Result<Vec<PsiArcFrame>> {
        // This is a placeholder implementation
        // In a real implementation, this would read all frames from the file

        Ok(Vec::new())
    }

    /// Read a specific frame from the PsiArc
    pub fn read_frame(&self, frame_id: u64) -> std::io::Result<PsiArcFrame> {
        // This is a placeholder implementation
        // In a real implementation, this would read a specific frame from the file

        Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("Frame {} not found", frame_id),
        ))
    }
}

/// PsiArc configuration
#[derive(Debug, Clone)]
pub struct PsiArcConfig {
    /// Path to the PsiArc file
    pub path: std::path::PathBuf,
}

impl Default for PsiArcConfig {
    fn default() -> Self {
        Self {
            path: std::path::PathBuf::new(),
        }
    }
}

/// PsiArc manager for working with multiple PsiArcs
pub struct PsiarcManager {
    /// Base directory for PsiArcs
    pub base_dir: std::path::PathBuf,
}

impl PsiarcManager {
    /// Create a new PsiarcManager
    pub fn new(base_dir: impl Into<std::path::PathBuf>) -> Self {
        Self {
            base_dir: base_dir.into(),
        }
    }

    /// List all PsiArcs in the base directory
    pub fn list_psiarcs(&self) -> std::io::Result<Vec<(std::path::PathBuf, PsiarcHeader)>> {
        header::list_psiarc_metadata(&self.base_dir)
    }

    /// Create a new PsiArc
    pub fn create_psiarc(
        &self,
        name: &str,
        title: Option<String>,
        tags: Option<Vec<String>>,
    ) -> std::io::Result<PsiArc> {
        // Generate a filename if not provided
        let filename = if name.is_empty() {
            format!(
                "Ïˆ-{}.psiarc",
                chrono::Utc::now().format("%Y-%m-%dT%H:%M:%S")
            )
        } else {
            if name.ends_with(".psiarc") {
                name.to_string()
            } else {
                format!("{}.psiarc", name)
            }
        };

        // Create the full path
        let path = self.base_dir.join(filename);

        // Create options
        let options = PsiarcOptions {
            path: path.clone(),
            create_sidecar: true,
            compress: true,
        };

        // Create a builder
        let mut builder = PsiarcBuilder::new(options);

        // Set title if provided
        if let Some(t) = title {
            builder = builder.with_title(t);
        }

        // Set tags if provided
        if let Some(t) = tags {
            builder = builder.with_tags(t);
        }

        // Build the PsiArc
        builder.build()
    }

    /// Open an existing PsiArc
    pub fn open_psiarc(&self, name: &str) -> std::io::Result<PsiArcReader> {
        // Check if the name includes the extension
        let filename = if name.ends_with(".psiarc") {
            name.to_string()
        } else {
            format!("{}.psiarc", name)
        };

        // Create the full path
        let path = self.base_dir.join(filename);

        // Open the PsiArc
        PsiArcReader::open(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::{set_current_session, PersonaMode, Session, User};
    use tempfile::tempdir;

    #[test]
    fn test_psiarc_builder() -> std::io::Result<()> {
        // Create a temporary directory
        let temp_dir = tempdir()?;
        let psiarc_path = temp_dir.path().join("test.psiarc");

        // Create options
        let options = PsiarcOptions {
            path: psiarc_path.clone(),
            create_sidecar: true,
            compress: true,
        };

        // Create a user and session
        let user = User::new(
            "test",
            "test_user",
            Some("test@example.com"),
            Some("Test User"),
            None,
        );

        let session = Session::new(user, PersonaMode::Researcher);
        set_current_session(session);

        // Create a PsiArc
        let builder = PsiarcBuilder::new(options)
            .with_title("Test PsiArc")
            .with_source("test_document.pdf")
            .with_tags(vec!["test".to_string(), "document".to_string()]);

        let psiarc = builder.build()?;

        // Verify the PsiArc
        assert_eq!(psiarc.header.title, Some("Test PsiArc".to_string()));
        assert_eq!(psiarc.header.source, Some("test_document.pdf".to_string()));
        assert_eq!(psiarc.header.tags, vec!["test", "document"]);

        // Verify the sidecar file was created
        let sidecar_path = PsiarcHeader::sidecar_path(&psiarc_path);
        assert!(sidecar_path.exists());

        // Load the header from the sidecar
        let loaded_header = PsiarcHeader::load_from_file(sidecar_path)?;

        // Verify the loaded header
        assert_eq!(loaded_header.title, Some("Test PsiArc".to_string()));
        assert_eq!(loaded_header.source, Some("test_document.pdf".to_string()));
        assert_eq!(loaded_header.tags, vec!["test", "document"]);

        Ok(())
    }
}

impl PsiarcManager {
    pub fn list_logs(&self) -> Vec<PathBuf> {
        // TODO: Implement actual log listing
        Vec::new()
    }
}

pub struct PsiarcLog {
    pub entries: Vec<PsiarcEntry>,
}

pub struct PsiarcEntry {
    pub timestamp: u64,
    pub data: Vec<u8>,
}
