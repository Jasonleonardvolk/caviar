//! ψarc Logging System
//!
//! The psiarc (psi-archive) module provides a way to persistently store and replay
//! ConceptDiffs. This enables time-travel debugging, concept evolution tracking,
//! and system state reconstruction.

use crate::diff::{ConceptDiff, ConceptDiffRef};
use crate::lcn::LargeConceptNetwork;

use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, Write};
use std::path::Path;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

/// File extension for psiarc files
pub const PSIARC_EXTENSION: &str = ".psiarc";

/// Options for PsiarcLog configuration
#[derive(Debug, Clone)]
pub struct PsiarcOptions {
    /// Whether to append to existing log file
    pub append: bool,
    
    /// Whether to flush after each write
    pub flush_after_write: bool,
    
    /// Whether to compress the log file
    pub compress: bool,
    
    /// Maximum log file size before rotation
    pub max_size_bytes: Option<u64>,
    
    /// Directory to store logs in
    pub directory: String,
}

impl Default for PsiarcOptions {
    fn default() -> Self {
        Self {
            append: true,
            flush_after_write: true,
            compress: false,
            max_size_bytes: Some(100 * 1024 * 1024), // 100MB
            directory: "logs".to_string(),
        }
    }
}

/// A psiarc log for recording ConceptDiffs
pub struct PsiarcLog {
    /// File where diffs are stored
    file: Mutex<Option<File>>,
    
    /// Path to the psiarc file
    path: String,
    
    /// Configuration options
    options: PsiarcOptions,
    
    /// Current log file size in bytes
    size_bytes: RwLock<u64>,
    
    /// Whether this log has recorded a GENESIS event
    genesis_recorded: RwLock<bool>,
    
    /// Timestamp when this log was created
    created_at: u64,
}

impl PsiarcLog {
    /// Create a new psiarc log with the given name and options
    pub fn new(name: &str, options: PsiarcOptions) -> io::Result<Self> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        
        // Create directory if it doesn't exist
        std::fs::create_dir_all(&options.directory)?;
        
        // Generate filename: logs/name_timestamp.psiarc
        let filename = format!("{}_{}{}", name, timestamp, PSIARC_EXTENSION);
        let path = Path::new(&options.directory).join(filename);
        let path_str = path.to_string_lossy().to_string();
        
        // Open file for writing
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .append(options.append)
            .truncate(!options.append)
            .open(&path)?;
        
        // Get file size if appending to existing file
        let size_bytes = if options.append {
            file.metadata()?.len()
        } else {
            0
        };
        
        Ok(Self {
            file: Mutex::new(Some(file)),
            path: path_str,
            options,
            size_bytes: RwLock::new(size_bytes),
            genesis_recorded: RwLock::new(false),
            created_at: timestamp,
        })
    }
    
    /// Get the path to the psiarc file
    pub fn path(&self) -> &str {
        &self.path
    }
    
    /// Check if this log has recorded a GENESIS event
    pub fn has_genesis(&self) -> bool {
        *self.genesis_recorded.read().unwrap()
    }
    
    /// Get the creation timestamp
    pub fn created_at(&self) -> u64 {
        self.created_at
    }
    
    /// Get the current log file size in bytes
    pub fn size_bytes(&self) -> u64 {
        *self.size_bytes.read().unwrap()
    }
    
    /// Record a ConceptDiff to the log
    pub fn record(&self, diff: &ConceptDiff) -> io::Result<()> {
        // Check if this is a GENESIS diff
        if diff.frame_id == crate::GENESIS_FRAME_ID {
            let mut genesis = self.genesis_recorded.write().unwrap();
            *genesis = true;
        }
        
        // Serialize to JSON
        let json = serde_json::to_string(diff)?;
        
        // Write to file
        let mut file_guard = self.file.lock().unwrap();
        if let Some(file) = file_guard.as_mut() {
            // Add a newline after each diff (JSONL format)
            writeln!(file, "{}", json)?;
            
            // Flush if configured to do so
            if self.options.flush_after_write {
                file.flush()?;
            }
            
            // Update size
            let mut size = self.size_bytes.write().unwrap();
            *size += json.len() as u64 + 1; // +1 for newline
            
            // Check if we need to rotate the log
            if let Some(max_size) = self.options.max_size_bytes {
                if *size >= max_size {
                    // Close current file
                    *file_guard = None;
                    
                    // Create new file with timestamp
                    let timestamp = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs();
                    
                    let name = Path::new(&self.path)
                        .file_stem()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .to_string();
                    
                    let new_filename = format!("{}_{}{}", name, timestamp, PSIARC_EXTENSION);
                    let new_path = Path::new(&self.options.directory).join(new_filename);
                    
                    let new_file = OpenOptions::new()
                        .write(true)
                        .create(true)
                        .truncate(true)
                        .open(new_path)?;
                    
                    *file_guard = Some(new_file);
                    *size = 0;
                }
            }
        } else {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "Psiarc log file is not open",
            ));
        }
        
        Ok(())
    }
    
    /// Close the log file
    pub fn close(&self) -> io::Result<()> {
        let mut file_guard = self.file.lock().unwrap();
        *file_guard = None;
        Ok(())
    }
}

/// Reader for psiarc log files
pub struct PsiarcReader {
    /// BufReader for reading the file
    reader: BufReader<File>,
    
    /// Path to the psiarc file
    path: String,
}

impl PsiarcReader {
    /// Open a psiarc file for reading
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let file = File::open(&path)?;
        let reader = BufReader::new(file);
        let path_str = path.as_ref().to_string_lossy().to_string();
        
        Ok(Self {
            reader,
            path: path_str,
        })
    }
    
    /// Get the path to the psiarc file
    pub fn path(&self) -> &str {
        &self.path
    }
    
    /// Read the next ConceptDiff from the log
    pub fn next_diff(&mut self) -> io::Result<Option<ConceptDiff>> {
        let mut line = String::new();
        let bytes_read = self.reader.read_line(&mut line)?;
        
        if bytes_read == 0 {
            // End of file
            return Ok(None);
        }
        
        // Parse JSON
        match serde_json::from_str(&line) {
            Ok(diff) => Ok(Some(diff)),
            Err(e) => Err(io::Error::new(io::ErrorKind::InvalidData, e)),
        }
    }
    
    /// Read all ConceptDiffs from the log
    pub fn read_all(&mut self) -> io::Result<Vec<ConceptDiff>> {
        let mut diffs = Vec::new();
        
        while let Some(diff) = self.next_diff()? {
            diffs.push(diff);
        }
        
        Ok(diffs)
    }
    
    /// Apply all diffs in the log to a LargeConceptNetwork
    pub fn apply_to_lcn(&mut self, lcn: &LargeConceptNetwork) -> io::Result<usize> {
        let mut count = 0;
        
        while let Some(diff) = self.next_diff()? {
            if let Err(e) = lcn.apply_diff(&diff) {
                return Err(io::Error::new(
                    io::ErrorKind::Other,
                    format!("Failed to apply diff: {}", e),
                ));
            }
            count += 1;
        }
        
        Ok(count)
    }
}

/// A PsiarcManager handles multiple psiarc log files
pub struct PsiarcManager {
    /// Directory where logs are stored
    directory: String,
    
    /// Currently active log
    active_log: Mutex<Option<Arc<PsiarcLog>>>,
    
    /// Default options for new logs
    default_options: PsiarcOptions,
}

impl PsiarcManager {
    /// Create a new PsiarcManager
    pub fn new(directory: impl Into<String>) -> Self {
        let directory = directory.into();
        let options = PsiarcOptions {
            directory: directory.clone(),
            ..Default::default()
        };
        
        Self {
            directory,
            active_log: Mutex::new(None),
            default_options: options,
        }
    }
    
    /// Create a new psiarc log
    pub fn create_log(&self, name: &str) -> io::Result<Arc<PsiarcLog>> {
        let log = Arc::new(PsiarcLog::new(name, self.default_options.clone())?);
        let mut active_log = self.active_log.lock().unwrap();
        *active_log = Some(Arc::clone(&log));
        Ok(log)
    }
    
    /// Create a new psiarc log with custom options
    pub fn create_log_with_options(&self, name: &str, options: PsiarcOptions) -> io::Result<Arc<PsiarcLog>> {
        let log = Arc::new(PsiarcLog::new(name, options)?);
        let mut active_log = self.active_log.lock().unwrap();
        *active_log = Some(Arc::clone(&log));
        Ok(log)
    }
    
    /// Get the active log
    pub fn active_log(&self) -> Option<Arc<PsiarcLog>> {
        let active_log = self.active_log.lock().unwrap();
        active_log.clone()
    }
    
    /// Get all psiarc files in the directory
    pub fn list_logs(&self) -> io::Result<Vec<String>> {
        let mut logs = Vec::new();
        
        for entry in std::fs::read_dir(&self.directory)? {
            let entry = entry?;
            let path = entry.path();
            
            if path.is_file() && path.extension().map_or(false, |ext| ext == "psiarc") {
                if let Some(filename) = path.file_name() {
                    logs.push(filename.to_string_lossy().to_string());
                }
            }
        }
        
        logs.sort();
        Ok(logs)
    }
    
    /// Open a psiarc file for reading
    pub fn open_log(&self, filename: &str) -> io::Result<PsiarcReader> {
        let path = Path::new(&self.directory).join(filename);
        PsiarcReader::open(path)
    }
    
    /// Record a ConceptDiff to the active log
    pub fn record(&self, diff: &ConceptDiff) -> io::Result<()> {
        let active_log = self.active_log.lock().unwrap();
        
        if let Some(log) = active_log.as_ref() {
            log.record(diff)
        } else {
            Err(io::Error::new(
                io::ErrorKind::Other,
                "No active psiarc log",
            ))
        }
    }
}

impl Default for PsiarcManager {
    fn default() -> Self {
        Self::new("logs")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diff::{Op, ConceptDiffBuilder};
    use std::collections::HashMap;
    use tempfile::tempdir;
    
    #[test]
    fn test_psiarc_write_read() -> io::Result<()> {
        // Create temporary directory
        let temp_dir = tempdir()?;
        let dir_path = temp_dir.path().to_string_lossy().to_string();
        
        // Create options
        let options = PsiarcOptions {
            directory: dir_path.clone(),
            ..Default::default()
        };
        
        // Create log
        let log = PsiarcLog::new("test", options)?;
        
        // Create some diffs
        let diff1 = ConceptDiffBuilder::new(1)
            .create("node1", "TestNode", HashMap::new())
            .build();
        
        let diff2 = ConceptDiffBuilder::new(2)
            .create("node2", "TestNode", HashMap::new())
            .bind("node1", "node2", None)
            .build();
        
        // Record diffs
        log.record(&diff1)?;
        log.record(&diff2)?;
        
        // Close log
        log.close()?;
        
        // Open log for reading
        let mut reader = PsiarcReader::open(log.path())?;
        
        // Read diffs
        let read_diff1 = reader.next_diff()?.unwrap();
        let read_diff2 = reader.next_diff()?.unwrap();
        
        // Verify
        assert_eq!(read_diff1.frame_id, 1);
        assert_eq!(read_diff2.frame_id, 2);
        
        Ok(())
    }
    
    #[test]
    fn test_psiarc_manager() -> io::Result<()> {
        // Create temporary directory
        let temp_dir = tempdir()?;
        let dir_path = temp_dir.path().to_string_lossy().to_string();
        
        // Create manager
        let manager = PsiarcManager::new(&dir_path);
        
        // Create log
        let log = manager.create_log("test")?;
        
        // Create and record a diff
        let diff = ConceptDiffBuilder::new(1)
            .create("node1", "TestNode", HashMap::new())
            .build();
        
        manager.record(&diff)?;
        
        // List logs
        let logs = manager.list_logs()?;
        assert_eq!(logs.len(), 1);
        
        Ok(())
    }
    
    #[test]
    fn test_apply_to_lcn() -> io::Result<()> {
        // Create temporary directory
        let temp_dir = tempdir()?;
        let dir_path = temp_dir.path().to_string_lossy().to_string();
        
        // Create options
        let options = PsiarcOptions {
            directory: dir_path.clone(),
            ..Default::default()
        };
        
        // Create log
        let log = PsiarcLog::new("test", options)?;
        
        // Create a GENESIS diff
        let genesis_diff = crate::diff::create_genesis_diff("TestCorpus");
        log.record(&genesis_diff)?;
        
        // Create some additional diffs
        let diff = ConceptDiffBuilder::new(1)
            .create("node1", "TestNode", HashMap::new())
            .build();
        
        log.record(&diff)?;
        
        // Close log
        log.close()?;
        
        // Create LCN
        let lcn = LargeConceptNetwork::new();
        
        // Open log for reading
        let mut reader = PsiarcReader::open(log.path())?;
        
        // Apply diffs to LCN
        let count = reader.apply_to_lcn(&lcn)?;
        
        // Verify
        assert_eq!(count, 2);
        assert!(lcn.is_genesis_complete());
        assert!(lcn.has_timeless_root());
        assert!(lcn.get_node(&"node1".to_string()).is_some());
        
        Ok(())
    }
}
