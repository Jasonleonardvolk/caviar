use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use anyhow::{Result, Context};
use chrono::{DateTime, Utc};

/// Manages WAL file paths and segments
pub struct PathManager {
    base_path: PathBuf,
    current_segment: Arc<AtomicU64>,
}

impl PathManager {
    /// Create a new path manager
    pub fn new(base_path: impl AsRef<Path>) -> Result<Self> {
        let base_path = base_path.as_ref().to_path_buf();
        
        // Find the highest existing segment number
        let current_segment = Self::find_latest_segment(&base_path)?;
        
        Ok(Self {
            base_path,
            current_segment: Arc::new(AtomicU64::new(current_segment)),
        })
    }
    
    /// Get current segment atomic counter
    pub fn current_segment(&self) -> &AtomicU64 {
        &self.current_segment
    }
    
    /// Get path for a specific segment
    pub fn segment_path(&self, segment_id: u64) -> PathBuf {
        self.base_path.join(format!("wal_{:08}.log", segment_id))
    }
    
    /// Get path for current segment
    pub fn current_segment_path(&self) -> PathBuf {
        let segment_id = self.current_segment.load(Ordering::SeqCst);
        self.segment_path(segment_id)
    }
    
    /// Advance to next segment
    pub fn next_segment(&self) -> (u64, PathBuf) {
        let new_segment = self.current_segment.fetch_add(1, Ordering::SeqCst) + 1;
        (new_segment, self.segment_path(new_segment))
    }
    
    /// Get checkpoint directory
    pub fn checkpoint_dir(&self) -> PathBuf {
        self.base_path.join("checkpoints")
    }
    
    /// Get checkpoint path for a specific ID
    pub fn checkpoint_path(&self, checkpoint_id: &str) -> PathBuf {
        self.checkpoint_dir().join(format!("{}.checkpoint", checkpoint_id))
    }
    
    /// List all WAL segments
    pub fn list_segments(&self) -> Result<Vec<(u64, PathBuf)>> {
        let mut segments = Vec::new();
        
        let entries = std::fs::read_dir(&self.base_path)
            .context("Failed to read WAL directory")?;
        
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            
            if let Some(segment_id) = Self::parse_segment_id(&path) {
                segments.push((segment_id, path));
            }
        }
        
        segments.sort_by_key(|(id, _)| *id);
        Ok(segments)
    }
    
    /// List segments in a range
    pub fn list_segments_range(&self, start: u64, end: u64) -> Result<Vec<(u64, PathBuf)>> {
        let all_segments = self.list_segments()?;
        Ok(all_segments.into_iter()
            .filter(|(id, _)| *id >= start && *id <= end)
            .collect())
    }
    
    /// Parse segment ID from filename
    fn parse_segment_id(path: &Path) -> Option<u64> {
        let filename = path.file_name()?.to_str()?;
        
        if filename.starts_with("wal_") && filename.ends_with(".log") {
            let id_str = &filename[4..12];
            id_str.parse().ok()
        } else {
            None
        }
    }
    
    /// Find the latest segment number
    fn find_latest_segment(base_path: &Path) -> Result<u64> {
        if !base_path.exists() {
            return Ok(0);
        }
        
        let segments = Self::list_segments(&Self {
            base_path: base_path.to_path_buf(),
            current_segment: Arc::new(AtomicU64::new(0)),
        })?;
        
        Ok(segments.last().map(|(id, _)| *id).unwrap_or(0))
    }
    
    /// Clean up old segments that have been checkpointed
    pub async fn cleanup_checkpointed_segments(&self, checkpoint_id: &str) -> Result<Vec<PathBuf>> {
        let checkpoint_path = self.checkpoint_path(checkpoint_id);
        
        // Read checkpoint metadata to find included segments
        let metadata = tokio::fs::read_to_string(&checkpoint_path).await?;
        let checkpoint_info: CheckpointInfo = serde_json::from_str(&metadata)?;
        
        let mut removed = Vec::new();
        
        for segment_id in checkpoint_info.segments {
            let path = self.segment_path(segment_id);
            if path.exists() {
                tokio::fs::remove_file(&path).await?;
                removed.push(path);
            }
        }
        
        Ok(removed)
    }
}

/// Checkpoint metadata
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct CheckpointInfo {
    id: String,
    segments: Vec<u64>,
    created_at: DateTime<Utc>,
    entry_count: u64,
}

/// Initialize the path subsystem
pub fn init(base_path: impl AsRef<Path>) -> Result<()> {
    let base_path = base_path.as_ref();
    
    // Create base directory
    std::fs::create_dir_all(base_path)?;
    
    // Create checkpoints subdirectory
    std::fs::create_dir_all(base_path.join("checkpoints"))?;
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[test]
    fn test_path_manager() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let manager = PathManager::new(temp_dir.path())?;
        
        // Test current segment path
        let current = manager.current_segment_path();
        assert!(current.to_str().unwrap().contains("wal_00000000.log"));
        
        // Test next segment
        let (id, path) = manager.next_segment();
        assert_eq!(id, 1);
        assert!(path.to_str().unwrap().contains("wal_00000001.log"));
        
        Ok(())
    }
}


