use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use anyhow::{Result, Context};
use chrono::{DateTime, Utc};
use tracing::{info, warn, error};

use super::{WalConfig, WalEntry};
use super::path::PathManager;
use super::reader::WalReader;

/// Checkpoint manager for creating consistent snapshots
pub struct CheckpointManager {
    config: WalConfig,
    path_manager: Arc<PathManager>,
    active_checkpoint: Arc<RwLock<Option<ActiveCheckpoint>>>,
}

/// Active checkpoint being created
struct ActiveCheckpoint {
    id: String,
    start_segment: u64,
    end_segment: u64,
    entries_processed: u64,
    started_at: DateTime<Utc>,
}

/// Checkpoint result
#[derive(Debug)]
pub struct CheckpointResult {
    pub id: String,
    pub segments_included: Vec<u64>,
    pub entries_processed: u64,
    pub size_bytes: u64,
    pub duration_ms: u64,
}

impl CheckpointManager {
    /// Create a new checkpoint manager
    pub fn new(config: WalConfig) -> Result<Self> {
        let path_manager = Arc::new(PathManager::new(&config.base_path)?);
        
        Ok(Self {
            config,
            path_manager,
            active_checkpoint: Arc::new(RwLock::new(None)),
        })
    }
    
    /// Create a checkpoint from WAL segments
    pub async fn create_checkpoint(&self) -> Result<CheckpointResult> {
        let start_time = Utc::now();
        let checkpoint_id = format!("ckpt_{}", start_time.timestamp());
        
        // Determine segment range
        let segments = self.path_manager.list_segments()?;
        if segments.is_empty() {
            return Ok(CheckpointResult {
                id: checkpoint_id,
                segments_included: vec![],
                entries_processed: 0,
                size_bytes: 0,
                duration_ms: 0,
            });
        }
        
        let start_segment = segments.first().unwrap().0;
        let end_segment = segments.last().unwrap().0;
        
        // Set active checkpoint
        {
            let mut active = self.active_checkpoint.write().await;
            *active = Some(ActiveCheckpoint {
                id: checkpoint_id.clone(),
                start_segment,
                end_segment,
                entries_processed: 0,
                started_at: start_time,
            });
        }
        
        info!("Starting checkpoint {} for segments {}-{}", 
              checkpoint_id, start_segment, end_segment);
        
        // Create checkpoint
        let result = self.perform_checkpoint(&checkpoint_id, start_segment, end_segment).await?;
        
        // Clear active checkpoint
        {
            let mut active = self.active_checkpoint.write().await;
            *active = None;
        }
        
        let duration_ms = (Utc::now() - start_time).num_milliseconds() as u64;
        
        Ok(CheckpointResult {
            id: checkpoint_id,
            segments_included: (start_segment..=end_segment).collect(),
            entries_processed: result.entries_processed,
            size_bytes: result.size_bytes,
            duration_ms,
        })
    }
    
    /// Perform the actual checkpoint creation
    async fn perform_checkpoint(
        &self, 
        checkpoint_id: &str, 
        start_segment: u64, 
        end_segment: u64
    ) -> Result<CheckpointData> {
        let checkpoint_path = self.path_manager.checkpoint_path(checkpoint_id);
        
        // Ensure checkpoint directory exists
        tokio::fs::create_dir_all(self.path_manager.checkpoint_dir()).await?;
        
        // Create temporary file for checkpoint
        let temp_path = checkpoint_path.with_extension("tmp");
        let mut file = File::create(&temp_path).await?;
        
        // Read and consolidate WAL entries
        let reader = WalReader::new(self.config.clone())?;
        let mut entries_processed = 0u64;
        let mut state = HashMap::new();
        
        for segment_id in start_segment..=end_segment {
            match reader.read_segment(segment_id).await {
                Ok(entries) => {
                    for entry in entries {
                        self.apply_entry_to_state(&mut state, &entry)?;
                        entries_processed += 1;
                        
                        // Update active checkpoint progress
                        if entries_processed % 1000 == 0 {
                            let mut active = self.active_checkpoint.write().await;
                            if let Some(ckpt) = active.as_mut() {
                                ckpt.entries_processed = entries_processed;
                            }
                        }
                    }
                }
                Err(e) => {
                    warn!("Failed to read segment {} during checkpoint: {}", segment_id, e);
                }
            }
        }
        
        // Write consolidated state
        let checkpoint_data = CheckpointData {
            id: checkpoint_id.to_string(),
            segments: (start_segment..=end_segment).collect(),
            entries_processed,
            state,
            created_at: Utc::now(),
        };
        
        let serialized = serde_json::to_vec_pretty(&checkpoint_data)?;
        file.write_all(&serialized).await?;
        file.sync_all().await?;
        
        // Atomically rename to final path
        tokio::fs::rename(&temp_path, &checkpoint_path).await?;
        
        info!("Checkpoint {} created: {} entries, {} bytes", 
              checkpoint_id, entries_processed, serialized.len());
        
        Ok(CheckpointData {
            size_bytes: serialized.len() as u64,
            ..checkpoint_data
        })
    }
    
    /// Apply a WAL entry to the checkpoint state
    fn apply_entry_to_state(
        &self, 
        state: &mut HashMap<String, serde_json::Value>,
        entry: &WalEntry
    ) -> Result<()> {
        match entry {
            WalEntry::MemoryOp { operation, key, value, .. } => {
                match operation.as_str() {
                    "set" => {
                        if let Some(val) = value {
                            state.insert(key.clone(), serde_json::from_slice(val)?);
                        }
                    }
                    "delete" => {
                        state.remove(key);
                    }
                    _ => {}
                }
            }
            WalEntry::ConceptDiff { id, diff, .. } => {
                // Store concept diff in state
                state.insert(
                    format!("concept:{}", id),
                    serde_json::json!({
                        "type": "concept_diff",
                        "data": diff,
                    })
                );
            }
            _ => {}
        }
        Ok(())
    }
    
    /// Restore from a checkpoint
    pub async fn restore_from_checkpoint(&self, checkpoint_id: &str) -> Result<()> {
        let checkpoint_path = self.path_manager.checkpoint_path(checkpoint_id);
        
        info!("Restoring from checkpoint: {}", checkpoint_id);
        
        let data = tokio::fs::read(&checkpoint_path).await
            .context("Failed to read checkpoint file")?;
        
        let checkpoint: CheckpointData = serde_json::from_slice(&data)
            .context("Failed to parse checkpoint data")?;
        
        // TODO: Apply checkpoint state to the system
        // This would involve restoring the concept mesh state
        
        info!("Restored {} entries from checkpoint {}", 
              checkpoint.entries_processed, checkpoint_id);
        
        Ok(())
    }
    
    /// List available checkpoints
    pub async fn list_checkpoints(&self) -> Result<Vec<CheckpointInfo>> {
        let checkpoint_dir = self.path_manager.checkpoint_dir();
        let mut checkpoints = Vec::new();
        
        let mut entries = tokio::fs::read_dir(&checkpoint_dir).await?;
        
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.extension().map(|e| e == "checkpoint").unwrap_or(false) {
                if let Ok(metadata) = entry.metadata().await {
                    if let Some(name) = path.file_stem().and_then(|s| s.to_str()) {
                        checkpoints.push(CheckpointInfo {
                            id: name.to_string(),
                            size_bytes: metadata.len(),
                            created_at: metadata.modified()?.into(),
                        });
                    }
                }
            }
        }
        
        checkpoints.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        Ok(checkpoints)
    }
}

/// Checkpoint data structure
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct CheckpointData {
    id: String,
    segments: Vec<u64>,
    entries_processed: u64,
    state: HashMap<String, serde_json::Value>,
    created_at: DateTime<Utc>,
    #[serde(skip)]
    size_bytes: u64,
}

/// Checkpoint information
#[derive(Debug)]
pub struct CheckpointInfo {
    pub id: String,
    pub size_bytes: u64,
    pub created_at: DateTime<Utc>,
}

/// Start automatic checkpoint manager
pub async fn start_checkpoint_manager(config: WalConfig) -> Result<()> {
    let manager = Arc::new(CheckpointManager::new(config.clone())?);
    
    // Start background checkpoint task
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(300)); // 5 minutes
        
        loop {
            interval.tick().await;
            
            // Check if we need to create a checkpoint
            if let Ok(segments) = manager.path_manager.list_segments() {
                if segments.len() >= manager.config.max_segments {
                    match manager.create_checkpoint().await {
                        Ok(result) => {
                            info!("Automatic checkpoint created: {:?}", result);
                            
                            // Clean up old segments
                            if let Err(e) = manager.path_manager
                                .cleanup_checkpointed_segments(&result.id).await {
                                error!("Failed to cleanup segments: {}", e);
                            }
                        }
                        Err(e) => {
                            error!("Failed to create automatic checkpoint: {}", e);
                        }
                    }
                }
            }
        }
    });
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wal::writer::WalWriter;
    use tempfile::TempDir;
    
    #[tokio::test]
    async fn test_checkpoint_creation() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let config = WalConfig {
            base_path: temp_dir.path().to_path_buf(),
            enable_compression: false,
            ..Default::default()
        };
        
        // Write some entries
        let writer = WalWriter::new(config.clone()).await?;
        
        for i in 0..10 {
            let entry = WalEntry::MemoryOp {
                operation: "set".to_string(),
                key: format!("key{}", i),
                value: Some(format!("value{}", i).into_bytes()),
                timestamp: Utc::now(),
            };
            writer.write(&entry).await?;
        }
        writer.sync().await?;
        
        // Create checkpoint
        let manager = CheckpointManager::new(config)?;
        let result = manager.create_checkpoint().await?;
        
        assert_eq!(result.entries_processed, 10);
        assert!(!result.segments_included.is_empty());
        
        // Verify checkpoint exists
        let checkpoints = manager.list_checkpoints().await?;
        assert_eq!(checkpoints.len(), 1);
        assert_eq!(checkpoints[0].id, result.id);
        
        Ok(())
    }
}

