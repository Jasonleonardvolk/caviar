use tokio::io::AsyncSeekExt;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::io::{AsyncWriteExt, BufWriter};
use tokio::fs::{File, OpenOptions};
use anyhow::{Result, Context};
use tracing::{info, debug, warn};

use super::{WalEntry, WalConfig, SyncPolicy};
use super::path::PathManager;

/// WAL writer for appending entries
pub struct WalWriter {
    config: WalConfig,
    path_manager: Arc<PathManager>,
    current_writer: Arc<Mutex<SegmentWriter>>,
}

/// Writer for a single WAL segment
struct SegmentWriter {
    file: BufWriter<File>,
    segment_id: u64,
    bytes_written: u64,
    entries_written: u64,
}

impl WalWriter {
    /// Create a new WAL writer
    pub async fn new(config: WalConfig) -> Result<Self> {
        let path_manager = Arc::new(PathManager::new(&config.base_path)?);
        
        // Open current segment for writing
        let current_path = path_manager.current_segment_path();
        let segment_id = path_manager.current_segment().load(std::sync::atomic::Ordering::SeqCst);
        
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&current_path)
            .await
            .context("Failed to open WAL segment for writing")?;
        
        let writer = SegmentWriter {
            file: BufWriter::new(file),
            segment_id,
            bytes_written: 0,
            entries_written: 0,
        };
        
        Ok(Self {
            config,
            path_manager,
            current_writer: Arc::new(Mutex::new(writer)),
        })
    }
    
    /// Write an entry to the WAL
    pub async fn write(&self, entry: &WalEntry) -> Result<()> {
        let serialized = self.serialize_entry(entry)?;
        let entry_size = serialized.len() as u64;
        
        let mut writer = self.current_writer.lock().await;
        
        // Check if we need to rotate segments
        if writer.bytes_written + entry_size > self.config.max_segment_size {
            debug!("Rotating WAL segment, current size: {}", writer.bytes_written);
            *writer = self.rotate_segment().await?;
        }
        
        // Write entry size (4 bytes) + data
        writer.file.write_u32_le(serialized.len() as u32).await?;
        writer.file.write_all(&serialized).await?;
        
        writer.bytes_written += 4 + entry_size;
        writer.entries_written += 1;
        
        // Apply sync policy
        match self.config.sync_policy {
            SyncPolicy::Always => {
                writer.file.flush().await?;
                writer.file.get_ref().sync_all().await?;
            }
            SyncPolicy::Periodic { .. } => {
                // Periodic sync handled by background task
                writer.file.flush().await?;
            }
            SyncPolicy::Never => {
                // Let OS handle syncing
            }
        }
        
        Ok(())
    }
    
    /// Write multiple entries as a batch
    pub async fn write_batch(&self, entries: &[WalEntry]) -> Result<()> {
        for entry in entries {
            self.write(entry).await?;
        }
        Ok(())
    }
    
    /// Force sync to disk
    pub async fn sync(&self) -> Result<()> {
        let writer = self.current_writer.lock().await;
        writer.file.get_ref().sync_all().await?;
        info!("WAL synced to disk, segment: {}, entries: {}", 
              writer.segment_id, writer.entries_written);
        Ok(())
    }
    
    /// Get current segment info
    pub async fn current_segment_info(&self) -> (u64, u64, u64) {
        let writer = self.current_writer.lock().await;
        (writer.segment_id, writer.bytes_written, writer.entries_written)
    }
    
    /// Rotate to a new segment
    async fn rotate_segment(&self) -> Result<SegmentWriter> {
        let (new_segment_id, new_path) = self.path_manager.next_segment();
        
        info!("Rotating to new WAL segment: {}", new_segment_id);
        
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .open(&new_path)
            .await
            .context("Failed to create new WAL segment")?;
        
        Ok(SegmentWriter {
            file: BufWriter::new(file),
            segment_id: new_segment_id,
            bytes_written: 0,
            entries_written: 0,
        })
    }
    
    /// Serialize WAL entry
    fn serialize_entry(&self, entry: &WalEntry) -> Result<Vec<u8>> {
        if self.config.enable_compression {
            // Use zstd compression
            let uncompressed = serde_json::to_vec(entry)?;
            Ok(zstd::encode_all(&uncompressed[..], 3)?)
        } else {
            Ok(serde_json::to_vec(entry)?)
        }
    }
}

/// Start periodic sync task if configured
pub async fn start_sync_task(writer: Arc<WalWriter>) {
    if let SyncPolicy::Periodic { interval_ms } = writer.config.sync_policy {
        tokio::spawn(async move {
            let interval = tokio::time::Duration::from_millis(interval_ms);
            let mut ticker = tokio::time::interval(interval);
            
            loop {
                ticker.tick().await;
                if let Err(e) = writer.sync().await {
                    warn!("Failed to sync WAL: {}", e);
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use chrono::Utc;
    
    #[tokio::test]
    async fn test_wal_writer() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let config = WalConfig {
            base_path: temp_dir.path().to_path_buf(),
            max_segment_size: 1024,
            ..Default::default()
        };
        
        let writer = WalWriter::new(config).await?;
        
        // Write some entries
        let entry = WalEntry::MemoryOp {
            operation: "set".to_string(),
            key: "test_key".to_string(),
            value: Some(b"test_value".to_vec()),
            timestamp: Utc::now(),
        };
        
        writer.write(&entry).await?;
        writer.sync().await?;
        
        let (segment_id, bytes, entries) = writer.current_segment_info().await;
        assert_eq!(segment_id, 0);
        assert!(bytes > 0);
        assert_eq!(entries, 1);
        
        Ok(())
    }
}



