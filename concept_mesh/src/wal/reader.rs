use tokio::io::AsyncSeekExt;
use std::path::{Path, PathBuf};
use tokio::io::{AsyncReadExt, BufReader};
use tokio::fs::File;
use anyhow::{Result, Context, bail};
use tracing::{debug, warn};

use super::{WalEntry, WalConfig};
use super::path::PathManager;

/// WAL reader for replaying entries
pub struct WalReader {
    config: WalConfig,
    path_manager: PathManager,
}

/// Iterator over WAL entries in a segment
pub struct SegmentIterator {
    reader: BufReader<File>,
    segment_id: u64,
    path: PathBuf,
    enable_compression: bool,
    entries_read: u64,
}

impl WalReader {
    /// Create a new WAL reader
    pub fn new(config: WalConfig) -> Result<Self> {
        let path_manager = PathManager::new(&config.base_path)?;
        
        Ok(Self {
            config,
            path_manager,
        })
    }
    
    /// Read all entries from a specific segment
    pub async fn read_segment(&self, segment_id: u64) -> Result<Vec<WalEntry>> {
        let mut entries = Vec::new();
        let mut iter = self.segment_iterator(segment_id).await?;
        
        while let Some(entry) = iter.next_entry().await? {
            entries.push(entry);
        }
        
        Ok(entries)
    }
    
    /// Create an iterator for a specific segment
    pub async fn segment_iterator(&self, segment_id: u64) -> Result<SegmentIterator> {
        let path = self.path_manager.segment_path(segment_id);
        
        let file = File::open(&path)
            .await
            .with_context(|| format!("Failed to open WAL segment: {}", path.display()))?;
        
        Ok(SegmentIterator {
            reader: BufReader::new(file),
            segment_id,
            path,
            enable_compression: self.config.enable_compression,
            entries_read: 0,
        })
    }
    
    /// Read all entries from a range of segments
    pub async fn read_range(&self, start_segment: u64, end_segment: u64) -> Result<Vec<WalEntry>> {
        let mut all_entries = Vec::new();
        
        for segment_id in start_segment..=end_segment {
            match self.read_segment(segment_id).await {
                Ok(entries) => all_entries.extend(entries),
                Err(e) => {
                    warn!("Failed to read segment {}: {}", segment_id, e);
                    // Continue with other segments
                }
            }
        }
        
        Ok(all_entries)
    }
    
    /// Read all available WAL entries
    pub async fn read_all(&self) -> Result<Vec<WalEntry>> {
        let segments = self.path_manager.list_segments()?;
        let mut all_entries = Vec::new();
        
        for (segment_id, _) in segments {
            match self.read_segment(segment_id).await {
                Ok(entries) => all_entries.extend(entries),
                Err(e) => {
                    warn!("Failed to read segment {}: {}", segment_id, e);
                }
            }
        }
        
        Ok(all_entries)
    }
    
    /// Find the last valid entry position in the WAL
    pub async fn find_last_valid_position(&self) -> Result<Option<(u64, u64)>> {
        let segments = self.path_manager.list_segments()?;
        
        for (segment_id, _) in segments.iter().rev() {
            let mut iter = self.segment_iterator(*segment_id).await?;
            let mut last_position = 0u64;
            let mut found_valid = false;
            
            while iter.next_entry().await?.is_some() {
                last_position = iter.reader.stream_position().await?;
                found_valid = true;
            }
            
            if found_valid {
                return Ok(Some((*segment_id, last_position)));
            }
        }
        
        Ok(None)
    }
}

impl SegmentIterator {
    /// Read the next entry from the segment
    pub async fn next_entry(&mut self) -> Result<Option<WalEntry>> {
        // Try to read entry size
        let mut size_buf = [0u8; 4];
        match self.reader.read_exact(&mut size_buf).await {
            Ok(_) => {},
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                return Ok(None);
            }
            Err(e) => return Err(e.into()),
        }
        
        let size = u32::from_le_bytes(size_buf) as usize;
        
        // Sanity check size
        if size > 100 * 1024 * 1024 {
            bail!("WAL entry size too large: {} bytes", size);
        }
        
        // Read entry data
        let mut data = vec![0u8; size];
        self.reader.read_exact(&mut data).await
            .context("Failed to read WAL entry data")?;
        
        // Deserialize entry
        let entry = if self.enable_compression {
            let decompressed = zstd::decode_all(&data[..])
                .context("Failed to decompress WAL entry")?;
            serde_json::from_slice(&decompressed)
                .context("Failed to deserialize WAL entry")?
        } else {
            serde_json::from_slice(&data)
                .context("Failed to deserialize WAL entry")?
        };
        
        self.entries_read += 1;
        Ok(Some(entry))
    }
    
    /// Get number of entries read so far
    pub fn entries_read(&self) -> u64 {
        self.entries_read
    }
}

/// Validate WAL integrity
pub async fn validate_wal(config: &WalConfig) -> Result<ValidationReport> {
    let reader = WalReader::new(config.clone())?;
    let segments = reader.path_manager.list_segments()?;
    
    let mut report = ValidationReport {
        total_segments: segments.len(),
        valid_segments: 0,
        corrupted_segments: Vec::new(),
        total_entries: 0,
        total_bytes: 0,
    };
    
    for (segment_id, path) in segments {
        match validate_segment(&reader, segment_id).await {
            Ok(stats) => {
                report.valid_segments += 1;
                report.total_entries += stats.entries;
                report.total_bytes += stats.bytes;
            }
            Err(e) => {
                report.corrupted_segments.push(CorruptedSegment {
                    segment_id,
                    path,
                    error: e.to_string(),
                });
            }
        }
    }
    
    Ok(report)
}

async fn validate_segment(reader: &WalReader, segment_id: u64) -> Result<SegmentStats> {
    let mut iter = reader.segment_iterator(segment_id).await?;
    let mut entries = 0u64;
    let start_pos = iter.reader.stream_position().await?;
    
    while iter.next_entry().await?.is_some() {
        entries += 1;
    }
    
    let end_pos = iter.reader.stream_position().await?;
    
    Ok(SegmentStats {
        entries,
        bytes: end_pos - start_pos,
    })
}

#[derive(Debug)]
pub struct ValidationReport {
    pub total_segments: usize,
    pub valid_segments: usize,
    pub corrupted_segments: Vec<CorruptedSegment>,
    pub total_entries: u64,
    pub total_bytes: u64,
}

#[derive(Debug)]
pub struct CorruptedSegment {
    pub segment_id: u64,
    pub path: PathBuf,
    pub error: String,
}

#[derive(Debug)]
struct SegmentStats {
    entries: u64,
    bytes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wal::{writer::{WalWriter, start_sync_task}};
    use tempfile::TempDir;
    use chrono::Utc;
    
    #[tokio::test]
    async fn test_wal_read_write() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let config = WalConfig {
            base_path: temp_dir.path().to_path_buf(),
            enable_compression: false,
            ..Default::default()
        };
        
        // Write some entries
        let writer = WalWriter::new(config.clone()).await?;
        
        let entries = vec![
            WalEntry::MemoryOp {
                operation: "set".to_string(),
                key: "key1".to_string(),
                value: Some(b"value1".to_vec()),
                timestamp: Utc::now(),
            },
            WalEntry::PhaseChange {
                from_phase: "alpha".to_string(),
                to_phase: "beta".to_string(),
                timestamp: Utc::now(),
            },
        ];
        
        for entry in &entries {
            writer.write(entry).await?;
        }
        writer.sync().await?;
        
        // Read them back
        let reader = WalReader::new(config)?;
        let read_entries = reader.read_all().await?;
        
        assert_eq!(read_entries.len(), entries.len());
        
        Ok(())
    }
}



