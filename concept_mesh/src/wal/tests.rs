use std::sync::Arc;
use crate::wal::{WalConfig, WalEntry, SyncPolicy, init};
use crate::wal::writer::WalWriter;
use crate::wal::reader::WalReader;
use crate::wal::checkpoint::CheckpointManager;
use crate::wal::path::PathManager;
use crate::soliton_memory::SolitonMemoryVault;
use chrono::Utc;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use tokio;
    
    #[tokio::test]
    async fn test_wal_integration() -> Result<()> {
        // Setup
        let temp_dir = TempDir::new()?;
        let config = WalConfig {
            base_path: temp_dir.path().to_path_buf(),
            max_segment_size: 1024, // Small for testing
            max_segments: 3,
            enable_compression: false,
            sync_policy: SyncPolicy::Always,
        };
        
        // Initialize WAL
        init(config.clone()).await?;
        
        // Test writer
        let writer = Arc::new(WalWriter::new(config.clone()).await?);
        
        // Write some entries
        for i in 0..5 {
            let entry = WalEntry::MemoryOp {
                operation: "set".to_string(),
                key: format!("key{}", i),
                value: Some(format!("value{}", i).into_bytes()),
                timestamp: Utc::now(),
            };
            writer.write(&entry).await?;
        }
        
        // Test reader
        let reader = WalReader::new(config.clone())?;
        let entries = reader.read_all().await?;
        assert_eq!(entries.len(), 5);
        
        // Test checkpoint
        let checkpoint_mgr = CheckpointManager::new(config.clone())?;
        let checkpoint_result = checkpoint_mgr.create_checkpoint().await?;
        assert_eq!(checkpoint_result.entries_processed, 5);
        
        // Verify checkpoint exists
        let checkpoints = checkpoint_mgr.list_checkpoints().await?;
        assert_eq!(checkpoints.len(), 1);
        
        Ok(())
    }
    
    #[tokio::test]
    async fn test_soliton_memory_wal() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let config = WalConfig {
            base_path: temp_dir.path().to_path_buf(),
            ..Default::default()
        };
        
        init(config.clone()).await?;
        let wal_writer = Arc::new(WalWriter::new(config.clone()).await?);
        
        // Create vault with WAL
        let mut vault = SolitonMemoryVault::with_wal("user123".to_string(), wal_writer).await;
        
        // Store memory
        let memory_id = vault.store_memory(
            "test_concept".to_string(),
            "Test memory content".to_string(),
            0.9
        ).await?;
        
        // Verify WAL contains the entry
        let reader = WalReader::new(config)?;
        let entries = reader.read_all().await?;
        
        assert_eq!(entries.len(), 1);
        match &entries[0] {
            WalEntry::MemoryOp { operation, key, .. } => {
                assert_eq!(operation, "store");
                assert!(key.starts_with("soliton:"));
            }
            _ => panic!("Expected MemoryOp entry"),
        }
        
        Ok(())
    }
    
    #[tokio::test]
    async fn test_wal_rotation() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let config = WalConfig {
            base_path: temp_dir.path().to_path_buf(),
            max_segment_size: 100, // Very small to force rotation
            ..Default::default()
        };
        
        init(config.clone()).await?;
        let writer = Arc::new(WalWriter::new(config.clone()).await?);
        
        // Write enough to force rotation
        for i in 0..10 {
            let entry = WalEntry::MemoryOp {
                operation: "set".to_string(),
                key: format!("large_key_{}", i),
                value: Some(vec![0u8; 50]), // Large value
                timestamp: Utc::now(),
            };
            writer.write(&entry).await?;
        }
        
        // Check that multiple segments were created
        let path_mgr = PathManager::new(&config.base_path)?;
        let segments = path_mgr.list_segments()?;
        assert!(segments.len() > 1, "Expected multiple segments after rotation");
        
        Ok(())
    }
}

