use anyhow::Result;
use clap::{Parser, Subcommand};
use concept_mesh_rs::wal::{
    checkpoint::{start_checkpoint_manager, CheckpointManager},
    reader::validate_wal,
    WalConfig, WalReader, WalWriter,
};
use std::path::PathBuf;
use tracing::{error, info};
use tracing_subscriber;

#[derive(Parser)]
#[command(name = "wal-cli")]
#[command(about = "WAL management CLI for Concept Mesh", long_about = None)]
struct Cli {
    /// WAL base directory
    #[arg(short, long, default_value = "data/wal")]
    base_path: PathBuf,

    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Validate WAL integrity
    Validate,

    /// Show WAL statistics
    Stats,

    /// List all segments
    ListSegments,

    /// Create a checkpoint
    Checkpoint,

    /// List available checkpoints
    ListCheckpoints,

    /// Restore from a checkpoint
    Restore {
        /// Checkpoint ID to restore from
        checkpoint_id: String,
    },

    /// Clean up old segments
    Cleanup {
        /// Keep this many recent segments
        #[arg(short, long, default_value = "10")]
        keep: usize,
    },

    /// Replay WAL entries
    Replay {
        /// Start segment (inclusive)
        #[arg(short, long)]
        start: Option<u64>,

        /// End segment (inclusive)
        #[arg(short, long)]
        end: Option<u64>,

        /// Show full entry details
        #[arg(short, long)]
        full: bool,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    let level = if cli.verbose { "debug" } else { "info" };
    tracing_subscriber::fmt().with_env_filter(level).init();

    let config = WalConfig {
        base_path: cli.base_path,
        ..Default::default()
    };

    match cli.command {
        Commands::Validate => validate(&config).await?,
        Commands::Stats => show_stats(&config).await?,
        Commands::ListSegments => list_segments(&config).await?,
        Commands::Checkpoint => create_checkpoint(&config).await?,
        Commands::ListCheckpoints => list_checkpoints(&config).await?,
        Commands::Restore { checkpoint_id } => restore(&config, &checkpoint_id).await?,
        Commands::Cleanup { keep } => cleanup(&config, keep).await?,
        Commands::Replay { start, end, full } => replay(&config, start, end, full).await?,
    }

    Ok(())
}

async fn validate(config: &WalConfig) -> Result<()> {
    info!("Validating WAL integrity...");

    let report = validate_wal(config).await?;

    println!("\nValidation Report:");
    println!("==================");
    println!("Total segments: {}", report.total_segments);
    println!("Valid segments: {}", report.valid_segments);
    println!("Total entries: {}", report.total_entries);
    println!("Total size: {} MB", report.total_bytes / (1024 * 1024));

    if !report.corrupted_segments.is_empty() {
        println!("\nCorrupted segments:");
        for segment in &report.corrupted_segments {
            println!("  - Segment {}: {}", segment.segment_id, segment.error);
        }
    } else {
        println!("\nâœ“ All segments are valid!");
    }

    Ok(())
}

async fn show_stats(config: &WalConfig) -> Result<()> {
    let reader = WalReader::new(config.clone())?;
    let segments = reader.path_manager.list_segments()?;

    let mut total_entries = 0u64;
    let mut total_bytes = 0u64;

    for (segment_id, path) in &segments {
        if let Ok(metadata) = tokio::fs::metadata(&path).await {
            total_bytes += metadata.len();

            if let Ok(entries) = reader.read_segment(*segment_id).await {
                total_entries += entries.len() as u64;
            }
        }
    }

    println!("\nWAL Statistics:");
    println!("===============");
    println!("Total segments: {}", segments.len());
    println!("Total entries: {}", total_entries);
    println!(
        "Total size: {:.2} MB",
        total_bytes as f64 / (1024.0 * 1024.0)
    );

    if let Some((first, _)) = segments.first() {
        println!("First segment: {}", first);
    }
    if let Some((last, _)) = segments.last() {
        println!("Last segment: {}", last);
    }

    Ok(())
}

async fn list_segments(config: &WalConfig) -> Result<()> {
    let reader = WalReader::new(config.clone())?;
    let segments = reader.path_manager.list_segments()?;

    println!("\nWAL Segments:");
    println!("=============");

    for (segment_id, path) in segments {
        if let Ok(metadata) = tokio::fs::metadata(&path).await {
            let size_mb = metadata.len() as f64 / (1024.0 * 1024.0);
            let modified = metadata.modified()?;

            println!(
                "  Segment {:08}: {:.2} MB - {}",
                segment_id,
                size_mb,
                humantime::format_rfc3339(modified)
            );
        }
    }

    Ok(())
}

async fn create_checkpoint(config: &WalConfig) -> Result<()> {
    info!("Creating checkpoint...");

    let manager = CheckpointManager::new(config.clone())?;
    let result = manager.create_checkpoint().await?;

    println!("\nCheckpoint created successfully!");
    println!("================================");
    println!("ID: {}", result.id);
    println!("Segments included: {:?}", result.segments_included);
    println!("Entries processed: {}", result.entries_processed);
    println!(
        "Size: {:.2} MB",
        result.size_bytes as f64 / (1024.0 * 1024.0)
    );
    println!("Duration: {} ms", result.duration_ms);

    Ok(())
}

async fn list_checkpoints(config: &WalConfig) -> Result<()> {
    let manager = CheckpointManager::new(config.clone())?;
    let checkpoints = manager.list_checkpoints().await?;

    println!("\nAvailable Checkpoints:");
    println!("=====================");

    if checkpoints.is_empty() {
        println!("  No checkpoints found.");
    } else {
        for ckpt in checkpoints {
            println!(
                "  {} - {:.2} MB - {}",
                ckpt.id,
                ckpt.size_bytes as f64 / (1024.0 * 1024.0),
                ckpt.created_at.format("%Y-%m-%d %H:%M:%S UTC")
            );
        }
    }

    Ok(())
}

async fn restore(config: &WalConfig, checkpoint_id: &str) -> Result<()> {
    info!("Restoring from checkpoint: {}", checkpoint_id);

    let manager = CheckpointManager::new(config.clone())?;
    manager.restore_from_checkpoint(checkpoint_id).await?;

    println!("âœ“ Successfully restored from checkpoint: {}", checkpoint_id);

    Ok(())
}

async fn cleanup(config: &WalConfig, keep: usize) -> Result<()> {
    let reader = WalReader::new(config.clone())?;
    let segments = reader.path_manager.list_segments()?;

    if segments.len() <= keep {
        println!(
            "Nothing to clean up. Current segments: {}, keep threshold: {}",
            segments.len(),
            keep
        );
        return Ok(());
    }

    let to_remove = segments.len() - keep;
    println!("Removing {} old segments...", to_remove);

    for (segment_id, path) in segments.iter().take(to_remove) {
        tokio::fs::remove_file(path).await?;
        println!("  Removed segment {}", segment_id);
    }

    println!("âœ“ Cleanup complete!");

    Ok(())
}

async fn replay(
    config: &WalConfig,
    start: Option<u64>,
    end: Option<u64>,
    full: bool,
) -> Result<()> {
    let reader = WalReader::new(config.clone())?;

    let entries = if let (Some(start), Some(end)) = (start, end) {
        reader.read_range(start, end).await?
    } else {
        reader.read_all().await?
    };

    println!("\nReplaying {} WAL entries:", entries.len());
    println!("========================");

    for (i, entry) in entries.iter().enumerate() {
        if full {
            println!("\n[Entry {}]\n{:#?}", i + 1, entry);
        } else {
            match entry {
                concept_mesh::wal::WalEntry::ConceptDiff { id, timestamp, .. } => {
                    println!(
                        "  [{}] ConceptDiff: {} at {}",
                        i + 1,
                        id,
                        timestamp.format("%H:%M:%S")
                    );
                }
                concept_mesh::wal::WalEntry::MemoryOp {
                    operation,
                    key,
                    timestamp,
                    ..
                } => {
                    println!(
                        "  [{}] MemoryOp: {} {} at {}",
                        i + 1,
                        operation,
                        key,
                        timestamp.format("%H:%M:%S")
                    );
                }
                concept_mesh::wal::WalEntry::PhaseChange {
                    from_phase,
                    to_phase,
                    timestamp,
                } => {
                    println!(
                        "  [{}] PhaseChange: {} -> {} at {}",
                        i + 1,
                        from_phase,
                        to_phase,
                        timestamp.format("%H:%M:%S")
                    );
                }
                concept_mesh::wal::WalEntry::Checkpoint { id, timestamp, .. } => {
                    println!(
                        "  [{}] Checkpoint: {} at {}",
                        i + 1,
                        id,
                        timestamp.format("%H:%M:%S")
                    );
                }
            }
        }
    }

    Ok(())
}

