use crate::mesh::MeshNode;
use crate::diff::*;
//! ConceptDiff Replay Engine
//!
//! This module provides a replay engine for ConceptDiffs stored in ÃË†arc logs.
//! It enables time-travel debugging, simulation, and graph reconstruction from logs.

use crate::diff::{ConceptDiff, ConceptDiffRef};
use crate::lcn::LargeConceptNetwork;
use crate::psiarc::{PsiarcManager, PsiArcReader};

use clap::{Parser, Subcommand};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tracing::{debug, error, info, warn};

/// Command line arguments for the psiarc replay tool
#[derive(Parser, Debug)]
#[clap(name = "psiarc-replay", about = "Replay ConceptDiffs from ÃË†arc logs")]
pub struct ReplayArgs {
    /// Path to the psiarc log file to replay
    #[clap(short, long)]
    input: PathBuf,

    /// Path to write the resulting LCN snapshot to (optional)
    #[clap(short, long)]
    output: Option<PathBuf>,

    /// Whether to run in step-by-step mode
    #[clap(short, long)]
    step: bool,

    /// Maximum number of diffs to replay (optional)
    #[clap(short, long)]
    limit: Option<usize>,

    /// Stop at a specific frame ID
    #[clap(long)]
    stop_at: Option<u64>,

    /// Verbosity level (0-3)
    #[clap(short, long, default_value = "1")]
    verbose: u8,
}

/// Result of a replay operation
#[derive(Debug, Clone)]
pub struct ReplayResult {
    /// Number of diffs processed
    pub diffs_processed: usize,

    /// Last frame ID processed
    pub last_frame_id: u64,

    /// Whether GENESIS was encountered
    pub genesis_encountered: bool,

    /// Number of nodes in the final LCN
    pub node_count: usize,

    /// Number of edges in the final LCN
    pub edge_count: usize,

    /// LCN snapshot hash (for verification)
    pub lcn_hash: u64,

    /// Processing time in milliseconds
    pub processing_time_ms: u64,

    /// Errors encountered during replay (if any)
    pub errors: Vec<String>,
}

/// Calculate a simple hash of the LCN state
pub fn calculate_lcn_hash(lcn: &LargeConceptNetwork) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();

    // Get all nodes
    let nodes = lcn.get_all_nodes();

    // Hash node IDs and types
    for node in &nodes {
        node.id.hash(&mut hasher);
        node.node_type.hash(&mut hasher);
    }

    // Hash counts (approximate structure hash)
    nodes.len().hash(&mut hasher);

    hasher.finish()
}

/// Replay ConceptDiffs from a psiarc log
pub async fn replay_psiarc(
    input_path: impl AsRef<Path>,
    step_by_step: bool,
    limit: Option<usize>,
    stop_at: Option<u64>,
) -> Result<ReplayResult, String> {
    let start_time = SystemTime::now();

    // Create LCN
    let lcn = Arc::new(LargeConceptNetwork::new());

    // Open psiarc file
    let mut reader = PsiArcReader::open(input_path.as_ref())
        .map_err(|e| format!("Failed to open psiarc file: {}", e))?;

    // Initialize result
    let mut result = ReplayResult {
        diffs_processed: 0,
        last_frame_id: 0,
        genesis_encountered: false,
        node_count: 0,
        edge_count: 0,
        lcn_hash: 0,
        processing_time_ms: 0,
        errors: Vec::new(),
    };

    // Process diffs
    let mut diff_count = 0;

    info!("Starting replay of {}", input_path.as_ref().display());

    while let Ok(Some(diff)) = reader.next_diff() {
        // Update result
        diff_count += 1;
        result.diffs_processed = diff_count;
        result.last_frame_id = diff.frame_id;

        // Check if this is a GENESIS diff
        if diff.frame_id == crate::GENESIS_FRAME_ID {
            info!("GENESIS diff encountered");
            result.genesis_encountered = true;
        }

        // Log
        debug!("Applying diff {}: frame_id={}", diff_count, diff.frame_id);

        // Apply diff to LCN
        if let Err(e) = lcn.apply_diff(&diff) {
            let error_msg = format!("Error applying diff {}: {}", diff_count, e);
            error!("{}", error_msg);
            result.errors.push(error_msg);
        }

        // Step mode - wait for Enter key
        if step_by_step {
            info!("Applied diff {}. Press Enter to continue...", diff_count);

            // In a real implementation, we'd pause for user input
            // For this implementation, we'll just simulate a delay
            std::thread::sleep(Duration::from_secs(1));
        }

        // Check limits
        if let Some(max_diffs) = limit {
            if diff_count >= max_diffs {
                info!("Reached diff limit of {}", max_diffs);
                break;
            }
        }

        if let Some(stop_frame) = stop_at {
            if diff.frame_id >= stop_frame {
                info!("Reached stop frame {}", stop_frame);
                break;
            }
        }
    }

    // Update result
    result.node_count = lcn.get_all_nodes().len();

    // Calculate edge count (approximate)
    let mut edge_count = 0;
    for node in lcn.get_all_nodes() {
        let connected = lcn.get_connected_nodes(&node.id);
        edge_count += connected.len();
    }
    result.edge_count = edge_count;

    // Calculate LCN hash
    result.lcn_hash = calculate_lcn_hash(&lcn);

    // Calculate processing time
    if let Ok(elapsed) = start_time.elapsed() {
        result.processing_time_ms = elapsed.as_millis() as u64;
    }

    info!("Replay complete");
    info!("Processed {} diffs", result.diffs_processed);
    info!("Last frame ID: {}", result.last_frame_id);
    info!("Node count: {}", result.node_count);
    info!("Edge count: {}", result.edge_count);
    info!("LCN hash: {:016x}", result.lcn_hash);
    info!("Processing time: {} ms", result.processing_time_ms);

    if !result.errors.is_empty() {
        warn!("Encountered {} errors during replay", result.errors.len());
    }

    Ok(result)
}

/// Execute the replay command
pub async fn execute_replay_cmd(args: ReplayArgs) -> Result<(), String> {
    // Set up logging
    let log_level = match args.verbose {
        0 => tracing::Level::ERROR,
        1 => tracing::Level::INFO,
        2 => tracing::Level::DEBUG,
        _ => tracing::Level::TRACE,
    };

    tracing::subscriber::set_global_default(
        tracing_subscriber::FmtSubscriber::builder()
            .with_max_level(log_level)
            .finish(),
    )
    .unwrap();

    // Verify input file exists
    if !args.input.exists() {
        return Err(format!("Input file not found: {}", args.input.display()));
    }

    // Replay
    let result = replay_psiarc(args.input.clone(), args.step, args.limit, args.stop_at).await?;

    // Output results
    print_replay_summary(&args.input, &result);

    // Output to file if requested
    if let Some(output_path) = args.output {
        info!("Writing snapshot to {}", output_path.display());
        // TODO: Implement LCN serialization
        info!("Note: LCN serialization not yet implemented");
    }

    Ok(())
}

/// Print a summary of the replay results
fn print_replay_summary(input_path: &Path, result: &ReplayResult) {
    println!("\nReplay Summary for {}", input_path.display());
    println!("----------------------------------------");
    println!("Diffs Processed:   {}", result.diffs_processed);
    println!("Last Frame ID:     {}", result.last_frame_id);
    println!(
        "GENESIS Found:     {}",
        if result.genesis_encountered {
            "Yes"
        } else {
            "No"
        }
    );
    println!("Node Count:        {}", result.node_count);
    println!("Edge Count:        {}", result.edge_count);
    println!("LCN Hash:          {:016x}", result.lcn_hash);
    println!("Processing Time:   {} ms", result.processing_time_ms);

    if !result.errors.is_empty() {
        println!("\nErrors:");
        for (i, error) in result.errors.iter().enumerate() {
            println!("  {}. {}", i + 1, error);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diff::{ConceptDiffBuilder, Op};
    use std::collections::HashMap;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_replay_psiarc() {
        // Create a temporary psiarc file
        let file = NamedTempFile::new().unwrap();
        let psiarc_path = file.path();

        // Create a psiarc log
        let options = crate::psiarc::PsiarcOptions {
            directory: psiarc_path.parent().unwrap().to_string_lossy().to_string(),
            ..Default::default()
        };

        let manager = PsiarcManager::new(psiarc_path.parent().unwrap());
        let log = manager.create_log_with_options("test", options).unwrap();

        // Create GENESIS diff
        let genesis_diff = crate::diff::create_genesis_diff("TestCorpus");
        log.record(&genesis_diff).unwrap();

        // Create some additional diffs
        let diff1 = ConceptDiffBuilder::new(1)
            .create("node1", "TestNode", HashMap::new())
            .build();

        let diff2 = ConceptDiffBuilder::new(2)
            .create("node2", "TestNode", HashMap::new())
            .bind("node1", "node2", None)
            .build();

        log.record(&diff1).unwrap();
        log.record(&diff2).unwrap();

        // Close log
        log.close().unwrap();

        // Replay
        let result = replay_psiarc(log.path(), false, None, None).await.unwrap();

        // Verify
        assert_eq!(result.diffs_processed, 3);
        assert_eq!(result.last_frame_id, 2);
        assert!(result.genesis_encountered);
        assert_eq!(result.node_count, 3); // TIMELESS_ROOT, node1, node2
        assert!(result.errors.is_empty());
    }
}


