//! ÃË†diff-viewer Tool
//!
//! This tool provides visualization and debugging capabilities for
//! ÃË†arc (psiarc) logs, allowing users to inspect ConceptDiff transactions,
//! view concept activations over time, and trace diff lineage.

use crate::diff::{ConceptDiff, ConceptDiffRef, Op};
use crate::lcn::LargeConceptNetwork;
use crate::psiarc::{PsiarcManager, PsiArcReader};

use clap::{Parser, Subcommand};
use comfy_table::{Cell, Color, ContentArrangement, Row, Table};
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc::{self, Receiver, Sender};
use tracing::{debug, error, info, warn};

/// Command line arguments for the ÃË†diff-viewer
#[derive(Parser, Debug)]
#[clap(
    name = "psidiff-viewer",
    about = "Visualize ConceptDiff transactions from ÃË†arc logs"
)]
struct Args {
    /// Subcommand to execute
    #[clap(subcommand)]
    command: Command,
}

/// Subcommands for the ÃË†diff-viewer
#[derive(Subcommand, Debug)]
enum Command {
    /// View a ÃË†arc log file
    View {
        /// Path to the ÃË†arc log file
        #[clap(name = "FILE")]
        file: PathBuf,

        /// Whether to follow the log file for new diffs
        #[clap(short, long)]
        follow: bool,

        /// Show concept activations over time
        #[clap(short, long)]
        activations: bool,

        /// Filter diffs by source
        #[clap(long)]
        source: Option<String>,

        /// Filter diffs by target
        #[clap(long)]
        target: Option<String>,

        /// Filter diffs by frame ID
        #[clap(long)]
        frame: Option<u64>,

        /// Filter diffs by operation type
        #[clap(long)]
        op: Option<String>,
    },

    /// List all ÃË†arc log files in the directory
    List {
        /// Directory containing ÃË†arc logs
        #[clap(name = "DIR", default_value = "logs")]
        dir: PathBuf,
    },

    /// Replay a ÃË†arc log to a fresh LCN
    Replay {
        /// Path to the ÃË†arc log file
        #[clap(name = "FILE")]
        file: PathBuf,

        /// Whether to stop at a specific frame
        #[clap(long)]
        stop_at: Option<u64>,

        /// Output directory for replay state snapshots
        #[clap(long)]
        snapshot_dir: Option<PathBuf>,
    },

    /// Live view of ConceptDiff events
    Live {
        /// Whether to show concept activations
        #[clap(short, long)]
        activations: bool,

        /// Filter events by source
        #[clap(long)]
        source: Option<String>,

        /// Filter events by target
        #[clap(long)]
        target: Option<String>,
    },
}

/// Formats a ConceptDiff for display
fn format_diff(diff: &ConceptDiff) -> String {
    let mut table = Table::new();
    table.set_content_arrangement(ContentArrangement::Dynamic);

    // Add header row
    table.set_header(vec!["Frame", "Source", "Timestamp", "Op Count", "Targets"]);

    // Add data row
    let frame_id = diff.frame_id.to_string();
    let source = diff.source.clone().unwrap_or_else(|| "-".to_string());

    // Format timestamp
    let timestamp = if diff.timestamp_ms > 0 {
        let dt = SystemTime::UNIX_EPOCH + Duration::from_millis(diff.timestamp_ms);
        format!("{:?}", dt)
    } else {
        "-".to_string()
    };

    let op_count = diff.ops.len().to_string();
    let targets = diff
        .targets
        .clone()
        .map(|t| t.join(", "))
        .unwrap_or_else(|| "-".to_string());

    table.add_row(vec![
        Cell::new(frame_id).fg(Color::Green),
        Cell::new(source),
        Cell::new(timestamp),
        Cell::new(op_count),
        Cell::new(targets),
    ]);

    // Add operations table
    let mut ops_table = Table::new();
    ops_table.set_content_arrangement(ContentArrangement::Dynamic);
    ops_table.set_header(vec!["#", "Operation", "Details"]);

    for (i, op) in diff.ops.iter().enumerate() {
        let op_str = match op {
            Op::Create {
                node, node_type, ..
            } => {
                format!("Create {}:{}", node_type, node)
            }
            Op::Delete { node } => {
                format!("Delete {}", node)
            }
            Op::Update { node, .. } => {
                format!("Update {}", node)
            }
            Op::Link {
                source,
                target,
                rel_type,
                ..
            } => {
                format!("Link {}--[{}]-->{}", source, rel_type, target)
            }
            Op::Unlink {
                source,
                target,
                rel_type,
            } => {
                if let Some(rel_type) = rel_type {
                    format!("Unlink {}--[{}]-->{}", source, rel_type, target)
                } else {
                    format!("Unlink {}-/->{}", source, target)
                }
            }
            Op::Bind {
                parent,
                node,
                bind_type,
            } => {
                format!("Bind {}--[{}]-->{}", parent, bind_type, node)
            }
            Op::Signal { event, target, .. } => {
                if let Some(target) = target {
                    format!("Signal {} to {}", event, target)
                } else {
                    format!("Signal {}", event)
                }
            }
            Op::Execute { operation, .. } => {
                format!("Execute {}", operation)
            }
        };

        ops_table.add_row(vec![Cell::new(i + 1), Cell::new(op_str), Cell::new("...")]);
    }

    format!("{}\n\n{}", table, ops_table)
}

/// Visualizes concept activations over time
fn visualize_activations(diffs: &[ConceptDiff]) -> String {
    // Extract all unique concepts
    let mut concepts = HashSet::new();

    for diff in diffs {
        for op in &diff.ops {
            match op {
                Op::Create {
                    node, node_type, ..
                } if node_type == "Concept" => {
                    concepts.insert(node.clone());
                }
                Op::Link {
                    source,
                    target,
                    rel_type,
                    ..
                } if rel_type == "HAS_CONCEPT" => {
                    concepts.insert(target.clone());
                }
                _ => {}
            }
        }
    }

    // Create activation table
    let mut table = Table::new();
    table.set_content_arrangement(ContentArrangement::Dynamic);

    // Add header row with frame IDs
    let mut header = vec!["Concept"];
    for diff in diffs {
        header.push(&format!("Frame {}", diff.frame_id));
    }
    table.set_header(header);

    // Add rows for each concept
    for concept in concepts {
        let mut row = vec![Cell::new(concept.clone())];

        for diff in diffs {
            let mut active = false;

            for op in &diff.ops {
                match op {
                    Op::Create {
                        node, node_type, ..
                    } if node_type == "Concept" && node == &concept => {
                        active = true;
                    }
                    Op::Link {
                        target, rel_type, ..
                    } if rel_type == "HAS_CONCEPT" && target == &concept => {
                        active = true;
                    }
                    _ => {}
                }
            }

            if active {
                row.push(Cell::new("Ã¢Å“â€œ").fg(Color::Green));
            } else {
                row.push(Cell::new(" "));
            }
        }

        table.add_row(row);
    }

    format!("\nConcept Activations:\n{}", table)
}

/// Traces the lineage of a ConceptDiff
fn trace_lineage(diff: &ConceptDiff, diffs: &[ConceptDiff]) -> String {
    let mut table = Table::new();
    table.set_content_arrangement(ContentArrangement::Dynamic);

    // Add header row
    table.set_header(vec!["Frame", "Relationship", "Source", "Operation"]);

    // Extract node IDs from current diff
    let mut nodes = HashSet::new();
    for op in &diff.ops {
        match op {
            Op::Create { node, .. } | Op::Delete { node } | Op::Update { node, .. } => {
                nodes.insert(node.clone());
            }
            Op::Link { source, target, .. } | Op::Unlink { source, target, .. } => {
                nodes.insert(source.clone());
                nodes.insert(target.clone());
            }
            Op::Bind { parent, node, .. } => {
                nodes.insert(parent.clone());
                nodes.insert(node.clone());
            }
            _ => {}
        }
    }

    // Find related diffs
    for other_diff in diffs {
        if other_diff.frame_id == diff.frame_id {
            continue;
        }

        let mut related = false;
        let mut relationship = "Unknown";
        let mut related_node = String::new();
        let mut operation = String::new();

        for op in &other_diff.ops {
            match op {
                Op::Create { node, .. } | Op::Delete { node } | Op::Update { node, .. } => {
                    if nodes.contains(node) {
                        related = true;
                        relationship = "Node Reference";
                        related_node = node.clone();
                        operation = format!("{:?}", op);
                        break;
                    }
                }
                Op::Link { source, target, .. } | Op::Unlink { source, target, .. } => {
                    if nodes.contains(source) || nodes.contains(target) {
                        related = true;
                        relationship = "Edge Reference";
                        related_node = if nodes.contains(source) {
                            source.clone()
                        } else {
                            target.clone()
                        };
                        operation = format!("{:?}", op);
                        break;
                    }
                }
                Op::Bind { parent, node, .. } => {
                    if nodes.contains(parent) || nodes.contains(node) {
                        related = true;
                        relationship = "Binding Reference";
                        related_node = if nodes.contains(parent) {
                            parent.clone()
                        } else {
                            node.clone()
                        };
                        operation = format!("{:?}", op);
                        break;
                    }
                }
                _ => {}
            }
        }

        if related {
            table.add_row(vec![
                Cell::new(other_diff.frame_id),
                Cell::new(relationship),
                Cell::new(related_node),
                Cell::new(&operation[..operation.len().min(40)]),
            ]);
        }
    }

    format!("\nDiff Lineage:\n{}", table)
}

/// Handles the 'view' command
async fn handle_view_command(
    file: PathBuf,
    follow: bool,
    activations: bool,
    source: Option<String>,
    target: Option<String>,
    frame: Option<u64>,
    op: Option<String>,
) -> Result<(), String> {
    // Open psiarc file
    let mut reader =
        PsiArcReader::open(&file).map_err(|e| format!("Failed to open file: {}", e))?;

    // Read all diffs
    let mut diffs = Vec::new();

    while let Ok(Some(diff)) = reader.next_diff() {
        // Apply filters
        if let Some(ref source_filter) = source {
            if diff.source.as_ref().map_or(true, |s| s != source_filter) {
                continue;
            }
        }

        if let Some(ref target_filter) = target {
            if diff
                .targets
                .as_ref()
                .map_or(true, |t| !t.contains(target_filter))
            {
                continue;
            }
        }

        if let Some(frame_filter) = frame {
            if diff.frame_id != frame_filter {
                continue;
            }
        }

        if let Some(ref op_filter) = op {
            let has_matching_op = diff.ops.iter().any(|op| match op {
                Op::Create { .. } => op_filter == "create",
                Op::Delete { .. } => op_filter == "delete",
                Op::Update { .. } => op_filter == "update",
                Op::Link { .. } => op_filter == "link",
                Op::Unlink { .. } => op_filter == "unlink",
                Op::Bind { .. } => op_filter == "bind",
                Op::Signal { .. } => op_filter == "signal",
                Op::Execute { .. } => op_filter == "execute",
            });

            if !has_matching_op {
                continue;
            }
        }

        diffs.push(diff);
    }

    // Display diffs
    for diff in &diffs {
        println!("\n{}", format_diff(diff));
    }

    // Display activations if requested
    if activations {
        println!("{}", visualize_activations(&diffs));
    }

    // Show summary
    println!("\nSummary: {} diffs displayed", diffs.len());

    // Follow mode
    if follow {
        println!("\nFollowing log file for new diffs (Ctrl+C to stop)...");

        // TODO: Implement follow mode
        // For Day 1, we'll just return for now
    }

    Ok(())
}

/// Handles the 'list' command
async fn handle_list_command(dir: PathBuf) -> Result<(), String> {
    let manager = PsiarcManager::new(&dir);

    let logs = manager
        .list_logs()
        .map_err(|e| format!("Failed to list logs: {}", e))?;

    let mut table = Table::new();
    table.set_content_arrangement(ContentArrangement::Dynamic);
    table.set_header(vec!["File", "Size", "Creation Time"]);

    for log_name in logs {
        let path = dir.join(&log_name);

        let size = if let Ok(metadata) = std::fs::metadata(&path) {
            format_size(metadata.len())
        } else {
            "Unknown".to_string()
        };

        let created = if let Ok(metadata) = std::fs::metadata(&path) {
            if let Ok(created) = metadata.created() {
                format!("{:?}", created)
            } else {
                "Unknown".to_string()
            }
        } else {
            "Unknown".to_string()
        };

        table.add_row(vec![
            Cell::new(log_name),
            Cell::new(size),
            Cell::new(created),
        ]);
    }

    println!("{}", table);

    Ok(())
}

/// Handles the 'replay' command
async fn handle_replay_command(
    file: PathBuf,
    stop_at: Option<u64>,
    snapshot_dir: Option<PathBuf>,
) -> Result<(), String> {
    // Create LCN
    let lcn = Arc::new(LargeConceptNetwork::new());

    // Open psiarc file
    let mut reader =
        PsiArcReader::open(&file).map_err(|e| format!("Failed to open file: {}", e))?;

    // Apply diffs to LCN
    let mut frame_count = 0;
    let mut last_frame = 0;

    println!("Replaying ÃË†arc log...");

    while let Ok(Some(diff)) = reader.next_diff() {
        // Check if we should stop
        if let Some(stop_frame) = stop_at {
            if diff.frame_id > stop_frame {
                println!("Stopping at frame {}", stop_frame);
                break;
            }
        }

        // Apply diff to LCN
        if let Err(e) = lcn.apply_diff(&diff) {
            return Err(format!("Failed to apply diff: {}", e));
        }

        frame_count += 1;
        last_frame = diff.frame_id;

        // Create snapshot if requested
        if let Some(ref snapshot_dir) = snapshot_dir {
            // TODO: Implement snapshot functionality
            // For Day 1, we'll just print a message
            if frame_count % 100 == 0 {
                println!("Snapshot would be created at frame {}", last_frame);
            }
        }
    }

    // Display summary
    println!("\nReplay complete");
    println!("Applied {} diffs", frame_count);
    println!("Last frame: {}", last_frame);
    println!(
        "LCN status: GENESIS complete = {}",
        lcn.is_genesis_complete()
    );

    Ok(())
}

/// Handles the 'live' command
async fn handle_live_command(
    activations: bool,
    source: Option<String>,
    target: Option<String>,
) -> Result<(), String> {
    println!("Live view not implemented yet");

    // TODO: Implement live view functionality
    // For Day 1, we'll just return

    Ok(())
}

/// Main function
async fn main() -> Result<(), String> {
    // Parse command line arguments
    let args = Args::parse();

    // Execute appropriate command
    match args.command {
        Command::View {
            file,
            follow,
            activations,
            source,
            target,
            frame,
            op,
        } => handle_view_command(file, follow, activations, source, target, frame, op).await,
        Command::List { dir } => handle_list_command(dir).await,
        Command::Replay {
            file,
            stop_at,
            snapshot_dir,
        } => handle_replay_command(file, stop_at, snapshot_dir).await,
        Command::Live {
            activations,
            source,
            target,
        } => handle_live_command(activations, source, target).await,
    }
}

/// Format file size in a human-readable way
fn format_size(size: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if size < KB {
        format!("{} B", size)
    } else if size < MB {
        format!("{:.2} KB", size as f64 / KB as f64)
    } else if size < GB {
        format!("{:.2} MB", size as f64 / MB as f64)
    } else {
        format!("{:.2} GB", size as f64 / GB as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diff::{ConceptDiffBuilder, Op};
    use std::collections::HashMap;

    #[test]
    fn test_format_diff() {
        let mut diff = ConceptDiff::new(1);
        diff.source = Some("TestSource".to_string());
        diff.ops.push(Op::Create {
            node: "node1".to_string(),
            node_type: "TestNode".to_string(),
            properties: HashMap::new(),
        });

        let result = format_diff(&diff);
        assert!(result.contains("Frame"));
        assert!(result.contains("Source"));
        assert!(result.contains("TestSource"));
        assert!(result.contains("Create"));
    }

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(500), "500 B");
        assert_eq!(format_size(1500), "1.46 KB");
        assert_eq!(format_size(1500000), "1.43 MB");
        assert_eq!(format_size(1500000000), "1.40 GB");
    }
}

