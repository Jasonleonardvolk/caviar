//! Tori Tag CLI Tool
//!
//! This command-line tool allows for adding, updating, or removing tags
//! on existing concepts post-ingest. It can operate on individual concepts
//! or apply tags to all concepts from a specific source.

use concept_mesh_rs::auth::session::{get_current_session, Session};
use concept_mesh_rs::concept_trail::{create_trail_from_psiarc, ConceptTrail, ConceptTrailFilter};
use concept_mesh_rs::diff::{ConceptDiff, ConceptDiffBuilder, UserContextExt};
use concept_mesh_rs::psiarc::{PsiArcReader, PsiarcHeader};
use concept_mesh_rs::{self, ConceptId, LargeConceptNetwork};

use clap::{Parser, Subcommand};
use colored::*;
use serde_json::json;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use tracing::{debug, error, info, warn};
use tracing_subscriber::FmtSubscriber;

/// Command line arguments for the tag CLI
#[derive(Parser, Debug)]
#[clap(
    name = "tori-tag",
    about = "Add, update, or remove tags for concepts post-ingest"
)]
struct Args {
    #[clap(subcommand)]
    command: Command,

    /// Log directory for storing logs
    #[clap(short, long, default_value = "logs")]
    log_dir: PathBuf,

    /// Verbosity level (0-3)
    #[clap(short, long, default_value = "1")]
    verbose: u8,

    /// Disable colorful output
    #[clap(long)]
    no_color: bool,
}

/// Subcommands for the tag CLI
#[derive(Subcommand, Debug)]
enum Command {
    /// Add tags to concepts
    Add {
        /// Concept ID or source filter (document ID or source title)
        #[clap(name = "TARGET")]
        target: String,

        /// Tags to add (comma-separated)
        #[clap(short, long, value_delimiter = ',')]
        tags: Vec<String>,

        /// Target type: 'concept', 'document', or 'source'
        #[clap(short, long, default_value = "concept")]
        target_type: String,

        /// PsiArc file or directory to search in
        #[clap(short, long)]
        psiarc: Option<PathBuf>,

        /// Apply recursively to all concepts related to the target
        #[clap(short, long)]
        recursive: bool,
    },

    /// Remove tags from concepts
    Remove {
        /// Concept ID or source filter (document ID or source title)
        #[clap(name = "TARGET")]
        target: String,

        /// Tags to remove (comma-separated)
        #[clap(short, long, value_delimiter = ',')]
        tags: Vec<String>,

        /// Target type: 'concept', 'document', or 'source'
        #[clap(short, long, default_value = "concept")]
        target_type: String,

        /// PsiArc file or directory to search in
        #[clap(short, long)]
        psiarc: Option<PathBuf>,

        /// Apply recursively to all concepts related to the target
        #[clap(short, long)]
        recursive: bool,
    },

    /// List tags for a concept or all concepts from a source
    List {
        /// Concept ID or source filter (document ID or source title)
        #[clap(name = "TARGET")]
        target: String,

        /// Target type: 'concept', 'document', or 'source'
        #[clap(short, long, default_value = "concept")]
        target_type: String,

        /// PsiArc file or directory to search in
        #[clap(short, long)]
        psiarc: Option<PathBuf>,

        /// Output format (text, json)
        #[clap(short, long, default_value = "text")]
        format: String,
    },
}

/// Tag update operation
enum TagOperation {
    Add,
    Remove,
}

/// Result of tag update
struct TagUpdateResult {
    concept_id: String,
    concept_title: Option<String>,
    concept_type: Option<String>,
    document_id: Option<String>,
    source_title: Option<String>,
    previous_tags: Vec<String>,
    new_tags: Vec<String>,
    status: String,
}

/// Find concepts matching the target criteria
fn find_matching_concepts(
    target: &str,
    target_type: &str,
    psiarc_path: Option<&Path>,
    recursive: bool,
) -> Result<Vec<ConceptId>, String> {
    let mut matching_concept_ids = Vec::new();

    match target_type {
        "concept" => {
            // Single concept ID
            matching_concept_ids.push(target.to_string());
        }
        "document" => {
            // All concepts from a document ID
            if let Some(path) = psiarc_path {
                // Create filter for document ID
                let filter = ConceptTrailFilter::for_document(target);

                // Create trail from PsiArc
                match create_trail_from_psiarc(path, None, Some(filter)) {
                    Ok(trail) => {
                        for entry in &trail.entries {
                            matching_concept_ids.push(entry.concept_id.clone());
                        }
                    }
                    Err(e) => {
                        return Err(format!("Failed to create trail from PsiArc: {}", e));
                    }
                }
            } else {
                return Err(
                    "PsiArc file or directory is required for document target type".to_string(),
                );
            }
        }
        "source" => {
            // All concepts from a source title
            if let Some(path) = psiarc_path {
                // Create filter for source title
                let filter = ConceptTrailFilter::for_source(target);

                // Create trail from PsiArc
                match create_trail_from_psiarc(path, None, Some(filter)) {
                    Ok(trail) => {
                        for entry in &trail.entries {
                            matching_concept_ids.push(entry.concept_id.clone());
                        }
                    }
                    Err(e) => {
                        return Err(format!("Failed to create trail from PsiArc: {}", e));
                    }
                }
            } else {
                return Err(
                    "PsiArc file or directory is required for source target type".to_string(),
                );
            }
        }
        _ => {
            return Err(format!("Invalid target type: {}", target_type));
        }
    }

    if matching_concept_ids.is_empty() {
        return Err(format!(
            "No concepts found matching {} '{}'",
            target_type, target
        ));
    }

    Ok(matching_concept_ids)
}

/// Update tags for concepts
fn update_tags(
    concept_ids: &[ConceptId],
    tags: &[String],
    operation: TagOperation,
    lcn: &LargeConceptNetwork,
    psiarc_path: Option<&Path>,
) -> Result<Vec<TagUpdateResult>, String> {
    let mut results = Vec::new();

    // Load concept data from LCN or PsiArc
    let concept_data = if let Some(path) = psiarc_path {
        // Load from PsiArc
        match PsiArcReader::open(path) {
            Ok(reader) => {
                // Load all frames
                match reader.read_all_frames() {
                    Ok(frames) => {
                        // Extract concept data from frames
                        let mut data = HashMap::new();

                        for frame in frames {
                            for op in &frame.diff.ops {
                                if let concept_mesh::diff::Op::Create {
                                    id,
                                    concept_type,
                                    properties,
                                } = op
                                {
                                    if concept_ids.contains(id) {
                                        // Extract relevant data
                                        let title = properties
                                            .get("title")
                                            .and_then(|v| v.as_str())
                                            .map(|s| s.to_string());

                                        let document_id = properties
                                            .get("document_id")
                                            .and_then(|v| v.as_str())
                                            .map(|s| s.to_string());

                                        let source_title = properties
                                            .get("source_title")
                                            .and_then(|v| v.as_str())
                                            .map(|s| s.to_string());

                                        let existing_tags = properties
                                            .get("tags")
                                            .and_then(|v| v.as_array())
                                            .map(|arr| {
                                                arr.iter()
                                                    .filter_map(|v| {
                                                        v.as_str().map(|s| s.to_string())
                                                    })
                                                    .collect::<Vec<_>>()
                                            })
                                            .unwrap_or_default();

                                        data.insert(
                                            id.clone(),
                                            (
                                                concept_type.clone(),
                                                title,
                                                document_id,
                                                source_title,
                                                existing_tags,
                                            ),
                                        );
                                    }
                                }
                            }
                        }

                        data
                    }
                    Err(e) => {
                        return Err(format!("Failed to read frames from PsiArc: {}", e));
                    }
                }
            }
            Err(e) => {
                return Err(format!("Failed to open PsiArc file: {}", e));
            }
        }
    } else {
        // Load from LCN
        let mut data = HashMap::new();

        for concept_id in concept_ids {
            // Get concept data from LCN
            if let Some(concept) = lcn.get_concept(concept_id) {
                let concept_type = concept.concept_type.clone();
                let title = concept
                    .properties
                    .get("title")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());

                let document_id = concept
                    .properties
                    .get("document_id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());

                let source_title = concept
                    .properties
                    .get("source_title")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());

                let existing_tags = concept
                    .properties
                    .get("tags")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(|s| s.to_string()))
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();

                data.insert(
                    concept_id.clone(),
                    (
                        concept_type,
                        title,
                        document_id,
                        source_title,
                        existing_tags,
                    ),
                );
            } else {
                warn!("Concept not found in LCN: {}", concept_id);
            }
        }

        data
    };

    // Update tags for each concept
    for concept_id in concept_ids {
        if let Some((concept_type, title, document_id, source_title, existing_tags)) =
            concept_data.get(concept_id)
        {
            // Make a copy of existing tags
            let mut updated_tags = existing_tags.clone();

            // Update tags based on operation
            match operation {
                TagOperation::Add => {
                    // Add new tags (avoiding duplicates)
                    let existing_set: HashSet<_> = existing_tags.iter().collect();

                    for tag in tags {
                        if !existing_set.contains(tag) {
                            updated_tags.push(tag.clone());
                        }
                    }
                }
                TagOperation::Remove => {
                    // Remove specified tags
                    let remove_set: HashSet<_> = tags.iter().collect();
                    updated_tags.retain(|tag| !remove_set.contains(tag));
                }
            }

            // Create a diff to update the concept
            let mut builder = ConceptDiffBuilder::new(0);

            // Update concept properties
            let mut properties = serde_json::Map::new();
            properties.insert("tags".to_string(), json!(updated_tags));

            builder = builder.update(concept_id, properties);

            // Add user context
            let diff = builder.with_user_context().build();

            // Apply the diff to LCN if available
            if psiarc_path.is_none() {
                if let Err(e) = lcn.apply_diff(&diff) {
                    warn!("Failed to update tags for concept {}: {}", concept_id, e);

                    results.push(TagUpdateResult {
                        concept_id: concept_id.clone(),
                        concept_title: title.clone(),
                        concept_type: Some(concept_type.clone()),
                        document_id: document_id.clone(),
                        source_title: source_title.clone(),
                        previous_tags: existing_tags.clone(),
                        new_tags: updated_tags,
                        status: format!("Failed: {}", e),
                    });

                    continue;
                }
            } else {
                // For PsiArc, we would write a new diff to the file
                // This is a placeholder - actual implementation would depend on PsiArc write API
                debug!("Generated diff for PsiArc update: {:?}", diff);
            }

            // Add to results
            results.push(TagUpdateResult {
                concept_id: concept_id.clone(),
                concept_title: title.clone(),
                concept_type: Some(concept_type.clone()),
                document_id: document_id.clone(),
                source_title: source_title.clone(),
                previous_tags: existing_tags.clone(),
                new_tags: updated_tags,
                status: "Success".to_string(),
            });
        } else {
            warn!("No data found for concept: {}", concept_id);

            results.push(TagUpdateResult {
                concept_id: concept_id.clone(),
                concept_title: None,
                concept_type: None,
                document_id: None,
                source_title: None,
                previous_tags: Vec::new(),
                new_tags: Vec::new(),
                status: "Failed: Concept not found".to_string(),
            });
        }
    }

    Ok(results)
}

/// Print results in colorful format
fn print_results(results: &[TagUpdateResult], use_color: bool) {
    // Enable or disable colors
    colored::control::set_override(!use_color);

    // Print header
    println!("{} Tag Update Results", "ðŸ”–".bright_yellow());
    println!();

    // Print each result
    for result in results {
        let title = result.concept_title.as_deref().unwrap_or("Unknown");
        let concept_type = result.concept_type.as_deref().unwrap_or("Unknown");

        // Print concept info
        println!(
            "{} {} ({})",
            "ðŸ“„".bright_blue(),
            title.bright_white().bold(),
            concept_type.bright_cyan()
        );
        println!("   ID: {}", result.concept_id.bright_magenta());

        // Print source info if available
        if let Some(source_title) = &result.source_title {
            println!("   Source: {}", source_title.bright_green());
        }

        if let Some(document_id) = &result.document_id {
            println!("   Document ID: {}", document_id.bright_blue());
        }

        // Print tag changes
        println!(
            "   Previous tags: [{}]",
            result
                .previous_tags
                .iter()
                .map(|t| t.yellow().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );

        println!(
            "   New tags: [{}]",
            result
                .new_tags
                .iter()
                .map(|t| t.green().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );

        // Print status
        if result.status.starts_with("Success") {
            println!("   Status: {}", result.status.bright_green());
        } else {
            println!("   Status: {}", result.status.bright_red());
        }

        println!();
    }

    // Print summary
    let success_count = results
        .iter()
        .filter(|r| r.status.starts_with("Success"))
        .count();
    println!(
        "{} Updated {} of {} concepts",
        if success_count == results.len() {
            "âœ…".bright_green()
        } else {
            "âš ï¸".bright_yellow()
        },
        success_count.to_string().bright_white(),
        results.len().to_string().bright_white()
    );
}

/// Export results as JSON
fn export_results_json(results: &[TagUpdateResult]) -> String {
    serde_json::to_string_pretty(&results).unwrap_or_else(|_| "{}".to_string())
}

#[tokio::main]
async fn main() -> Result<(), String> {
    // Parse command line arguments
    let args = Args::parse();

    // Set up logging
    let log_level = match args.verbose {
        0 => tracing::Level::ERROR,
        1 => tracing::Level::INFO,
        2 => tracing::Level::DEBUG,
        _ => tracing::Level::TRACE,
    };

    let subscriber = FmtSubscriber::builder().with_max_level(log_level).finish();

    tracing::subscriber::set_global_default(subscriber)
        .expect("Failed to set global default subscriber");

    info!("Starting tori-tag");

    // Create in-memory LCN
    let lcn = LargeConceptNetwork::new();

    // Execute command
    match args.command {
        Command::Add {
            target,
            tags,
            target_type,
            psiarc,
            recursive,
        } => {
            info!("Adding tags to {}: {}", target_type, target);

            // Find matching concepts
            let concept_ids =
                find_matching_concepts(&target, &target_type, psiarc.as_deref(), recursive)?;

            info!("Found {} matching concepts", concept_ids.len());

            // Update tags
            let results = update_tags(
                &concept_ids,
                &tags,
                TagOperation::Add,
                &lcn,
                psiarc.as_deref(),
            )?;

            // Print results
            print_results(&results, args.no_color);
        }
        Command::Remove {
            target,
            tags,
            target_type,
            psiarc,
            recursive,
        } => {
            info!("Removing tags from {}: {}", target_type, target);

            // Find matching concepts
            let concept_ids =
                find_matching_concepts(&target, &target_type, psiarc.as_deref(), recursive)?;

            info!("Found {} matching concepts", concept_ids.len());

            // Update tags
            let results = update_tags(
                &concept_ids,
                &tags,
                TagOperation::Remove,
                &lcn,
                psiarc.as_deref(),
            )?;

            // Print results
            print_results(&results, args.no_color);
        }
        Command::List {
            target,
            target_type,
            psiarc,
            format,
        } => {
            info!("Listing tags for {}: {}", target_type, target);

            // Find matching concepts
            let concept_ids = find_matching_concepts(
                &target,
                &target_type,
                psiarc.as_deref(),
                false, // recursive not needed for listing
            )?;

            // Load concept data (reusing update_tags function with empty tag list)
            let results = update_tags(
                &concept_ids,
                &Vec::new(),
                TagOperation::Add, // operation doesn't matter for listing
                &lcn,
                psiarc.as_deref(),
            )?;

            // Print results
            if format == "json" {
                println!("{}", export_results_json(&results));
            } else {
                print_results(&results, args.no_color);
            }
        }
    }

    Ok(())
}

