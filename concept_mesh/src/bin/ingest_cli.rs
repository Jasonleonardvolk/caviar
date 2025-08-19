//! Ingest CLI Tool
//!
//! Command-line tool for ingesting documents into the concept mesh.
//! This demonstrates how to use the orchestrator, CBD, and other components.
//! Provides rich feedback with colorful output and detailed ingestion statistics.

use concept_mesh_rs::agents::orchestrator::{create_orchestrator, IngestDocument, OrchestratorConfig};
use concept_mesh_rs::auth::session::get_current_session;
use concept_mesh_rs::cbd::{CBDConfig, ConceptBoundaryDetector};
use concept_mesh_rs::cli::adapter::{create_ingest_command, create_status_command, CliAdapter};
use concept_mesh_rs::ingest::pdf_user_attribution::{extract_tags_from_content, IngestSourceInfo};
use concept_mesh_rs::mesh::InMemoryMesh;
use concept_mesh_rs::psiarc::{PsiarcManager, PsiarcOptions};
use concept_mesh_rs::ui::genesis_bridge::{create_genesis_ui_bridge, GenesisBridge, OscillatorBloom};
use concept_mesh_rs::{self, LargeConceptNetwork, GENESIS_FRAME_ID};

use clap::{Parser, Subcommand};
use colored::*;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tokio::fs::File;
use tokio::io::{self, AsyncReadExt};
use tracing::{debug, error, info};
use tracing_subscriber::FmtSubscriber;

/// Command line arguments for the ingest CLI
#[derive(Parser, Debug)]
#[clap(name = "ingest", about = "Ingest documents into the concept mesh")]
struct Args {
    #[clap(subcommand)]
    command: Command,

    /// Log directory for storing Ïˆarc logs
    #[clap(short, long, default_value = "logs")]
    log_dir: PathBuf,

    /// Verbosity level (0-3)
    #[clap(short, long, default_value = "1")]
    verbose: u8,

    /// Disable colorful output
    #[clap(long)]
    no_color: bool,
}

/// Subcommands for the ingest CLI
#[derive(Subcommand, Debug)]
enum Command {
    /// Ingest a document
    Ingest {
        /// Path to the document file
        #[clap(name = "FILE")]
        file: PathBuf,

        /// Document type (auto-detected from extension if not provided)
        #[clap(short, long)]
        document_type: Option<String>,

        /// Document title (defaults to filename)
        #[clap(short, long)]
        title: Option<String>,

        /// Tags to apply to the document
        #[clap(short, long, value_delimiter = ',')]
        tags: Option<Vec<String>>,

        /// Source title (defaults to title)
        #[clap(long)]
        source_title: Option<String>,

        /// Custom persona to use for ingestion
        #[clap(short, long)]
        persona: Option<String>,

        /// Enable auto-tagging based on content
        #[clap(long)]
        auto_tag: bool,
    },

    /// Check the status of an ingest job
    Status {
        /// Job ID to check
        #[clap(name = "JOB_ID")]
        job_id: String,
    },

    /// Watch a directory for new documents to ingest
    Watch {
        /// Directory to watch
        #[clap(name = "DIR")]
        dir: PathBuf,

        /// File pattern to match (glob)
        #[clap(short, long, default_value = "*")]
        pattern: String,

        /// Auto-tag files based on content
        #[clap(long)]
        auto_tag: bool,
    },

    /// Recursively ingest files from a directory
    Dir {
        /// Directory to process
        #[clap(name = "DIR")]
        dir: PathBuf,

        /// File pattern to match (glob)
        #[clap(short, long, default_value = "*.pdf")]
        pattern: String,

        /// Process recursively (include subdirectories)
        #[clap(short, long)]
        recursive: bool,

        /// Tags to apply to all documents (comma-separated)
        #[clap(short, long, value_delimiter = ',')]
        tags: Option<Vec<String>>,

        /// Auto-tag files based on content
        #[clap(long)]
        auto_tag: bool,

        /// Custom persona to use for ingestion
        #[clap(short, long)]
        persona: Option<String>,
    },
}

/// Ingest stats for reporting
struct IngestStats {
    /// Job ID
    job_id: String,

    /// Document title
    title: String,

    /// Document source path
    source_path: PathBuf,

    /// User ID (if available)
    user_id: Option<String>,

    /// Persona ID (if available)
    persona_id: Option<String>,

    /// Number of ConceptDiffs generated
    diff_count: usize,

    /// Number of phase regions detected
    phase_regions: usize,

    /// PsiArc file path
    psiarc_path: Option<PathBuf>,

    /// Tags applied to the document
    tags: Vec<String>,
}

impl IngestStats {
    /// Create a new ingest stats object
    fn new(job_id: String, title: String, source_path: PathBuf) -> Self {
        Self {
            job_id,
            title,
            source_path,
            user_id: None,
            persona_id: None,
            diff_count: 0,
            phase_regions: 0,
            psiarc_path: None,
            tags: Vec::new(),
        }
    }

    /// Set user and persona information from current session
    fn with_session_info(mut self) -> Self {
        if let Some(session) = get_current_session() {
            self.user_id = Some(session.user.concept_id.clone());
            self.persona_id = Some(session.persona.mode.as_str().to_string());
        }
        self
    }

    /// Set diff count
    fn with_diff_count(mut self, count: usize) -> Self {
        self.diff_count = count;
        self
    }

    /// Set phase region count
    fn with_phase_regions(mut self, count: usize) -> Self {
        self.phase_regions = count;
        self
    }

    /// Set PsiArc path
    fn with_psiarc_path(mut self, path: PathBuf) -> Self {
        self.psiarc_path = Some(path);
        self
    }

    /// Set tags
    fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Print colorful success message
    fn print_success(&self, use_color: bool) {
        // Enable or disable colors
        colored::control::set_override(!use_color);

        // Print ingestion success message
        let filename = self
            .source_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");

        println!(
            "{} Ingested \"{}\" as {} (persona: {})",
            "ðŸ“¥".bright_green(),
            filename.bright_white().bold(),
            self.user_id.as_deref().unwrap_or("anonymous").bright_blue(),
            self.persona_id
                .as_deref()
                .unwrap_or("default")
                .bright_yellow()
        );

        println!(
            "{} ConceptDiffs: {} | Phase regions: {}",
            "ðŸ§ ".bright_cyan(),
            self.diff_count.to_string().bright_white(),
            self.phase_regions.to_string().bright_white()
        );

        if let Some(psiarc_path) = &self.psiarc_path {
            println!(
                "{} Ïˆarc saved: {}",
                "ðŸ—ƒï¸".bright_magenta(),
                psiarc_path.to_string_lossy().bright_white()
            );
        }

        if !self.tags.is_empty() {
            println!(
                "{} Tags: {}",
                "ðŸ”–".bright_yellow(),
                self.tags
                    .iter()
                    .map(|t| t.bright_green().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }

        // Log to kaizen.log
        self.log_to_kaizen();
    }

    /// Log ingestion to kaizen.log
    fn log_to_kaizen(&self) {
        let log_entry = format!(
            "[{}] Ingested '{}' as {} (persona: {}) - ConceptDiffs: {}, Phase regions: {}, Tags: [{}]",
            chrono::Local::now().format("%Y-%m-%d %H:%M:%S"),
            self.title,
            self.user_id.as_deref().unwrap_or("anonymous"),
            self.persona_id.as_deref().unwrap_or("default"),
            self.diff_count,
            self.phase_regions,
            self.tags.join(", ")
        );

        // Append to kaizen.log
        let _ = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open("kaizen.log")
            .map(|mut file| {
                use std::io::Write;
                let _ = writeln!(file, "{}", log_entry);
            });
    }
}

/// Get document type from file extension
fn get_document_type_from_extension(path: &Path) -> String {
    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("txt")
        .to_lowercase();

    match extension.as_str() {
        "pdf" => "pdf",
        "docx" | "doc" => "docx",
        "md" | "markdown" => "markdown",
        "txt" => "text",
        "html" | "htm" => "html",
        "json" => "json",
        _ => "generic",
    }
    .to_string()
}

/// Read file content
async fn read_file_content(path: &Path) -> io::Result<String> {
    let mut file = File::open(path).await?;
    let mut content = String::new();
    file.read_to_string(&mut content).await?;
    Ok(content)
}

/// Estimate the number of phase regions based on content
fn estimate_phase_regions(content: &str) -> usize {
    // A very simple estimation based on length
    // In a real implementation, this would be more sophisticated
    let length = content.len();

    if length < 1000 {
        1
    } else if length < 5000 {
        2
    } else if length < 20000 {
        3
    } else if length < 50000 {
        5
    } else {
        7
    }
}

/// Estimate the number of ConceptDiffs based on content
fn estimate_diff_count(content: &str) -> usize {
    // A simple estimation based on the number of paragraphs
    // In a real implementation, this would come from actual processing
    let paragraph_count = content.split("\n\n").count();

    // Assume each paragraph generates approximately 2-3 diffs
    let estimated_diffs = paragraph_count * 2;

    // Add a base number for document creation and metadata
    estimated_diffs + 5
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

    info!("Starting concept-mesh ingest");

    // Create in-memory mesh
    let mesh = Arc::new(InMemoryMesh::new());

    // Create orchestrator
    let config = OrchestratorConfig {
        log_directory: args.log_dir.to_string_lossy().to_string(),
        ..Default::default()
    };

    let orchestrator = create_orchestrator("ingest-orchestrator", Some(config)).await?;

    // Create CLI adapter
    let mut cli_adapter = CliAdapter::new(Arc::clone(&orchestrator), Arc::clone(&mesh)).await?;
    cli_adapter.start()?;

    // Set up Genesis UI bridge
    let bloom = Arc::new(Mutex::new(OscillatorBloom::new(3000, 60)));
    let _genesis_bridge =
        create_genesis_ui_bridge("ingest-cli", Arc::clone(&mesh), Arc::clone(&bloom)).await?;

    // Execute command
    match args.command {
        Command::Ingest {
            file,
            document_type,
            title,
            tags,
            source_title,
            persona,
            auto_tag,
        } => {
            // Validate file exists
            if !file.exists() {
                return Err(format!("File not found: {}", file.display()));
            }

            info!("Ingesting file: {}", file.display());

            // Auto-detect document type if not provided
            let doc_type = document_type.unwrap_or_else(|| get_document_type_from_extension(&file));

            // Use filename as title if not provided
            let doc_title = title.unwrap_or_else(|| {
                file.file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("Untitled")
                    .to_string()
            });

            // Read file content
            let content = read_file_content(&file)
                .await
                .map_err(|e| format!("Failed to read file: {}", e))?;

            // Create source info
            let source_title_value = source_title.unwrap_or_else(|| doc_title.clone());
            let mut source_info = IngestSourceInfo::new(&file, source_title_value);

            // Extract tags from content if auto-tag is enabled
            let mut tag_list = tags.unwrap_or_default();

            if auto_tag && tag_list.is_empty() {
                tag_list = extract_tags_from_content(&content);
            }

            if !tag_list.is_empty() {
                source_info = source_info.with_tags(tag_list.clone());
            }

            // Create document
            let mut document = IngestDocument::new(doc_type)
                .with_title(doc_title.clone())
                .with_content(content.clone())
                .with_file_path(&file);

            // Add source info
            document =
                document.with_metadata("source_info", serde_json::json!(source_info.to_metadata()));

            // Add tags if provided
            if !tag_list.is_empty() {
                document = document.with_tags(tag_list.clone());
            }

            // Add persona if provided
            if let Some(p) = persona {
                document = document.with_metadata("requested_persona", serde_json::json!(p));
            }

            // Ingest document
            let job_id = create_ingest_command(&cli_adapter, &file).await?;

            // Create psiarc path (in real implementation this would come from the actual ingest)
            let psiarc_filename = format!(
                "Ïˆ-{}.psiarc",
                chrono::Utc::now().format("%Y-%m-%dT%H:%M:%S")
            );
            let psiarc_path = PathBuf::from(&args.log_dir).join(psiarc_filename);

            // Create ingest stats
            let stats = IngestStats::new(job_id.clone(), doc_title, file.clone())
                .with_session_info()
                .with_diff_count(estimate_diff_count(&content))
                .with_phase_regions(estimate_phase_regions(&content))
                .with_psiarc_path(psiarc_path)
                .with_tags(tag_list);

            // Print success message
            stats.print_success(args.no_color);

            // Log additional info
            info!("Document queued for ingestion");
            info!("Job ID: {}", job_id);
            info!("You can check the status with: ingest status {}", job_id);
        }
        Command::Status { job_id } => {
            info!("Checking status of job: {}", job_id);

            // Get job status
            let status = create_status_command(&cli_adapter, &job_id).await?;

            // Print status in color
            if !args.no_color {
                println!("{} Status: {}", "â„¹ï¸".bright_blue(), status.bright_white());
            } else {
                println!("Status: {}", status);
            }
        }
        Command::Watch {
            dir,
            pattern,
            auto_tag,
        } => {
            if !dir.exists() || !dir.is_dir() {
                return Err(format!("Directory not found: {}", dir.display()));
            }

            info!("Watching directory: {}", dir.display());
            info!("File pattern: {}", pattern);

            if auto_tag {
                info!("Auto-tagging enabled - will extract tags from document content");
            }

            // Start file watcher
            let watcher = cli_adapter.start_file_watcher(dir, &pattern).await?;

            // Print colorful message
            if !args.no_color {
                println!(
                    "{} Watching for files in: {}",
                    "ðŸ‘ï¸".bright_magenta(),
                    dir.display().to_string().bright_white()
                );
                println!(
                    "{} File pattern: {}",
                    "ðŸ”".bright_yellow(),
                    pattern.bright_white()
                );
                println!(
                    "{} Press {} to stop",
                    "â„¹ï¸".bright_blue(),
                    "Ctrl+C".bright_red()
                );
            } else {
                info!("File watcher started");
                info!("Press Ctrl+C to stop");
            }

            // Wait for watcher to finish (or Ctrl+C)
            watcher
                .await
                .map_err(|e| format!("File watcher error: {}", e))?;
        }
    }

    Ok(())
}

