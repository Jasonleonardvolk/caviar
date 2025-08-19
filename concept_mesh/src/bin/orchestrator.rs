//! Orchestrator Daemon
//!
//! The main daemon for the concept mesh, which manages the orchestrator
//! and provides a persistent service for document ingestion.

use concept_mesh_rs::agents::orchestrator::{create_orchestrator, OrchestratorConfig};
use concept_mesh_rs::cbd::{CBDConfig, ConceptBoundaryDetector};
use concept_mesh_rs::mesh::InMemoryMesh;
use concept_mesh_rs::psiarc::{PsiarcManager, PsiarcOptions};
use concept_mesh_rs::ui::genesis_bridge::{create_genesis_ui_bridge, GenesisBridge, OscillatorBloom};
use concept_mesh_rs::{self, LargeConceptNetwork, GENESIS_FRAME_ID};

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::signal;
use tokio::time::sleep;
use tracing::{debug, error, info, warn};
use tracing_subscriber::FmtSubscriber;

/// Command line arguments for the orchestrator daemon
#[derive(Parser, Debug)]
#[clap(name = "orchestrator", about = "Concept Mesh orchestrator daemon")]
struct Args {
    /// The name of the corpus to use
    #[clap(short, long, default_value = "MainCorpus")]
    corpus: String,

    /// Log directory for storing Ïˆarc logs
    #[clap(short, long, default_value = "logs")]
    log_dir: PathBuf,

    /// Maximum number of concurrent ingest jobs
    #[clap(long, default_value = "4")]
    max_jobs: usize,

    /// Whether to force a new GENESIS event
    #[clap(long)]
    force_genesis: bool,

    /// API port for HTTP interface
    #[clap(long, default_value = "8080")]
    api_port: u16,

    /// Verbosity level (0-3)
    #[clap(short, long, default_value = "1")]
    verbose: u8,
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

    info!("Starting concept-mesh orchestrator daemon");
    info!("Corpus: {}", args.corpus);
    info!("Log directory: {}", args.log_dir.display());
    info!("Max concurrent jobs: {}", args.max_jobs);

    // Create in-memory mesh
    let mesh = Arc::new(InMemoryMesh::new());

    // Create orchestrator config
    let config = OrchestratorConfig {
        genesis_on_startup: true,
        corpus_id: args.corpus.clone(),
        log_directory: args.log_dir.to_string_lossy().to_string(),
        max_concurrent_jobs: args.max_jobs,
        ..Default::default()
    };

    // Create orchestrator
    info!("Creating orchestrator...");
    let orchestrator = create_orchestrator("main-orchestrator", Some(config)).await?;

    // Set up Genesis UI bridge
    info!("Setting up GENESIS UI bridge...");
    let bloom = Arc::new(Mutex::new(OscillatorBloom::new(3000, 60)));
    let genesis_bridge =
        create_genesis_ui_bridge("main-ui", Arc::clone(&mesh), Arc::clone(&bloom)).await?;

    // Force GENESIS if requested
    if args.force_genesis {
        info!("Forcing GENESIS event...");
        orchestrator.trigger_genesis().await?;
    }

    // Start API server if port is provided
    if args.api_port > 0 {
        info!("Starting API server on port {}...", args.api_port);

        // For Day 1, we'll just print a message
        // TODO: Implement real API server
        info!(
            "API server would start on port {} (not implemented yet)",
            args.api_port
        );
    }

    // Log startup complete
    info!("Orchestrator daemon started successfully");
    info!("Press Ctrl+C to stop");

    // Wait for Ctrl+C signal
    match signal::ctrl_c().await {
        Ok(()) => {
            info!("Ctrl+C received, shutting down...");
        }
        Err(e) => {
            error!("Error waiting for Ctrl+C: {}", e);
        }
    }

    // Graceful shutdown
    info!("Stopping Genesis UI bridge...");
    if let Err(e) = genesis_bridge.stop().await {
        error!("Error stopping Genesis UI bridge: {}", e);
    }

    info!("Stopping orchestrator...");
    if let Err(e) = orchestrator.stop().await {
        error!("Error stopping orchestrator: {}", e);
    }

    // Give things a moment to shut down
    sleep(Duration::from_millis(500)).await;

    info!("Orchestrator daemon shutdown complete");
    Ok(())
}

