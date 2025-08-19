//! PSIDiff Validate CLI Tool
//!
//! Command-line interface for the ConceptDiff Validator Tool.
//! This tool validates the integrity of a psiarc log by ensuring all diffs
//! can be applied to an LCN and comparing against a canonical snapshot if provided.

use clap::Parser;
use concept_mesh_rs::tools::psidiff_validate::{execute_validate_cmd, ValidateArgs};

#[tokio::main]
async fn main() -> Result<(), String> {
    // Parse command line arguments
    let args = ValidateArgs::parse();

    // Execute validate command
    execute_validate_cmd(args).await
}

