# ALAN IDE Phase 3 - Sprint 1 Implementation

This directory contains the core implementation for the first sprint of the ALAN IDE Phase 3. The implementation focuses on:

1. Python AST to Concept Graph Converter
2. Project Vault for Secret Management
3. Import Wizard for Python Projects

## Prerequisites

Before running the ALAN IDE, make sure you have the following Python packages installed:

```bash
pip install cryptography flask rich
```

These packages provide the core functionality for encryption, REST API, and terminal UI.

## Components

### 1. Python AST to Concept Graph (`python_to_concept_graph.py`)

This module converts Python code into a concept graph suitable for visualization in the ConceptFieldCanvas. It:

- Parses Python code using the AST module
- Converts the AST to a graph of nodes and edges
- Scans for potential secrets using regex patterns
- Calculates phase and resonance values for visualization
- Implements Koopman spectral decomposition

### 2. Project Vault Service (`project_vault_service.py`)

This module provides secure storage for sensitive information like API keys and passwords. It:

- Implements AES-GCM encryption for secure storage
- Supports multiple storage backends (file and OS keychain)
- Exposes a REST API for vault operations
- Includes a CLI for vault management

### 3. Import Wizard (`import_wizard.py`)

This module provides a guided process for importing Python projects, including:

- Python project selection and validation
- Secret scanning and detection
- Interactive review of detected secrets
- Migration of secrets to the vault
- Concept graph generation

### 4. Demo Runner (`run_alan_ide.py`)

This script demonstrates the integration of all components and provides a simple way to test the implementation.

## Running the Demo

To try out the ALAN IDE Phase 3 implementation, run the following command:

```bash
python src/run_alan_ide.py [path_to_python_project]
```

By default, it will use the `src` directory if no path is provided.

## Command Line Options

### Import Wizard

```bash
python src/import_wizard.py [path_to_python_project] [--output results.json] [--graph graph.json]
```

- `path_to_python_project`: Path to a Python file or directory
- `--output`, `-o`: Output path for import results (optional)
- `--graph`, `-g`: Output path for concept graph JSON (optional)

### Project Vault Service

```bash
python src/project_vault_service.py [command] [options]
```

Commands:

- `put <key> <value>`: Store a secret
- `get <key>`: Retrieve a secret
- `delete <key>`: Delete a secret
- `list`: List all stored secrets
- `server`: Start the vault API server (with `--host` and `--port` options)

### Python to Concept Graph

```bash
python src/python_to_concept_graph.py <python_file_or_directory>
```

Analyzes Python code and generates a concept graph, saving it to `concept_graph.json`.

## Output Files

The demo and import wizard generate the following files in the `output` directory:

- `import_results.json`: Summary of the import process, including detected secrets
- `concept_graph.json`: The complete concept graph in JSON format

In a full implementation, the concept graph would be used to populate the ConceptFieldCanvas for visualization.
