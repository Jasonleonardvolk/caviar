# Concept Mesh

**Important**: This is the only active mesh crate; ignore concept_mesh (underscore) directory

A ConceptDiff-based distributed cognitive architecture that enables concept-oriented agent communication and phase-aligned concept storage.

## Overview

The Concept Mesh is a framework for building concept-oriented agents that communicate via ConceptDiffs. It provides a phase-aligned storage system that replaces traditional embedding databases with a concept-first approach to creating a shared cognitive space.

## Key Components

![Concept Mesh Architecture](docs/concept-mesh-architecture.png)

- **Concept Boundary Detector (CBD)**: Segments content at semantic breakpoints rather than arbitrary fixed-size chunks. It uses phase shifts, Koopman-mode inflections, and eigen-entropy slopes to determine natural concept boundaries.

- **ConceptDiff**: A set of graph operations representing changes to the concept network. This is the primary communication mechanism in the mesh, allowing agents to share cognitive updates.

- **Large Concept Network (LCN)**: A phase-aligned concept storage system. Unlike traditional vector databases, the LCN organizes information around concepts rather than embeddings.

- **Mesh**: The communication infrastructure that connects agents and enables ConceptDiff propagation through the system.

- **Agentic Orchestrator**: Coordinates agents in the mesh, handling the lifecycle of ingest jobs, routing ConceptPacks, and managing the phase-synchronization.

- **PsiArc (ψarc)**: Persistent storage and replay system for ConceptDiffs, enabling time-travel debugging and system reconstruction.

## GENESIS Event

The GENESIS event initializes the concept mesh, creating the `TIMELESS_ROOT` concept that serves as the anchor for all other concepts. This event triggers the oscillator bloom animation in the UI and establishes the foundational cognitive structure.

## Getting Started

### Prerequisites

- Rust 1.70+ with Cargo
- OpenBLAS or another BLAS implementation (for ndarray-linalg)

### Building

```bash
cargo build --release
```

### Running the Daemon

```bash
cargo run --release --bin orchestrator -- --corpus MyCorpus
```

### Initializing GENESIS

```bash
cargo run --release --bin genesis -- --corpus MyCorpus
```

### Ingesting Documents

```bash
cargo run --release --bin ingest -- ingest my_document.txt
```

### Viewing ψarc Logs

```bash
cargo run --release --bin psidiff-viewer -- view logs/orchestrator_*.psiarc
```

## Architecture

### Phase-Aligned Storage

Unlike traditional embedding databases that store vectors, the Concept Mesh uses phase-aligned storage to maintain conceptual coherence across the system. This approach allows for:

1. Preservation of semantic relationships
2. Dynamic concept evolution
3. Cross-modal conceptual alignment
4. Cognitive resonance between agents

### ConceptDiff Communication

Agents in the mesh communicate via ConceptDiffs, which represent graph operations that modify the shared concept space. This approach has several benefits:

1. Bandwidth efficiency - only changes are transmitted
2. Semantic expressiveness - operations capture intent
3. Temporal coherence - operations can be replayed
4. Transactional integrity - related changes are grouped

### Concept Boundary Detection

The CBD segments content at natural conceptual boundaries rather than arbitrary fixed-size chunks. This results in more coherent concept packs that preserve semantic units, improving understanding and making concept mapping more precise.

## CLI Tools

The system provides several command-line tools:

- **genesis**: Initialize a concept mesh with a GENESIS event
- **orchestrator**: Run the mesh orchestrator daemon
- **ingest**: Ingest documents into the concept mesh
- **psidiff-viewer**: Visualize ConceptDiff transactions from ψarc logs

## Development

### Testing

```bash
cargo test
```

### Documentation

```bash
cargo doc --open
```

## License

[MIT License](LICENSE)

## Acknowledgments

- The TORI team
- Contributors to the ScholarSphere cognitive architecture
