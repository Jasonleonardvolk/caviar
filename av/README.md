# TORI Ingest Bus

A microservice for ingesting documents, conversations, and other content into the ScholarSphere knowledge system.

## Overview

The TORI Ingest Bus serves as the central ingestion pipeline for the TORI system, enabling the following capabilities:

- Queue and process documents of various types (PDF, markdown, conversations, etc.)
- Extract text content and structure it appropriately
- Chunk content for optimal knowledge representation
- Vectorize content for semantic search and concept mapping
- Map content to ScholarSphere concepts with phase vectors
- Store processed content with full provenance tracking
- Provide status monitoring and metrics for the ingestion process
- Expose functionality through a REST API and MCP interface

## Architecture

```
ingest-bus/
├── main.py                 # FastAPI application entry point
├── routes/                 # API route handlers
│   ├── queue.py            # Document queue endpoints
│   ├── status.py           # Job status endpoints
│   └── metrics.py          # System metrics endpoints
├── workers/                # Background processing workers
│   └── extract.py          # Content extraction and processing
├── models/                 # Data models and schemas
│   └── schemas.py          # Pydantic models
└── utils/                  # Utility modules
    ├── config_loader.py    # Configuration loading
    └── logger.py           # Logging utilities
```

## Integration Points

### ScholarSphere

The Ingest Bus connects to ScholarSphere for concept mapping and storage:

- Phase vectors are generated for all chunks
- Content is mapped to existing concepts in ScholarSphere
- New concepts may be created if needed
- Full provenance tracking links chunks to source documents

### CLINE Integration

CLINE can use the Ingest Bus through the MCP interface (defined in `registry/mcp/ingest.schema.yaml`):

- `ingest.queue`: Queue a document for processing
- `ingest.status`: Check the status of an ingest job
- `ingest.metrics`: Get system metrics
- `kb.search`: Search for content in the knowledge base
- `kb.retrieve`: Retrieve specific content by ID

### Conversation Extractor

The Ingest Bus leverages the existing conversation extractor (`extract_conversation.js`) to process CLINE conversations:

- Code snippets are properly extracted and processed
- Notes and documentation are captured
- Conversation text is cleaned and structured
- Metadata is preserved for provenance

## Running the Service

### Prerequisites

- Python 3.8+
- FastAPI and dependencies: `pip install fastapi uvicorn pydantic prometheus_client`
- Node.js (for conversation extraction)

### Starting the Server

```bash
cd ingest-bus
uvicorn main:app --reload --port 8080
```

The server will be available at http://localhost:8080 with API documentation at http://localhost:8080/docs.

### Configuration

Configuration is loaded from the `conversation_config.json` file in the project root. Key settings include:

- ScholarSphere connection parameters
- Chunking settings (size, overlap)
- Vectorization parameters
- MCP registry settings

## API Usage Examples

### Queue a Document

```bash
curl -X POST http://localhost:8080/queue \
  -H "Content-Type: application/json" \
  -d '{
    "document_type": "pdf",
    "title": "Example Document",
    "tags": ["example", "documentation"],
    "metadata": {"author": "TORI Team"}
  }' \
  --data-binary @path/to/document.pdf
```

### Check Job Status

```bash
curl http://localhost:8080/status/job/12345
```

### Get Metrics

```bash
curl http://localhost:8080/metrics
```

## Monitoring

The service exposes Prometheus metrics at http://localhost:8080/metrics/prometheus for integration with monitoring systems.

Key metrics include:

- `ingest_files_queued_total`
- `ingest_files_processed_total`
- `ingest_failures_total`
- `chunk_avg_len_chars`
- `concept_recall_accuracy`
- `active_jobs`
- `queue_depth`

## Development

### Running Tests

```bash
pytest
```

### Adding New Document Types

To add support for a new document type:

1. Add the type to the `DocumentType` enum in `models/schemas.py`
2. Implement an extraction function in `workers/extract.py`
3. Update the extraction routing in `process_ingest_job` in `routes/queue.py`
4. Update the MCP schema in `registry/mcp/ingest.schema.yaml`

## Future Enhancements

- Database persistence for jobs and metrics
- Advanced chunking strategies based on semantic boundaries
- Support for more multimedia content types (images, audio, video)
- Integration with TORI's phase-aligned reasoning system
- Hologram and spectral emotion mapping for audio/video content
