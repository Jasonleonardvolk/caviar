# MCP Server Creator & PDF Manager v2

This module provides tools for creating and managing MCP micro-servers with PDF document support in the TORI ecosystem.

## 🆕 Version 2 Updates

Based on production code review, v2 includes:
- **Supervisor Pattern**: Automatic restart of crashed loops with exponential backoff
- **Critic Hub Integration**: Proper health and performance monitoring
- **Metrics Tracking**: Comprehensive success rate and error tracking  
- **Enhanced Configuration**: Environment-based settings for all parameters
- **Improved Resilience**: Better error handling and recovery mechanisms

See [MIGRATION_GUIDE_V2.md](./MIGRATION_GUIDE_V2.md) for upgrading existing servers.

## Features

- **Create servers** with or without PDF specifications
- **Multiple PDF support** - Add multiple PDFs during creation or later
- **Batch operations** - Add entire directories of PDFs at once
- **PDF management** - Add, remove, list, and refresh PDFs for any server
- **Statistics tracking** - View PDF counts and character totals
- **Automatic text extraction** - PDFs are processed and text is extracted for use
- **Self-healing servers** - Supervisor pattern ensures continuous operation
- **Critic consensus** - Integrated health monitoring and evaluation

## Installation

```bash
pip install PyPDF2  # Required for PDF extraction
pip install httpx   # For async operations

# Optional for advanced features
pip install scikit-learn numpy
```

## Quick Start

### Setup

```bash
cd ${IRIS_ROOT}\mcp_metacognitive\create

# Run initial setup
python setup.py
```

### Creating Servers

```bash
# Create a simple server without PDFs
python mk_server.py create intent "Prajna intent tracker"

# Create a server with one PDF
python mk_server.py create empathy "Empathy sentiment module" empathy_paper.pdf

# Create a server with multiple PDFs
python mk_server.py create reasoning "Advanced reasoning module" paper1.pdf paper2.pdf paper3.pdf
```

### Managing PDFs

```bash
# Add PDFs to existing server
python mk_server.py add-pdf empathy new_research.pdf another_paper.pdf

# List all PDFs for a server
python mk_server.py list-pdfs empathy

# Batch add all PDFs from a directory
python pdf_manager.py batch-add empathy ./research/empathy_papers/

# Remove a specific PDF
python pdf_manager.py remove-pdf empathy old_paper.pdf

# Refresh seed.txt from all PDFs
python pdf_manager.py refresh empathy

# View statistics
python pdf_manager.py stats              # All servers
python pdf_manager.py stats empathy      # Specific server
```

## Configuration (v2)

Set environment variables for flexible configuration:

```bash
# Global defaults
export DEFAULT_ANALYSIS_INTERVAL=300  # 5 minutes for development

# Per-server overrides
export EMPATHY_ANALYSIS_INTERVAL=600
export EMPATHY_ENABLE_WATCHDOG=true
export EMPATHY_ENABLE_CRITICS=true
export EMPATHY_WATCHDOG_TIMEOUT=60
```

## Server Architecture (v2)

Generated servers now include:

1. **Supervisor Loop**: Monitors and restarts the main processing loop
2. **Metrics Tracking**: Success rates, error counts, execution statistics
3. **Critic Integration**: Automatic health and performance reporting
4. **Configurable Intervals**: Environment-based timing control
5. **Graceful Degradation**: Optional features fail safely

```
agents\empathy\
   ├─ __init__.py
   ├─ empathy.py              # Server with v2 enhancements
   ├─ spec.json               # PDF metadata and tracking
   ├─ seed.txt                # Combined text (max 50k chars)
   └─ resources\
       ├─ paper1.pdf          # Original PDFs
       ├─ paper2.pdf
       └─ paper3.pdf
```

## Server API (v2)

Generated servers include these methods:

```python
# Core execution with metrics
result = await server.execute(input_data)

# PDF management
pdfs = server.list_pdfs()
content = server.get_pdf_content("paper1.pdf")

# Lifecycle with supervisor
await server.start()      # Starts supervisor + main loop
await server.shutdown()   # Graceful shutdown

# Health monitoring
success_rate = server.metrics.success_rate
total_execs = server.metrics.total_executions
```

## Integration with TORI

### Auto-Registration
Servers are automatically registered with the agent_registry on import.

### PSI Archive Logging
Comprehensive event logging including:
- Server lifecycle events
- Execution metrics
- Supervisor restarts
- Error tracking

### Critic Hub Integration (v2)
Each server automatically registers:
- `{server}_performance` critic
- `{server}_health` critic

Metrics flow to the consensus panel for system-wide monitoring.

### Kaizen Integration
The Kaizen agent can trigger paper searches and use the PDF content:

```python
# In kaizen.py
from mcp_bridge import dispatch

# Trigger paper search
dispatch("paper_fetcher.request", {
    "query": "empathy measurement techniques",
    "destination_server": "empathy"
})
```

## Production Deployment

See [PRODUCTION_CHECKLIST.md](./PRODUCTION_CHECKLIST.md) for comprehensive deployment guide.

Key steps:
1. Run `python setup.py`
2. Configure environment variables
3. Create your servers
4. Start TORI and monitor logs
5. Check critic consensus dashboard

## Best Practices

1. **Use supervisor pattern** - All v2 servers auto-recover from crashes
2. **Monitor metrics** - Check success rates via critic hub
3. **Configure intervals** - Use env vars for different environments
4. **Organize PDFs** - Keep related papers in directories
5. **Regular health checks** - Monitor supervisor restart events

## Troubleshooting

See detailed troubleshooting in [PRODUCTION_CHECKLIST.md](./PRODUCTION_CHECKLIST.md#-troubleshooting-v2).

Common issues:
- **Loop crashes**: Check supervisor logs for restart patterns
- **Slow development**: Set `DEFAULT_ANALYSIS_INTERVAL=300`
- **Missing critics**: System degrades gracefully, check imports

## Documentation

- [README.md](./README.md) - This file
- [PRODUCTION_CHECKLIST.md](./PRODUCTION_CHECKLIST.md) - Deployment guide
- [MIGRATION_GUIDE_V2.md](./MIGRATION_GUIDE_V2.md) - Upgrade from v1
- [watchdog_enhancements.py](./watchdog_enhancements.py) - Registry improvements

## Future Enhancements

- OCR support for scanned PDFs
- Automatic categorization of PDFs
- Smart text chunking for better context
- PDF content search functionality
- Integration with vector databases for semantic search
- Advanced similarity matching (TF-IDF, embeddings)

## Version History

- **v2.0**: Supervisor pattern, critic integration, enhanced metrics
- **v1.0**: Basic server creation with PDF support
