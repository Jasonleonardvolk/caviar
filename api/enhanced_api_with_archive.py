#!/usr/bin/env python3
"""
Enhanced API Integration - Adds PsiArchive endpoints to existing API
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import existing enhanced API app
from api.enhanced_api import app

# Import and mount archive endpoints
from api.archive_endpoints import archive_router

# Mount the archive router
app.include_router(archive_router)

# Add to the API documentation
if hasattr(app, 'description'):
    app.description += """

## PsiArchive Endpoints

The API now includes comprehensive PsiArchive query endpoints:

- **GET /api/archive/origin/{concept_id}** - Find when and where a concept was first learned
- **GET /api/archive/session/{session_id}** - Debug session events for hallucination analysis  
- **GET /api/archive/delta?since=ISO_TIMESTAMP** - Get mesh deltas for incremental sync
- **GET /api/archive/query** - Query archive with filters
- **POST /api/archive/seal** - Manually seal yesterday's archive
- **GET /api/archive/health** - Archive health status

These endpoints provide full provenance tracking, time-travel debugging, and efficient sync capabilities.
"""

print("✅ PsiArchive endpoints integrated into Enhanced API")
print("📍 Archive routes available at /api/archive/*")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
