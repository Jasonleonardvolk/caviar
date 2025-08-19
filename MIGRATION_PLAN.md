# TORI MCP Server Migration Plan

## Overview
This document outlines the migration plan to consolidate functionality from `mcp_server_arch` into `mcp_metacognitive` and fix the server startup issues.

## Current State Analysis

### mcp_server_arch
- **Architecture**: Hybrid TypeScript/Python with npm-based MCP server
- **Key Components**:
  - TypeScript MCP gateway (port 3000/3001)
  - Python bridge (`mcp_bridge_real_tori.py`) with TORI filtering
  - Agent system (Daniel, Tonka) with hot-swapping
  - PsiArchive for event logging
  - Integration with `ingest_pdf` module

### mcp_metacognitive
- **Architecture**: Pure Python using FastMCP
- **Issues**:
  - Missing `mcp` and `fastmcp` package dependencies
  - Import errors in `enhanced_launcher.py`
  - No actual MCP server implementation (just structure)

## Migration Strategy

### Phase 1: Fix Immediate Issues (Quick Fix)
1. **Install missing dependencies**
   ```bash
   cd ${IRIS_ROOT}\mcp_metacognitive
   pip install mcp-server-sdk fastmcp
   ```

2. **Fix enhanced_launcher.py imports**
   - Change incorrect import statements
   - Add proper error handling for missing modules

3. **Create fallback server implementation**
   - Add a minimal working server that doesn't require external MCP packages

### Phase 2: Full Migration (Complete Solution)
1. **Migrate core functionality**:
   - Port PsiArchive to mcp_metacognitive
   - Port agent registry and hot-swapping
   - Port TORI filtering integration

2. **Consolidate architecture**:
   - Choose between TypeScript MCP + Python bridge OR pure Python FastMCP
   - Migrate all functionality to chosen architecture

3. **Update enhanced_launcher.py**:
   - Fix import paths
   - Add proper server initialization

## Implementation Steps

### Step 1: Create Compatibility Layer
Create a compatibility module that bridges the gap between the two architectures.

### Step 2: Port Core Components
- PsiArchive → mcp_metacognitive/core/psi_archive.py
- Agent Registry → mcp_metacognitive/core/agent_registry.py
- TORI Bridge → mcp_metacognitive/core/tori_bridge.py

### Step 3: Update Server Implementation
Modify server.py to work without external MCP dependencies if they're not available.

### Step 4: Fix Enhanced Launcher
Update the launcher to properly handle both architectures.

## Timeline
- Phase 1 (Quick Fix): Immediate - Get server running
- Phase 2 (Full Migration): 2-3 days - Complete consolidation

## Dependencies to Resolve
1. `mcp` package - Either install or create mock
2. `fastmcp` package - Either install or create mock
3. `ingest_pdf` module - Check if available or create stubs
4. TypeScript MCP server - Decide if keeping or removing

## Next Actions
1. Implement Phase 1 quick fixes
2. Test server startup
3. Plan Phase 2 migration based on what works
