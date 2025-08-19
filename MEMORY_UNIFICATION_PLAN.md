# TORI Memory Unification & Group Multi-Tenancy Implementation Plan

## Current State Analysis
- **Concept Mesh**: Split between Rust (Penrose) and Python implementations
- **Soliton Memory**: Using in-memory stub, not integrated with Concept Mesh
- **Data Persistence**: JSON files, no database (keeping it that way)
- **Multi-tenancy**: Currently global mesh, needs user/group scoping

## Phase 0: Preparation & Backup
```bash
# Create backup tag
git tag backup-before-groups
git push origin backup-before-groups

# Create new branch
git checkout -b memory-unification-groups
```

## Phase 1: Core Infrastructure

### 1.1 Group Manager
**File**: `${IRIS_ROOT}\core\group_manager.py`
- CRUD operations for groups
- User-to-group mapping
- Permission management
- Path resolution for user/group data

### 1.2 Unified Memory Manager
**File**: `${IRIS_ROOT}\core\unified_memory.py`
- Single interface for all memory operations
- Routes to appropriate Concept Mesh instance (user or group scoped)
- Integrates FractalSolitonMemory

### 1.3 Data Structure