# TORI System Debug Report and Fix Plan
Date: January 19, 2025

## Issues Identified

### 1. ScholarSphere Upload Error
- **Issue**: User reports error when uploading a standard PDF to ScholarSphere
- **Findings**: 
  - The upload endpoint `/api/upload` exists and has comprehensive error handling
  - ScholarSphere is configured for local file storage only (no external service)
  - SSE progress tracking is implemented
  - Possible causes: PDF processing failure, concept extraction issues, or file size/format problems

### 2. Enola Persona Not Showing
- **Issue**: Enola should be the default persona but isn't displayed
- **Findings**:
  - Enola IS defined in `ghostPersona.ts` and set as default (`persona: 'Enola'`, `activePersona: 'Enola'`)
  - Enola is in the ghost registry
  - BUT: Enola is NOT in the PersonaSelector component's personas array
  - The PersonaSelector only includes: Mentor, Scholar, Explorer, Architect, Creator

### 3. Hologram Integration
- **Issue**: Hologram should show default persona
- **Findings**:
  - HologramBridge is implemented for SSE connection
  - HologramBadge only shows connection status, not persona information
  - Need to create a hologram display component that shows the active persona

## Fix Implementation Plan

### Fix 1: Add Enola to PersonaSelector
Update the PersonaSelector component to include Enola with proper 4D coordinates.

### Fix 2: Create Hologram Persona Display
Create a component that shows the active persona in the hologram interface.

### Fix 3: Debug ScholarSphere Upload
Add better error logging and fallback handling for PDF uploads.

### Fix 4: Integration Testing Script
Create a script to verify all components are working together.

## Implementation Files

1. `fix_persona_selector.py` - Adds Enola to PersonaSelector
2. `fix_hologram_display.py` - Creates hologram persona display
3. `fix_scholarsphere_upload.py` - Improves upload error handling
4. `test_integration.py` - Tests the complete system

Let's implement these fixes!
