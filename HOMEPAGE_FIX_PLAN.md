# TORI HOMEPAGE FIX PLAN

## Current State Analysis

### Issues Identified:
1. **Stuck Loading Spinner** - In the main chat area (center)
2. **Chat System Not Working** - "MESSAGE SYSTEM (PRAJNA?) IS NOT WORKING"
3. **PDF Upload Not Working** - Right panel ScholarSphere showing "Debug: Ready for upload"
4. **Memory System Empty** - Left panel showing "No memory entries yet"
5. **Layout Issues** - Poor spacing, overlapping elements
6. **Color Issues** - Colors too bright/harsh

## File Structure Understanding:

### Layout Composition:
- **+layout.svelte** - Main layout with 3-panel structure
  - Left: `<MemoryPanel />` 
  - Center: `<slot />` (contains +page.svelte chat)
  - Right: `<ScholarSpherePanel />` (admin only)

### Current Issues by File:

1. **+page.svelte** (Chat Component):
   - Simplified version only has chat interface
   - Missing complex homepage elements from screenshot
   - Shows "Memory: Initializing" stuck state
   - Phase showing "idle" 

2. **ScholarSpherePanel.svelte** (Upload Component):
   - Upload endpoint `/api/upload` may not be working
   - SSE progress tracking implemented but may have issues
   - Shows "Debug: Ready for upload" but uploads fail

3. **MemoryPanel.svelte** (Left Panel):
   - Not examined yet but showing empty state

## Fix Priority Order:

### 1. IMMEDIATE: Clear Browser Cache
```bash
# The current code doesn't match what's showing in browser
# This suggests heavy caching issues
```

### 2. Fix Chat System (Highest Priority - User spent 4 days on this)
- Check if Prajna API is running (port 8002 or alternative)
- Verify `/api/answer` endpoint is working
- Fix the stuck "Memory: Initializing" state
- Test message sending functionality

### 3. Fix Upload Pipeline (User spent 4.5 days on this)
- Verify `/api/upload` endpoint exists and works
- Check if backend PDF processing is functional
- Test SSE progress tracking
- Ensure uploaded documents appear in list

### 4. Fix Layout & Visual Issues
- Adjust color scheme (dim the vibrancy)
- Fix spacing and alignment
- Remove/hide debug information
- Implement hologram space (left panel)
- Show previous conversations instead of brain icon

### 5. Clean Up Information Overload
- Hide excessive status indicators
- Move debug info to proper debug panel
- Simplify the interface

## Next Steps:

1. **First, we need to verify what's actually running**:
   - Check if the simplified +page.svelte is being served
   - Or if there's a cached/different version

2. **Start with backend verification**:
   - Test API endpoints directly
   - Check server logs for errors

3. **Then fix frontend systematically**:
   - One component at a time
   - Test after each change

## Safety Measures:
- Make backups before any changes
- Test each fix in isolation
- Use the filesystem server to save all work
- Proceed slowly and carefully
