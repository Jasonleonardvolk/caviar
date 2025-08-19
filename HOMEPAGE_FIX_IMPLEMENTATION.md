# TORI HOMEPAGE FIX - IMPLEMENTATION PLAN

## Architecture Understanding:

1. **Frontend (SvelteKit on port 5173)**:
   - `+layout.svelte` - Main 3-panel layout
   - `+page.svelte` - Chat component (center panel)
   - `ScholarSpherePanel.svelte` - Upload component (right panel)
   - `MemoryPanel.svelte` - Memory system (left panel)

2. **Backend (Prajna API on port 8002 or dynamic)**:
   - `/api/upload` - PDF upload endpoint with SSE progress
   - `/api/answer` - Chat/Prajna responses
   - `/api/soliton/*` - Memory system endpoints

## Issues and Fixes:

### 1. FIX CHAT SYSTEM (Priority 1)
**Issue**: "MESSAGE SYSTEM (PRAJNA?) IS NOT WORKING", stuck "Memory: Initializing"

**Root Cause**: The chat is trying to call `/api/answer` but may be hitting the wrong backend

**Fix**:
```javascript
// In +page.svelte, update the API call to use the correct backend port
const chatResponse = await fetch('http://localhost:8002/api/answer', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    user_query: currentMessage,
    persona: { name: data.user?.name || 'anonymous' }
  })
});
```

### 2. FIX UPLOAD SYSTEM (Priority 2)
**Issue**: Upload shows "Debug: Ready for upload" but doesn't work

**Root Cause**: Frontend is calling `/api/upload` but SvelteKit proxy may not be configured

**Fix Options**:

**Option A - Add SvelteKit proxy** (Recommended):
Create `vite.config.js` proxy:
```javascript
proxy: {
  '/api': {
    target: 'http://localhost:8002',
    changeOrigin: true
  }
}
```

**Option B - Direct API calls**:
Update `ScholarSpherePanel.svelte`:
```javascript
const uploadUrl = `http://localhost:8002/api/upload?progress_id=${encodeURIComponent(progressId)}`;
```

### 3. FIX MEMORY SYSTEM
**Issue**: Shows "No memory entries yet"

**Fix**: Ensure soliton endpoints are being called correctly:
```javascript
// Initialize memory
await fetch('http://localhost:8002/api/soliton/init', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ user: currentUserId })
});
```

### 4. FIX LAYOUT & COLORS
**Issue**: Colors too bright, layout issues

**Fix in +layout.svelte**:
```css
/* Dim the vibrancy */
.bg-gradient-to-br.from-indigo-500.to-purple-600 {
  opacity: 0.85;
}

/* Fix spacing */
.flex.h-screen {
  gap: 0;
  align-items: stretch;
}
```

## Implementation Steps:

### STEP 1: Check if backend is running
```bash
curl http://localhost:8002/api/health
```

### STEP 2: Fix the proxy configuration
1. Check `vite.config.js` in tori_ui_svelte
2. Add proxy configuration if missing
3. Restart frontend

### STEP 3: Update API endpoints in components
1. Fix chat endpoint in +page.svelte
2. Fix upload endpoint in ScholarSpherePanel.svelte
3. Fix memory endpoints in MemoryPanel.svelte

### STEP 4: Clear browser cache
```
Ctrl+Shift+R or Cmd+Shift+R
```

### STEP 5: Test each component
1. Test chat first
2. Test upload second
3. Test memory system last

## Safety Measures:
- Make backups of all files before changes
- Test one fix at a time
- Use filesystem server to save all changes
- Monitor browser console for errors
