# Frontend Loading Debug Guide

## Quick Diagnosis

### 1. Clear Vite Cache (Most Common Fix)
```bash
# Stop the launcher (Ctrl+C)
cd tori_ui_svelte
rm -rf node_modules/.vite
npm run dev
```

### 2. Check Browser Console
Open http://localhost:5173 and check for these common errors:

**CORS Error?**
```
Access to fetch at 'http://localhost:8002' from origin 'http://localhost:5173' has been blocked by CORS
```
**Fix**: Check if API is running and CORS is configured

**404 on API calls?**
```
GET http://localhost:8002/api/... 404 (Not Found)
```
**Fix**: API routes might be wrong or API not running

**Module errors?**
```
Failed to resolve module specifier
```
**Fix**: Missing npm dependencies

### 3. Test API Connection
```javascript
// Paste this in browser console:
fetch('http://localhost:8002/api/health')
  .then(r => r.json())
  .then(data => console.log('API Working:', data))
  .catch(err => console.error('API Error:', err))
```

### 4. Check Network Tab
1. Open DevTools Network tab
2. Refresh page
3. Look for:
   - Red requests (failed)
   - Pending requests (hanging)
   - CORS errors

### 5. Common Fixes

**If blank white page:**
```bash
cd tori_ui_svelte
npm install
npm run build
npm run preview
```

**If API connection fails:**
- Check `tori_ui_svelte/src/lib/config.ts` has correct API URL
- Verify API is on port 8002: `curl http://localhost:8002/api/health`

**If "autofocus" warning is blocking:**
Remove autofocus from login page as shown in logs

### 6. Force Refresh
- Chrome/Edge: Ctrl+Shift+R
- Clear browser cache: Ctrl+Shift+Delete → Cached images and files

Let me know what errors you see in the console!
