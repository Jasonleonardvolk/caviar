# Building and Deploying TORI Chat - Complete Instructions

This document provides step-by-step instructions for building and deploying the TORI Chat application, incorporating all fixes for React 18 compatibility issues.

## Quick Start (Automated Method)

### Windows
```cmd
# From Command Prompt
deploy-tori-chat.bat

# From PowerShell
.\deploy-i wtori-chat.bat
```

### Mac/Linux
```bash
# Make script executable
chmod +x deploy-tori-chat.sh

# Run script
./deploy-tori-chat.sh
```

## Manual Steps (For Developers and Troubleshooting)

If you prefer to understand each step of the process or need to troubleshoot specific issues:

### 1. Fix React Dependency Conflicts

The key issue is that `react-diff-viewer@3.1.1` conflicts with React 18. Fix it by:

```bash
# Option A: Replace with React 18 compatible version
npm uninstall react-diff-viewer
npm install react-diff-viewer-continued@4.0.0 --save-exact

# OR Option B: Remove entirely if not needed for production
npm uninstall react-diff-viewer
```

You can also use our helper script:
```bash
node fix-react-deps.js
```

### 2. Install Dependencies

```bash
cd tori_chat_frontend
npm ci --omit dev
```

### 3. Build the Application

```bash
npm run build
```

### 4. Check if Port 3000 is Available

```bash
# Simple check
node check-port.js

# Check with option to kill conflicting process
node check-port.js --kill
```

### 5. Serve the Application

```bash
node start-production.cjs
```

Access the application at: http://localhost:3000 (or 3001 if you changed the port)

## Common Issues & Solutions

### 1. Script Path in HTML

If you see a build error related to main.jsx, verify that `src/index.html` has the correct script path:
```html
<script type="module" src="/src/main.jsx"></script>
```

### 2. Vite Configuration 

Verify that `vite.config.js` has:
```js
build: {
  outDir: 'dist',
  emptyOutDir: true,
  base: './', // Ensures assets load from any subfolder
  rollupOptions: {
    input: {
      main: path.resolve(__dirname, 'src/index.html')
    }
  }
},
```

### 3. Environment Variables

Create/verify `.env.production` contains:
```
VITE_APP_MODE=chat
PUBLIC_URL=/
```

### 4. Port in Use

If port 3000 is already in use:
```bash
# On Windows
netstat -ano | findstr :3000
taskkill /F /PID <PID>

# On Mac/Linux
lsof -i :3000
kill -9 <PID>
```

Or set a different port:
```bash
# Windows
set PORT=3001
node start-production.cjs

# Mac/Linux
PORT=3001 node start-production.cjs
```

### 5. Verifying the Build

To verify that you have the correct UI and not the redirect page:
1. Check that dist/index.html is larger than 1000 bytes
2. Look for "ReactDOM" in dist/index.html
3. Open in browser - you should see the full Chat UI, not a redirect message

## For CI/CD Systems

Add these steps to your CI pipeline:

```yaml
- name: Fix React dependency conflicts
  run: |
    if npm ls react-diff-viewer; then
      npm uninstall react-diff-viewer
      npm install react-diff-viewer-continued@4.0.0 --save-exact
    fi

- name: Install dependencies
  working-directory: ./tori_chat_frontend
  run: npm ci --omit dev
  
- name: Build application
  working-directory: ./tori_chat_frontend
  run: npm run build
```

## Troubleshooting

If you encounter build issues:

1. Clean the node_modules directory and lockfile:
   ```bash
   rm -rf node_modules package-lock.json
   npm ci --omit dev
   ```

2. Check for Turborepo lock issues:
   ```bash
   # Windows
   taskkill /IM turbo.exe /F
   # Mac/Linux
   pkill -f turbo
   ```

3. Verify the built UI is not a redirect:
   ```bash
   grep "ReactDOM" tori_chat_frontend/dist/index.html
   ```
