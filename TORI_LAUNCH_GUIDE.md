# TORI Chat Launch Instructions

## Quick Start

There are several ways to launch TORI Chat:

### Method 1: Double-click from Windows Explorer
1. Navigate to `${IRIS_ROOT}`
2. Double-click on `LAUNCH_TORI.bat`

### Method 2: From Command Prompt (cmd.exe)
```cmd
cd ${IRIS_ROOT}
deploy-tori-chat-with-mcp.bat
```

### Method 3: From PowerShell (Recommended)
```powershell
cd ${IRIS_ROOT}
.\deploy-tori-chat-with-mcp.ps1
```

### Method 4: Using the Simple Launcher
```cmd
cd ${IRIS_ROOT}
LAUNCH_TORI.bat
```

## Common Issues and Solutions

### Issue: "The term is not recognized" in PowerShell
**Solution**: In PowerShell, you must prefix batch files with `.\` 
```powershell
.\deploy-tori-chat-with-mcp.bat
```

### Issue: Python SyntaxError with ✓ character
**Solution**: Don't run batch files with Python. Use one of the methods above.

### Issue: Port 3000 is already in use
**Solution**: The script will ask if you want to kill the process or use port 3001 instead.

### Issue: MCP key not configured
**Solution**: Ensure `.env.production` contains:
```
VITE_MCP_KEY=ed8c312bbb55b6e1fd9c81b44e0019ea
```

## What the Deployment Script Does

1. **Checks MCP Configuration**: Verifies the MCP key is properly set
2. **Fixes React Dependencies**: Resolves any React 18 compatibility issues
3. **Installs Dependencies**: Does a clean install of all required packages
4. **Builds the Application**: Creates the production build
5. **Verifies the Build**: Ensures the build output is valid
6. **Checks Port Availability**: Makes sure port 3000 (or 3001) is free
7. **Starts the Server**: Launches the production server

## Access the Application

Once started successfully, access TORI Chat at:
- Primary: http://localhost:3000
- Alternative: http://localhost:3001 (if port 3000 was in use)

## Stopping the Server

Press `Ctrl+C` in the terminal window to stop the server gracefully.

## File Locations

- Main deployment script: `deploy-tori-chat-with-mcp.bat`
- PowerShell version: `deploy-tori-chat-with-mcp.ps1`
- Simple launcher: `LAUNCH_TORI.bat`
- Frontend directory: `tori_chat_frontend/`
- Production server: `tori_chat_frontend/start-production.cjs`

## Memory System Integration

TORI Chat integrates with the Soliton Memory System for advanced cognitive features.
The MCP (Model Context Protocol) key enables communication with the memory servers.

## Troubleshooting

If you encounter issues:

1. Ensure Node.js and npm are installed:
   ```cmd
   node --version
   npm --version
   ```

2. Check that you're in the correct directory:
   ```cmd
   cd ${IRIS_ROOT}
   dir deploy-tori-chat-with-mcp.bat
   ```

3. Verify the frontend directory exists:
   ```cmd
   dir tori_chat_frontend
   ```

4. Check available disk space for node_modules installation

5. Ensure you have internet connection for npm package downloads

For additional help, check the logs in the terminal window for specific error messages.
