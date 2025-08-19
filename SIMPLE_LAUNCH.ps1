Write-Host "TORI Launcher - Choose an option:"
Write-Host "1. Frontend only"
Write-Host "2. Frontend + Backend"
Write-Host "3. Frontend + Backend + MCP"
Write-Host "4. Full System (ALL services)"

$choice = Read-Host "Enter choice (1-4)"

if ($choice -eq "1") {
    cd tori_ui_svelte
    npm run dev
}
elseif ($choice -eq "2") {
    ./START_FULL_TORI_SYSTEM.bat
}
elseif ($choice -eq "3") {
    Write-Host "Starting MCP servers..."
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd mcp-server-architecture; npm run start"
    Start-Sleep -Seconds 5
    Write-Host "Starting TORI system..."
    ./START_FULL_TORI_SYSTEM.bat
}
elseif ($choice -eq "4") {
    Write-Host "Starting MCP servers..."
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd mcp-server-architecture; npm run start"
    Start-Sleep -Seconds 5
    Write-Host "Starting TORI system..."
    Start-Process cmd -ArgumentList "/c", "START_FULL_TORI_SYSTEM.bat"
    Start-Sleep -Seconds 5
    Write-Host "Starting PDF service..."
    python run_stable_server.py
}
else {
    Write-Host "Invalid choice"
}
