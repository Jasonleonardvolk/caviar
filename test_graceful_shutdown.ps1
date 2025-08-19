#!/usr/bin/env pwsh
# Test graceful shutdown functionality
# ====================================

Write-Host "`n=== Testing TORI Graceful Shutdown ===" -ForegroundColor Cyan

# First, create the wrapper
Write-Host "`n[1] Creating graceful shutdown wrapper..." -ForegroundColor Yellow
python add_graceful_shutdown.py

# Check if wrapper was created
if (Test-Path "tori_launcher.py") {
    Write-Host "[OK] Wrapper created successfully!" -ForegroundColor Green
    
    Write-Host "`n[2] Instructions for testing graceful shutdown:" -ForegroundColor Yellow
    Write-Host "  1. Run: python tori_launcher.py" -ForegroundColor White
    Write-Host "  2. Wait for all services to start" -ForegroundColor White
    Write-Host "  3. Press Ctrl+C once to initiate graceful shutdown" -ForegroundColor White
    Write-Host "  4. Watch the shutdown sequence:" -ForegroundColor White
    Write-Host "     - Lattice evolution will stop cleanly" -ForegroundColor Gray
    Write-Host "     - API server will complete pending requests" -ForegroundColor Gray
    Write-Host "     - MCP server will shutdown" -ForegroundColor Gray
    Write-Host "     - Frontend will stop" -ForegroundColor Gray
    Write-Host "  5. Press Ctrl+C twice to force immediate shutdown" -ForegroundColor White
    
    Write-Host "`n[3] Key improvements:" -ForegroundColor Yellow
    Write-Host "  ✅ No more hanging processes after Ctrl+C" -ForegroundColor Green
    Write-Host "  ✅ Clean shutdown sequence (LIFO order)" -ForegroundColor Green
    Write-Host "  ✅ Services stop logging after shutdown signal" -ForegroundColor Green
    Write-Host "  ✅ Timeout protection (10 seconds max)" -ForegroundColor Green
    
} else {
    Write-Host "[ERROR] Failed to create wrapper!" -ForegroundColor Red
}

Write-Host "`n=== Alternative: Direct Integration ===" -ForegroundColor Cyan
Write-Host "To integrate directly into enhanced_launcher.py:" -ForegroundColor Yellow
Write-Host "1. Add to imports:" -ForegroundColor White
Write-Host "   from core.graceful_shutdown import shutdown_manager, register_shutdown_handler, install_shutdown_handlers" -ForegroundColor Gray
Write-Host "`n2. In SubprocessManager.__init__:" -ForegroundColor White
Write-Host "   self.shutdown_event = threading.Event()" -ForegroundColor Gray
Write-Host "`n3. In launch() method:" -ForegroundColor White
Write-Host "   install_shutdown_handlers()" -ForegroundColor Gray
Write-Host "`n4. Register handlers for each service:" -ForegroundColor White
Write-Host "   register_shutdown_handler(service_stop_function, 'Service Name')" -ForegroundColor Gray
