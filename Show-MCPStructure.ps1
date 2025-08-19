Write-Host "MCP-TORI Integration Structure" -ForegroundColor Green
Write-Host "===============================" -ForegroundColor Green

Write-Host "`nC:\Users\jason\Desktop\tori\kha\" -ForegroundColor Yellow

# Key Python files
Write-Host "|- Python Backend" -ForegroundColor Magenta
Write-Host "|  |- run_stable_server.py (Main server)" -ForegroundColor Cyan
Write-Host "|  |- mcp_bridge_real_tori.py (TORI Bridge)" -ForegroundColor Cyan
Write-Host "|  |- test_real_tori_filtering.py (Tests)" -ForegroundColor Cyan
Write-Host "|  \- ingest_pdf\" -ForegroundColor Yellow
Write-Host "|     |- pipeline.py (TORI filters)" -ForegroundColor Cyan
Write-Host "|     \- source_validator.py (Quality checks)" -ForegroundColor Cyan

# MCP TypeScript
Write-Host "|- MCP TypeScript Architecture" -ForegroundColor Blue
Write-Host "|  \- mcp-server-architecture\" -ForegroundColor Yellow
Write-Host "|     |- src\" -ForegroundColor Yellow
Write-Host "|     |  |- core\" -ForegroundColor Yellow
Write-Host "|     |  |  |- trust-kernel.ts" -ForegroundColor Cyan
Write-Host "|     |  |  \- types.ts" -ForegroundColor Cyan
Write-Host "|     |  |- integration\" -ForegroundColor Yellow
Write-Host "|     |  |  |- mcp-gateway.ts" -ForegroundColor Cyan
Write-Host "|     |  |  |- python-bridge.ts" -ForegroundColor Cyan
Write-Host "|     |  |  \- tori-filter.ts" -ForegroundColor Cyan
Write-Host "|     |  |- servers\" -ForegroundColor Yellow
Write-Host "|     |  |  |- mcp-kaizen.ts" -ForegroundColor Cyan
Write-Host "|     |  |  \- mcp-celery.ts" -ForegroundColor Cyan
Write-Host "|     |  \- main.ts" -ForegroundColor Cyan
Write-Host "|     \- package.json" -ForegroundColor Green

# Frontend
Write-Host "\- Svelte Frontend" -ForegroundColor Red
Write-Host "   \- tori_ui_svelte\" -ForegroundColor Yellow
Write-Host "      |- src\" -ForegroundColor Yellow
Write-Host "      \- package.json" -ForegroundColor Green

Write-Host "`nConnection Flow:" -ForegroundColor Magenta
Write-Host "Svelte (:5173) -> Python TORI (:8002) -> MCP Bridge -> MCP Gateway (:8080)" -ForegroundColor White