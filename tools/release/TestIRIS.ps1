param(
  [string]$RepoRoot = "D:\Dev\kha"
)

Set-Location $RepoRoot

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "TEST IRIS LOCALLY" -ForegroundColor Cyan  
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "`nOptions to test IRIS:" -ForegroundColor Yellow
Write-Host "1. Preview production build (recommended)" -ForegroundColor White
Write-Host "2. Run development server" -ForegroundColor White
Write-Host "3. Serve static files" -ForegroundColor White

$choice = Read-Host "`nChoose option (1-3)"

switch ($choice) {
    "1" {
        Write-Host "`nStarting production preview..." -ForegroundColor Green
        Write-Host "This will serve the built application" -ForegroundColor Gray
        
        Set-Location "tori_ui_svelte"
        
        Write-Host "`nStarting preview server..." -ForegroundColor Yellow
        Write-Host "The app should open at: http://localhost:4173" -ForegroundColor Cyan
        Write-Host "`nPress Ctrl+C to stop the server" -ForegroundColor Gray
        Write-Host "-------------------------------------------------------" -ForegroundColor Gray
        
        npm run preview
    }
    
    "2" {
        Write-Host "`nStarting development server..." -ForegroundColor Green
        Write-Host "This runs the app in development mode with hot reload" -ForegroundColor Gray
        
        Set-Location "tori_ui_svelte"
        
        Write-Host "`nStarting dev server..." -ForegroundColor Yellow
        Write-Host "The app should open at: http://localhost:5173" -ForegroundColor Cyan
        Write-Host "`nPress Ctrl+C to stop the server" -ForegroundColor Gray
        Write-Host "-------------------------------------------------------" -ForegroundColor Gray
        
        npm run dev
    }
    
    "3" {
        Write-Host "`nStarting static file server..." -ForegroundColor Green
        Write-Host "This serves the built files directly" -ForegroundColor Gray
        
        $outputPath = Join-Path $RepoRoot "tori_ui_svelte\.svelte-kit\output\client"
        
        if (Test-Path $outputPath) {
            Write-Host "`nServing files from: $outputPath" -ForegroundColor Yellow
            
            # Check if Python is available
            $pythonCmd = $null
            if (Get-Command python -ErrorAction SilentlyContinue) {
                $pythonCmd = "python"
            } elseif (Get-Command python3 -ErrorAction SilentlyContinue) {
                $pythonCmd = "python3"
            }
            
            if ($pythonCmd) {
                Write-Host "Starting Python HTTP server..." -ForegroundColor Green
                Write-Host "The app should open at: http://localhost:8000" -ForegroundColor Cyan
                Write-Host "`nPress Ctrl+C to stop the server" -ForegroundColor Gray
                Write-Host "-------------------------------------------------------" -ForegroundColor Gray
                
                Set-Location $outputPath
                & $pythonCmd -m http.server 8000
            } else {
                Write-Host "Python not found. Install Python or use option 1 instead." -ForegroundColor Red
            }
        } else {
            Write-Host "Build output not found at: $outputPath" -ForegroundColor Red
        }
    }
    
    default {
        Write-Host "`nNo option selected. Try option 1 for the easiest test." -ForegroundColor Yellow
    }
}

exit 0