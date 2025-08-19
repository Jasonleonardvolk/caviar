# Free TORI Bridge Ports Script
# Kills processes on ports 8765 and 8766

Write-Host "🔍 Checking ports 8765 and 8766..." -ForegroundColor Cyan

# Function to kill process on specific port
function Kill-ProcessOnPort {
    param($Port)
    
    Write-Host "`nChecking port $Port..." -ForegroundColor Yellow
    
    # Get process using the port
    $netstat = netstat -ano | Select-String ":$Port\s+.*LISTENING"
    
    if ($netstat) {
        foreach ($line in $netstat) {
            # Extract PID from netstat output
            $parts = $line -split '\s+'
            $pid = $parts[-1]
            
            if ($pid -match '^\d+$') {
                try {
                    $process = Get-Process -Id $pid -ErrorAction SilentlyContinue
                    if ($process) {
                        Write-Host "Found process '$($process.Name)' (PID: $pid) on port $Port" -ForegroundColor Red
                        Write-Host "Killing process..." -ForegroundColor Yellow
                        Stop-Process -Id $pid -Force
                        Write-Host "✅ Process killed successfully" -ForegroundColor Green
                    }
                } catch {
                    Write-Host "⚠️ Could not kill process $pid : $_" -ForegroundColor Red
                }
            }
        }
    } else {
        Write-Host "✅ Port $Port is free" -ForegroundColor Green
    }
}

# Kill processes on both ports
Kill-ProcessOnPort -Port 8765
Kill-ProcessOnPort -Port 8766

Write-Host "`n✅ Port cleanup complete!" -ForegroundColor Green
Write-Host "You can now start the audio and concept bridges without port conflicts." -ForegroundColor Cyan
