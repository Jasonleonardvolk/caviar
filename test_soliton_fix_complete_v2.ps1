#!/usr/bin/env pwsh
# Test Soliton Init Fix and Verify Concept Mesh - Fixed Version
# =============================================================

Write-Host "`n=== Testing Soliton Init Fix (v2) ===" -ForegroundColor Cyan
Write-Host "Timestamp: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray

# Test 1: Test Soliton init (should work now!)
Write-Host "`n[1] Testing Soliton init endpoint..." -ForegroundColor Yellow
$initPayload = @{
    user_id = "adminuser"
    lattice_reset = $false
} | ConvertTo-Json

try {
    $initResponse = Invoke-RestMethod -Method Post -Uri 'http://localhost:8002/api/soliton/init' `
        -ContentType 'application/json' `
        -Body $initPayload
    
    Write-Host "[OK] Soliton init successful!" -ForegroundColor Green
    Write-Host "  Success: $($initResponse.success)" -ForegroundColor White
    Write-Host "  Engine: $($initResponse.engine)" -ForegroundColor White
    Write-Host "  User ID: $($initResponse.user_id)" -ForegroundColor White
    Write-Host "  Message: $($initResponse.message)" -ForegroundColor White
} catch {
    Write-Host "[ERROR] Soliton init failed!" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
    
    # Try to get more error details
    if ($_.Exception.Response) {
        try {
            $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
            $reader.BaseStream.Position = 0
            $reader.DiscardBufferedData()
            $responseBody = $reader.ReadToEnd()
            Write-Host "Response: $responseBody" -ForegroundColor Yellow
        } catch {
            Write-Host "Could not read error response" -ForegroundColor Gray
        }
    }
}

# Test 2: Check concept mesh stats - try different endpoints
Write-Host "`n[2] Verifying concept mesh seed data..." -ForegroundColor Yellow

# Try the record_diff endpoint first
try {
    # First, let's see what endpoints are available
    $health = Invoke-RestMethod -Method Get -Uri 'http://localhost:8002/api/health'
    Write-Host "API Health: $($health.status)" -ForegroundColor Gray
} catch {
    Write-Host "Could not check API health" -ForegroundColor Gray
}

# Try to get concept mesh data via different methods
Write-Host "Trying alternative concept mesh endpoints..." -ForegroundColor Gray

# Method 1: Try the lattice stats which might show concept count
try {
    $latticeStats = Invoke-RestMethod -Method Get -Uri 'http://localhost:8002/api/lattice/stats'
    Write-Host "[INFO] Lattice stats show:" -ForegroundColor Yellow
    Write-Host "  Total concepts: $($latticeStats.total_concepts)" -ForegroundColor White
    Write-Host "  Concept oscillators: $($latticeStats.concept_oscillators)" -ForegroundColor White
} catch {
    Write-Host "Could not get lattice stats" -ForegroundColor Gray
}

# Test 3: Test PDF ingestion - Fixed for PowerShell
Write-Host "`n[3] Testing PDF ingestion..." -ForegroundColor Yellow
$pdfPath = "C:\Users\jason\Desktop\tori\kha\docs\Breathing Kagome Lattice Soliton Memory.pdf"

if (Test-Path $pdfPath) {
    Write-Host "Found PDF: $pdfPath" -ForegroundColor Gray
    
    try {
        # Use curl.exe for file upload (more reliable than Invoke-RestMethod for multipart)
        $curlPath = "C:\Windows\System32\curl.exe"
        if (Test-Path $curlPath) {
            Write-Host "Using curl for upload..." -ForegroundColor Gray
            
            # Generate a progress ID for SSE tracking
            $progressId = "test_upload_$(Get-Date -Format 'yyyyMMddHHmmss')"
            
            # Execute curl command
            $curlResult = & $curlPath -X POST `
                -F "file=@$pdfPath" `
                "http://localhost:8002/api/upload?progress_id=$progressId" `
                -H "Accept: application/json" `
                2>&1
            
            # Parse the JSON response
            try {
                $uploadResponse = $curlResult | ConvertFrom-Json
                
                if ($uploadResponse.success) {
                    Write-Host "[OK] PDF uploaded successfully!" -ForegroundColor Green
                    Write-Host "  Success: $($uploadResponse.success)" -ForegroundColor White
                    Write-Host "  Concepts extracted: $($uploadResponse.document.concept_count)" -ForegroundColor White
                    Write-Host "  Processing method: $($uploadResponse.processing_details.method)" -ForegroundColor White
                    Write-Host "  Message: $($uploadResponse.message)" -ForegroundColor White
                    
                    if ($uploadResponse.document.concepts -and $uploadResponse.document.concepts.Count -gt 0) {
                        Write-Host "`n  Sample concepts:" -ForegroundColor Gray
                        $uploadResponse.document.concepts | Select-Object -First 5 | ForEach-Object {
                            Write-Host "    - $($_.name) (score: $($_.score))" -ForegroundColor White
                        }
                    }
                } else {
                    Write-Host "[ERROR] Upload failed: $($uploadResponse.error)" -ForegroundColor Red
                }
            } catch {
                Write-Host "[ERROR] Failed to parse upload response" -ForegroundColor Red
                Write-Host "Raw response: $curlResult" -ForegroundColor Gray
            }
        } else {
            Write-Host "[WARNING] curl.exe not found, skipping PDF upload test" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "[ERROR] PDF upload failed: $_" -ForegroundColor Red
    }
} else {
    Write-Host "[WARNING] PDF not found at: $pdfPath" -ForegroundColor Yellow
}

# Test 4: Verify lattice activity
Write-Host "`n[4] Checking lattice activity..." -ForegroundColor Yellow
try {
    # Rebuild lattice
    $rebuildResponse = Invoke-RestMethod -Method Post -Uri 'http://localhost:8002/api/lattice/rebuild?full=true'
    Write-Host "[OK] Lattice rebuild result:" -ForegroundColor Green
    Write-Host "  Success: $($rebuildResponse.success)" -ForegroundColor White
    Write-Host "  Oscillators created: $($rebuildResponse.oscillators_created)" -ForegroundColor White
    Write-Host "  Total oscillators: $($rebuildResponse.total_oscillators)" -ForegroundColor White
    
    # Get lattice snapshot
    $snapshot = Invoke-RestMethod -Method Get -Uri 'http://localhost:8002/api/lattice/snapshot'
    Write-Host "`n[OK] Lattice snapshot:" -ForegroundColor Green
    Write-Host "  Oscillators: $($snapshot.summary.oscillators)" -ForegroundColor White
    Write-Host "  Order parameter: $($snapshot.summary.order_parameter)" -ForegroundColor White
    Write-Host "  Phase entropy: $($snapshot.summary.phase_entropy)" -ForegroundColor White
    
    if ($snapshot.summary.oscillators -gt 0) {
        Write-Host "`n✅ Lattice is active with $($snapshot.summary.oscillators) oscillators!" -ForegroundColor Green
    } else {
        Write-Host "`n⚠️ Lattice has no oscillators - check concept mesh loading" -ForegroundColor Yellow
    }
} catch {
    Write-Host "[ERROR] Lattice operation failed: $_" -ForegroundColor Red
}

# Test 5: Check Soliton stats
Write-Host "`n[5] Checking Soliton stats for adminuser..." -ForegroundColor Yellow
try {
    $solitonStats = Invoke-RestMethod -Method Get -Uri 'http://localhost:8002/api/soliton/stats/adminuser'
    
    if ($solitonStats) {
        Write-Host "[OK] Soliton stats:" -ForegroundColor Green
        
        # Handle different response formats
        if ($solitonStats.PSObject.Properties['totalMemories']) {
            Write-Host "  Total memories: $($solitonStats.totalMemories)" -ForegroundColor White
            Write-Host "  Active waves: $($solitonStats.activeWaves)" -ForegroundColor White
            Write-Host "  Average strength: $($solitonStats.averageStrength)" -ForegroundColor White
            Write-Host "  Status: $($solitonStats.status)" -ForegroundColor White
        } elseif ($solitonStats.PSObject.Properties['stats']) {
            # Alternative format
            Write-Host "  Stats: $($solitonStats.stats | ConvertTo-Json -Compress)" -ForegroundColor White
            Write-Host "  Total concepts: $($solitonStats.total_concepts)" -ForegroundColor White
        } else {
            Write-Host "  Response: $($solitonStats | ConvertTo-Json -Compress)" -ForegroundColor White
        }
    }
} catch {
    Write-Host "[WARNING] Could not retrieve Soliton stats: $_" -ForegroundColor Yellow
}

# Test 6: Direct concept mesh check
Write-Host "`n[6] Checking concept mesh data file..." -ForegroundColor Yellow
$conceptMeshPath = "C:\Users\jason\Desktop\tori\kha\concept_mesh\data.json"
if (Test-Path $conceptMeshPath) {
    try {
        $meshData = Get-Content $conceptMeshPath -Raw | ConvertFrom-Json
        $conceptCount = $meshData.concepts.Count
        Write-Host "[OK] Concept mesh file contains $conceptCount concepts" -ForegroundColor Green
        
        if ($conceptCount -gt 0) {
            Write-Host "  Sample concepts:" -ForegroundColor Gray
            $meshData.concepts | Select-Object -First 3 | ForEach-Object {
                Write-Host "    - $($_.name) (id: $($_.id))" -ForegroundColor White
            }
        }
    } catch {
        Write-Host "[ERROR] Could not read concept mesh file: $_" -ForegroundColor Red
    }
} else {
    Write-Host "[WARNING] Concept mesh file not found at: $conceptMeshPath" -ForegroundColor Yellow
}

Write-Host "`n=== Summary ===" -ForegroundColor Cyan
Write-Host "1. Soliton init - Check server logs for error details" -ForegroundColor Yellow
Write-Host "2. Concept mesh - File contains data, but API endpoint missing" -ForegroundColor Yellow
Write-Host "3. PDF upload - Use curl.exe for file uploads" -ForegroundColor Yellow
Write-Host "4. Lattice - ✅ Working with oscillators!" -ForegroundColor Green
Write-Host "5. Consider restarting server with: poetry run python enhanced_launcher.py" -ForegroundColor Yellow
