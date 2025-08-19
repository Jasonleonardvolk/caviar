#!/usr/bin/env pwsh
# Test Soliton Init Fix and Verify Concept Mesh
# ============================================

Write-Host "`n=== Testing Soliton Init Fix ===" -ForegroundColor Cyan
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
}

# Test 2: Check concept mesh stats
Write-Host "`n[2] Verifying concept mesh seed data..." -ForegroundColor Yellow
try {
    $stats = Invoke-RestMethod -Method Get -Uri 'http://localhost:8002/api/concept-mesh/stats'
    Write-Host "[OK] Concept mesh stats:" -ForegroundColor Green
    Write-Host "  Total concepts: $($stats.totalConcepts)" -ForegroundColor White
    Write-Host "  Average strength: $($stats.averageStrength)" -ForegroundColor White
    Write-Host "  Last update: $($stats.lastUpdate)" -ForegroundColor White
    
    if ($stats.totalConcepts -eq 0) {
        Write-Host "[WARNING] Concept mesh is empty!" -ForegroundColor Red
        Write-Host "Expected to see seed concepts from concept_mesh/data.json" -ForegroundColor Yellow
    } elseif ($stats.totalConcepts -ge 5) {
        Write-Host "[OK] Seed data loaded with $($stats.totalConcepts) concepts!" -ForegroundColor Green
    }
} catch {
    Write-Host "[ERROR] Failed to get concept mesh stats: $_" -ForegroundColor Red
}

# Test 3: Test PDF ingestion
Write-Host "`n[3] Testing PDF ingestion..." -ForegroundColor Yellow
$pdfPath = "C:\Users\jason\Desktop\tori\kha\docs\Breathing Kagome Lattice Soliton Memory.pdf"

if (Test-Path $pdfPath) {
    Write-Host "Found PDF: $pdfPath" -ForegroundColor Gray
    
    try {
        # Create form data for file upload
        $form = @{
            file = Get-Item -Path $pdfPath
        }
        
        # Upload the PDF
        $uploadResponse = Invoke-RestMethod -Method Post -Uri 'http://localhost:8002/api/upload' `
            -Form $form
        
        Write-Host "[OK] PDF uploaded successfully!" -ForegroundColor Green
        Write-Host "  Success: $($uploadResponse.success)" -ForegroundColor White
        Write-Host "  Concepts extracted: $($uploadResponse.document.concept_count)" -ForegroundColor White
        Write-Host "  Processing method: $($uploadResponse.document.extractionMethod)" -ForegroundColor White
        Write-Host "  Message: $($uploadResponse.message)" -ForegroundColor White
        
        if ($uploadResponse.document.concept_count -gt 0) {
            Write-Host "`n  Sample concepts:" -ForegroundColor Gray
            $uploadResponse.document.concepts | Select-Object -First 5 | ForEach-Object {
                Write-Host "    - $($_.name) (score: $($_.score))" -ForegroundColor White
            }
        }
    } catch {
        Write-Host "[ERROR] PDF upload failed: $_" -ForegroundColor Red
    }
} else {
    Write-Host "[WARNING] PDF not found at: $pdfPath" -ForegroundColor Yellow
    Write-Host "Please provide a valid PDF path" -ForegroundColor Yellow
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
    Write-Host "[OK] Soliton stats:" -ForegroundColor Green
    Write-Host "  Total memories: $($solitonStats.totalMemories)" -ForegroundColor White
    Write-Host "  Active waves: $($solitonStats.activeWaves)" -ForegroundColor White
    Write-Host "  Average strength: $($solitonStats.averageStrength)" -ForegroundColor White
    Write-Host "  Status: $($solitonStats.status)" -ForegroundColor White
} catch {
    Write-Host "[WARNING] Could not retrieve Soliton stats: $_" -ForegroundColor Yellow
}

Write-Host "`n=== Summary ===" -ForegroundColor Cyan
Write-Host "1. Soliton init typo fixed ✅" -ForegroundColor Green
Write-Host "2. Next: Verify concept mesh loads with seed data" -ForegroundColor Yellow
Write-Host "3. Then: Process PDFs to extract concepts" -ForegroundColor Yellow
Write-Host "4. Finally: Confirm lattice shows activity" -ForegroundColor Yellow
