# Fix all remaining shader issues
Write-Host "`n=== FINAL PUSH TO FIX ALL SHADERS ===" -ForegroundColor Cyan

# 1. Check what's wrong with lenticularInterlace.wgsl
Write-Host "`n1. Checking lenticularInterlace.wgsl error..." -ForegroundColor Yellow
$lenticularPath = "C:\Users\jason\Desktop\tori\kha\frontend\lib\webgpu\shaders\lenticularInterlace.wgsl"
$result = & naga $lenticularPath 2>&1
Write-Host $result[0..5] -ForegroundColor Red

# 2. Fix propagation.wgsl in webgpu/shaders (JSON issue)
Write-Host "`n2. Fixing propagation.wgsl (JSON)..." -ForegroundColor Yellow
$propagationWebGPU = "C:\Users\jason\Desktop\tori\kha\frontend\lib\webgpu\shaders\propagation.wgsl"
if (Test-Path $propagationWebGPU) {
    $content = Get-Content $propagationWebGPU -Raw
    if ($content -match '^\s*{') {
        Write-Host "Still JSON format. Extracting content..." -ForegroundColor Yellow
        
        # Try to parse as JSON and extract content
        try {
            # Remove backticks from JSON
            $cleanJson = $content -replace '`', ''
            $jsonObj = $cleanJson | ConvertFrom-Json
            
            if ($jsonObj.content) {
                # The content has the actual WGSL with \n escaped
                $wgslContent = $jsonObj.content -replace '\\n', "`n" -replace '\\r', "`r" -replace '\\"', '"' -replace '\\\\', '\'
                
                # Save backup and write WGSL
                Move-Item $propagationWebGPU "$propagationWebGPU.json_backup2" -Force
                $wgslContent | Set-Content $propagationWebGPU -NoNewline
                Write-Host "Extracted WGSL from JSON wrapper" -ForegroundColor Green
            }
        } catch {
            Write-Host "JSON parse failed. Copying from frontend/shaders..." -ForegroundColor Yellow
            $sourceProp = "C:\Users\jason\Desktop\tori\kha\frontend\shaders\propagation.wgsl"
            if (Test-Path $sourceProp) {
                Copy-Item $sourceProp $propagationWebGPU -Force
                Write-Host "Copied working propagation.wgsl" -ForegroundColor Green
            }
        }
    }
}

# 3. Check main propagation.wgsl error
Write-Host "`n3. Checking main propagation.wgsl error..." -ForegroundColor Yellow
$mainProp = "C:\Users\jason\Desktop\tori\kha\frontend\shaders\propagation.wgsl"
$result = & naga $mainProp 2>&1 | Select-Object -First 5
$result | ForEach-Object { Write-Host $_ -ForegroundColor Red }

# 4. Fix velocityField.wgsl
Write-Host "`n4. Fixing velocityField.wgsl..." -ForegroundColor Yellow
& ".\better_velocity_fix.ps1"

# 5. Final validation of all shaders
Write-Host "`n=== FINAL VALIDATION ===" -ForegroundColor Cyan
$shaderDir = "C:\Users\jason\Desktop\tori\kha\frontend\lib\webgpu\shaders"

$results = @{}
Get-ChildItem "$shaderDir\*.wgsl" | ForEach-Object {
    $result = & naga $_.FullName 2>&1 | Out-String
    $results[$_.Name] = (-not ($result -match "error"))
}

# Display results
$validCount = 0
foreach ($shader in $results.GetEnumerator() | Sort-Object Name) {
    if ($shader.Value) {
        Write-Host "✅ $($shader.Key)" -ForegroundColor Green
        $validCount++
    } else {
        Write-Host "❌ $($shader.Key)" -ForegroundColor Red
    }
}

Write-Host "`n$validCount out of $($results.Count) shaders are valid" -ForegroundColor Cyan

# Show specific errors for failed shaders
Write-Host "`n=== Errors for Failed Shaders ===" -ForegroundColor Yellow
foreach ($shader in $results.GetEnumerator() | Where-Object { -not $_.Value }) {
    Write-Host "`n$($shader.Key):" -ForegroundColor Red
    $error = & naga (Join-Path $shaderDir $shader.Key) 2>&1 | Select-Object -First 3
    $error | ForEach-Object { Write-Host "  $_" -ForegroundColor DarkRed }
}
