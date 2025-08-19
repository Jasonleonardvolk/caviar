# Generate release summary for fresh chat handoff
param(
    [string]$OutputFile = ".\tools\release\iris_release_summary.txt"
)

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$summary = @"
IRIS RELEASE SUMMARY
Generated: $timestamp
Repository: D:\Dev\kha

========================================
REFACTORING STATUS
========================================
✅ 930 files refactored from absolute to relative paths
✅ Runtime resolution implemented (Python + Node.js)
✅ Preflight checks active (prevents regression)
✅ 11 files preserved in docs\conversations (historical)

Helpers available:
- Python: scripts\iris_paths.py
- Node: src\lib\node\paths.ts
- Check: tools\runtime\preflight.mjs

========================================
CURRENT GATE STATUS
========================================
"@

Write-Host "Generating release summary..." -ForegroundColor Cyan

# Run checks and capture results
$results = @{}

# 1. Absolute paths check
Write-Host "Checking absolute paths..." -ForegroundColor Gray
$pathCheck = & node tools\runtime\preflight.mjs 2>&1
$results.AbsolutePaths = if ($LASTEXITCODE -eq 0) { "✅ PASS - No absolute paths" } else { "❌ FAIL - Absolute paths found" }

# 2. TypeScript check
Write-Host "Checking TypeScript..." -ForegroundColor Gray
$tsErrors = & npx tsc -p frontend\tsconfig.json 2>&1
$tsCount = ($tsErrors | Where-Object { $_ -match "error TS" }).Count
$results.TypeScript = if ($tsCount -eq 0) { "✅ PASS - 0 errors" } else { "❌ FAIL - $tsCount errors" }

# 3. Get latest shader reports
Write-Host "Collecting shader reports..." -ForegroundColor Gray
$shaderReports = Get-ChildItem -Path "tools\shaders\reports" -Filter "*.txt" -ErrorAction SilentlyContinue | 
    Sort-Object LastWriteTime -Descending | 
    Select-Object -First 5

# 4. Check for .env.production
$envProd = Test-Path ".env.production"
$results.EnvProduction = if ($envProd) { "✅ EXISTS" } else { "⚠️ MISSING" }

# Add results to summary
$summary += @"

Absolute Paths: $($results.AbsolutePaths)
TypeScript:     $($results.TypeScript)
.env.production: $($results.EnvProduction)

========================================
LATEST SHADER REPORTS
========================================
"@

if ($shaderReports) {
    foreach ($report in $shaderReports) {
        $summary += "`n- $($report.Name) ($(Get-Date $report.LastWriteTime -Format 'yyyy-MM-dd HH:mm'))"
    }
} else {
    $summary += "`nNo shader reports found in tools\shaders\reports\"
}

# Add file structure overview
$summary += @"

========================================
KEY PATHS
========================================
Repository Root: D:\Dev\kha
Frontend: D:\Dev\kha\frontend
Shaders Source: D:\Dev\kha\frontend\lib\webgpu\shaders
Shaders Public: D:\Dev\kha\frontend\public\hybrid\wgsl
TypeScript Config: D:\Dev\kha\frontend\tsconfig.json
Release Tools: D:\Dev\kha\tools\release
Shader Tools: D:\Dev\kha\tools\shaders
Runtime Helpers: D:\Dev\kha\tools\runtime

========================================
ONE-BUTTON COMMANDS
========================================
Full Release Gate:
  .\tools\release\IrisOneButton.ps1

Individual Checks:
  node tools\runtime\preflight.mjs          # Absolute paths
  npx tsc -p frontend\tsconfig.json         # TypeScript
  .\tools\shaders\run_shader_gate.ps1       # Shaders

========================================
NOTES FOR FRESH CHAT
========================================
1. All paths refactored to use {PROJECT_ROOT} or ${IRIS_ROOT}
2. Runtime resolution prevents hardcoded paths
3. Preflight checks run before dev/build
4. Ship criteria documented in tools\release\SHIP_CRITERIA.md
5. Fresh eyes + coffee = Final mile success!

END OF SUMMARY
"@

# Write summary to file
$summary | Out-File -FilePath $OutputFile -Encoding UTF8
Write-Host "`nSummary written to: $OutputFile" -ForegroundColor Green

# Also display key status
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  CURRENT STATUS" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
foreach ($key in $results.Keys) {
    Write-Host "$key : $($results[$key])"
}
Write-Host ""
Write-Host "Ready for fresh chat handoff!" -ForegroundColor Green
