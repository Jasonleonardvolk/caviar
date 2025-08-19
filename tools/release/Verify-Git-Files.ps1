param(
    [string]$ProjectRoot = "D:\Dev\kha"
)

$ErrorActionPreference = "Stop"

Write-Host "Checking if all claimed files exist in Git HEAD..." -ForegroundColor Cyan
Write-Host ""

Push-Location $ProjectRoot

$files = @(
    "frontend/src/lib/device/capabilities.ts",
    "frontend/src/lib/stores/userPlan.ts",
    "frontend/static/config/plans.json",
    "frontend/src/lib/utils/exportVideo.ts",
    "frontend/src/lib/components/HologramRecorder.svelte",
    "frontend/src/routes/api/billing/checkout/+server.ts",
    "frontend/src/lib/components/PricingTable.svelte",
    "frontend/src/lib/hologram/engineShim.ts",
    "frontend/src/routes/hologram/+page.svelte",
    "frontend/src/routes/pricing/+page.svelte",
    "frontend/src/routes/api/billing/portal/+server.ts",
    "frontend/src/routes/api/templates/export/+server.ts",
    "frontend/src/routes/api/templates/upload/+server.ts",
    "frontend/src/routes/api/templates/file/[name]/+server.ts",
    "frontend/src/routes/templates/+page.svelte",
    "frontend/src/routes/templates/+page.server.ts",
    "frontend/src/routes/templates/upload/+page.svelte",
    "frontend/src/routes/publish/+page.svelte",
    "frontend/src/routes/publish/+page.server.ts",
    "frontend/src/routes/health/+server.ts",
    "frontend/src/routes/health/+page.svelte",
    "frontend/src/routes/health/+page.server.ts",
    "frontend/src/lib/health/checks.server.ts",
    "tools/exporters/glb-from-conceptmesh.ts",
    "tools/exporters/encode-ktx2.ps1",
    "tools/release/sync-plans.mjs",
    "tools/release/Sync-Plans.ps1",
    "tools/release/build-templates-index.mjs",
    "tools/release/Build-Templates-Index.ps1"
)

$gitFiles = git ls-tree -r --name-only HEAD

$found = 0
$missing = @()

foreach ($file in $files) {
    $normalized = $file.Replace("/", "\")
    if ($gitFiles -contains $file -or $gitFiles -contains $normalized) {
        Write-Host "[✓] $file" -ForegroundColor Green
        $found++
    } else {
        Write-Host "[X] $file" -ForegroundColor Red
        $missing += $file
    }
}

Pop-Location

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Results: $found/$($files.Count) files in Git HEAD" -ForegroundColor $(if ($missing.Count -eq 0) { "Green" } else { "Yellow" })

if ($missing.Count -gt 0) {
    Write-Host ""
    Write-Host "Missing files need to be added and committed:" -ForegroundColor Yellow
    $missing | ForEach-Object { Write-Host "  git add $_" -ForegroundColor Gray }
    Write-Host "  git commit -m 'Add missing monetization files'" -ForegroundColor Gray
    exit 1
}

exit 0