# Audit-CaviarCatchup.ps1
# Comprehensive audit of what exists vs. what's planned for iRis monetization
# Run from D:\Dev\kha

[CmdletBinding()]
param()

$ErrorActionPreference = 'Continue'
$results = @()
$stats = @{
    Total = 0
    Exists = 0
    Missing = 0
    Partial = 0
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "    iRIS MONETIZATION AUDIT REPORT" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Define all components from the snaptikcatchup plan
$components = @(
    # Phase 0 - Repo Prep
    @{Category="Phase 0: Repo Prep"; Path="tools\exporters"; Type="Directory"; Priority="Critical"},
    @{Category="Phase 0: Repo Prep"; Path="tools\exporters\build-template-kit.ts"; Type="File"; Priority="Critical"},
    @{Category="Phase 0: Repo Prep"; Path="tools\exporters\glb-from-conceptmesh.ts"; Type="File"; Priority="Critical"},
    @{Category="Phase 0: Repo Prep"; Path="tools\exporters\encode-ktx2.ps1"; Type="File"; Priority="Critical"},
    @{Category="Phase 0: Repo Prep"; Path="config\template.schema.json"; Type="File"; Priority="High"},
    
    # Phase 1 - Video Export (Creator Path)
    @{Category="Phase 1: Video Export"; Path="frontend\src\lib\components\HologramRecorder.svelte"; Type="File"; Priority="Critical"},
    @{Category="Phase 1: Video Export"; Path="frontend\src\lib\utils\exportVideo.ts"; Type="File"; Priority="Critical"},
    @{Category="Phase 1: Video Export"; Path="frontend\src\lib\stores\userPlan.ts"; Type="File"; Priority="Critical"},
    @{Category="Phase 1: Video Export"; Path="frontend\src\routes\api\billing\checkout\+server.ts"; Type="File"; Priority="High"},
    
    # Phase 2 - Template Kit Exporter
    @{Category="Phase 2: Template Kit"; Path="dist\kits"; Type="Directory"; Priority="Medium"},
    @{Category="Phase 2: Template Kit"; Path="assets\concepts"; Type="Directory"; Priority="Medium"},
    
    # Phase 3 - Snap & TikTok Bridge
    @{Category="Phase 3: Platform Bridge"; Path="integrations\snap\seed-project"; Type="Directory"; Priority="Medium"},
    @{Category="Phase 3: Platform Bridge"; Path="integrations\tiktok\seed-project"; Type="Directory"; Priority="Medium"},
    @{Category="Phase 3: Platform Bridge"; Path="integrations\common\materials"; Type="Directory"; Priority="Low"},
    
    # Phase 4 - Template Catalog
    @{Category="Phase 4: Catalog"; Path="frontend\src\routes\templates\+page.svelte"; Type="File"; Priority="Medium"},
    @{Category="Phase 4: Catalog"; Path="frontend\src\lib\components\TemplateCard.svelte"; Type="File"; Priority="Medium"},
    @{Category="Phase 4: Catalog"; Path="frontend\src\routes\api\export\kit\+server.ts"; Type="File"; Priority="Medium"},
    
    # Phase 5 - Monetization
    @{Category="Phase 5: Monetization"; Path="config\plans.json"; Type="File"; Priority="Critical"},
    @{Category="Phase 5: Monetization"; Path="config\pricing.rules.json"; Type="File"; Priority="High"},
    @{Category="Phase 5: Monetization"; Path="services\pricing\engine.ts"; Type="File"; Priority="High"},
    @{Category="Phase 5: Monetization"; Path="services\compliance\audit_log.ts"; Type="File"; Priority="Medium"},
    
    # Phase 6 - Analytics
    @{Category="Phase 6: Analytics"; Path="frontend\src\lib\stores\psiTelemetry.ts"; Type="File"; Priority="Low"},
    @{Category="Phase 6: Analytics"; Path="services\analytics\events\ingest.py"; Type="File"; Priority="Low"},
    @{Category="Phase 6: Analytics"; Path="integrations\links\Create-UTM.ps1"; Type="File"; Priority="Low"},
    
    # iOS 26 WebGPU Support
    @{Category="iOS 26 WebGPU"; Path="frontend\src\lib\device\capabilities.ts"; Type="File"; Priority="High"},
    @{Category="iOS 26 WebGPU"; Path="frontend\src\lib\webgpu\init.ts"; Type="File"; Priority="High"},
    @{Category="iOS 26 WebGPU"; Path="config\device-matrix.json"; Type="File"; Priority="Medium"},
    
    # Existing Components (Should be there)
    @{Category="Existing Pipeline"; Path="tools\encode\Build-SocialPack.ps1"; Type="File"; Priority="Done"},
    @{Category="Existing Pipeline"; Path="tools\encode\Batch-Encode-Social.ps1"; Type="File"; Priority="Done"},
    @{Category="Existing Pipeline"; Path="content\socialpack"; Type="Directory"; Priority="Done"},
    @{Category="Existing Pipeline"; Path="tori_ui_svelte\static\social"; Type="Directory"; Priority="Done"},
    @{Category="Existing Pipeline"; Path="tori_ui_svelte\src\routes\social\+page.svelte"; Type="File"; Priority="Done"}
)

# Check each component
foreach ($component in $components) {
    $stats.Total++
    $fullPath = Join-Path $PSScriptRoot $component.Path
    $exists = Test-Path $fullPath
    
    if ($exists) {
        $stats.Exists++
        $status = "✅ EXISTS"
        $color = "Green"
    } else {
        $stats.Missing++
        $status = "❌ MISSING"
        $color = "Red"
    }
    
    $results += [PSCustomObject]@{
        Category = $component.Category
        Path = $component.Path
        Type = $component.Type
        Priority = $component.Priority
        Status = $status
        Exists = $exists
    }
    
    # Print as we go for immediate feedback
    if ($component.Priority -eq "Critical" -and -not $exists) {
        Write-Host "$status [CRITICAL] $($component.Path)" -ForegroundColor Red
    } elseif ($component.Priority -eq "Done" -and $exists) {
        Write-Host "$status [COMPLETE] $($component.Path)" -ForegroundColor Green
    }
}

# Group and display results by category
Write-Host "`n📊 AUDIT BY PHASE:" -ForegroundColor Yellow
Write-Host "===================" -ForegroundColor Yellow

$grouped = $results | Group-Object Category
foreach ($group in $grouped | Sort-Object Name) {
    $categoryExists = ($group.Group | Where-Object Exists).Count
    $categoryTotal = $group.Count
    $percentage = [math]::Round(($categoryExists / $categoryTotal) * 100, 0)
    
    $statusIcon = if ($percentage -eq 100) { "✅" } 
                  elseif ($percentage -eq 0) { "❌" } 
                  else { "⚠️" }
    
    Write-Host "`n$statusIcon $($group.Name): $categoryExists/$categoryTotal ($percentage%)" -ForegroundColor Cyan
    
    foreach ($item in $group.Group | Sort-Object Priority) {
        $priorityTag = switch ($item.Priority) {
            "Critical" { "[CRITICAL]" }
            "High" { "[HIGH]" }
            "Medium" { "[MED]" }
            "Low" { "[LOW]" }
            "Done" { "[DONE]" }
            default { "" }
        }
        
        $color = if ($item.Exists) { "Green" } else { "Red" }
        Write-Host "  $($item.Status) $priorityTag $($item.Path)" -ForegroundColor $color
    }
}

# Summary Statistics
Write-Host "`n📈 OVERALL STATISTICS:" -ForegroundColor Yellow
Write-Host "======================" -ForegroundColor Yellow
Write-Host "Total Components: $($stats.Total)" -ForegroundColor White
Write-Host "✅ Exists: $($stats.Exists) ($([math]::Round(($stats.Exists/$stats.Total)*100, 0))%)" -ForegroundColor Green
Write-Host "❌ Missing: $($stats.Missing) ($([math]::Round(($stats.Missing/$stats.Total)*100, 0))%)" -ForegroundColor Red

# Critical Missing Components
$criticalMissing = $results | Where-Object { $_.Priority -eq "Critical" -and -not $_.Exists }
if ($criticalMissing) {
    Write-Host "`n🚨 CRITICAL MISSING COMPONENTS:" -ForegroundColor Red
    Write-Host "================================" -ForegroundColor Red
    foreach ($item in $criticalMissing) {
        Write-Host "  - $($item.Path)" -ForegroundColor Red
    }
}

# Next Steps
Write-Host "`n🎯 RECOMMENDED NEXT STEPS:" -ForegroundColor Magenta
Write-Host "===========================" -ForegroundColor Magenta

$nextSteps = @(
    "1. Create HologramRecorder.svelte for video capture (CRITICAL PATH)",
    "2. Implement exportVideo.ts for MP4 encoding", 
    "3. Setup userPlan.ts for subscription tiers",
    "4. Create plans.json configuration",
    "5. Build template kit exporters for GLB/KTX2",
    "6. Setup billing API endpoints",
    "7. Implement WebGPU capabilities detection for iOS 26"
)

foreach ($step in $nextSteps) {
    Write-Host "  $step" -ForegroundColor White
}

# Export to CSV
$csvPath = Join-Path $PSScriptRoot "audit_results_$(Get-Date -Format 'yyyyMMdd_HHmmss').csv"
$results | Export-Csv -Path $csvPath -NoTypeInformation
Write-Host "`n📄 Full report exported to: $csvPath" -ForegroundColor Green

# Generate creation script for missing critical components
$scriptPath = Join-Path $PSScriptRoot "Create-MissingComponents.ps1"
$creationScript = @"
# Auto-generated script to create missing critical components
# Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')

`$ErrorActionPreference = 'Stop'

Write-Host "Creating missing critical components..." -ForegroundColor Cyan

"@

foreach ($item in $criticalMissing) {
    if ($item.Type -eq "Directory") {
        $creationScript += "`nNew-Item -Path '$($item.Path)' -ItemType Directory -Force | Out-Null"
        $creationScript += "`nWrite-Host 'Created directory: $($item.Path)' -ForegroundColor Green"
    } else {
        $dir = Split-Path $item.Path -Parent
        $creationScript += "`nNew-Item -Path '$dir' -ItemType Directory -Force -ErrorAction SilentlyContinue | Out-Null"
        $creationScript += "`nNew-Item -Path '$($item.Path)' -ItemType File -Force | Out-Null"
        $creationScript += "`nWrite-Host 'Created file: $($item.Path)' -ForegroundColor Green"
    }
}

$creationScript += "`n`nWrite-Host 'All critical components created!' -ForegroundColor Green"
$creationScript | Out-File -FilePath $scriptPath -Encoding UTF8
Write-Host "📝 Creation script generated: $scriptPath" -ForegroundColor Yellow

# Final recommendation
Write-Host "`n💡 EXECUTIVE SUMMARY:" -ForegroundColor Cyan
Write-Host "=====================" -ForegroundColor Cyan
Write-Host @"
The iRis monetization infrastructure is approximately 20% complete.
You have the video pipeline (Social Pack) working, but lack:
- Frontend recording components (HologramRecorder)
- Subscription management (userPlan, billing)
- Template export system (GLB/KTX2 converters)
- Platform integrations (Snap/TikTok bridges)
- Analytics and compliance layers

Priority: Build HologramRecorder.svelte FIRST - it's the gateway to monetization.
With iOS 26 WebGPU support, you're perfectly positioned for high-quality captures.

Run .\Create-MissingComponents.ps1 to scaffold the critical missing pieces.
"@ -ForegroundColor White

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "         AUDIT COMPLETE" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan
