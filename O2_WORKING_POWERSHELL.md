# O2 Files Download - Working PowerShell Command

## Copy and paste this command directly into PowerShell:

```powershell
$s="${IRIS_ROOT}"; $z="C:\Users\jason\Desktop\o2.zip"; $d=Get-Date "2025-07-03"; $t=New-Item -ItemType Directory -Path "$env:TEMP\o2_$(Get-Date -Format 'yyyyMMddHHmmss')" -Force; Get-ChildItem -Path $s -Recurse -File | Where-Object {$_.LastWriteTime -ge $d -and $_.FullName -notmatch '\\(\.git|node_modules|\.yarn|__pycache__|\.pytest_cache|\.turbo|dist|build)\\' -and $_.Extension -match '\.(py|js|ts|jsx|tsx|rs|yaml|yml|json|md|bat|sh|ps1|html|css|svelte)$'} | ForEach-Object {$r=$_.FullName.Substring($s.Length+1); $p=Join-Path $t $r; $pd=Split-Path $p -Parent; if(-not(Test-Path $pd)){New-Item -ItemType Directory -Path $pd -Force | Out-Null}; Copy-Item $_.FullName -Destination $p -Force}; Compress-Archive -Path "$t\*" -DestinationPath $z -Force; Remove-Item -Path $t -Recurse -Force; Write-Host "Created $z" -ForegroundColor Green
```

## Alternative: Simpler approach with fewer filters

If the above still has issues, use this simpler version:

```powershell
$sourceDir = "${IRIS_ROOT}"
$outputZip = "C:\Users\jason\Desktop\o2.zip"
$tempDir = "$env:TEMP\o2_temp"

# Create temp directory
New-Item -ItemType Directory -Path $tempDir -Force | Out-Null

# Copy files modified since July 3, 2025
Get-ChildItem -Path $sourceDir -Recurse -File | Where-Object {
    $_.LastWriteTime -ge (Get-Date "2025-07-03") -and
    $_.DirectoryName -notlike "*\.git*" -and
    $_.DirectoryName -notlike "*\node_modules*" -and
    $_.DirectoryName -notlike "*\__pycache__*" -and
    ($_.Extension -eq ".py" -or 
     $_.Extension -eq ".js" -or 
     $_.Extension -eq ".ts" -or 
     $_.Extension -eq ".jsx" -or 
     $_.Extension -eq ".tsx" -or 
     $_.Extension -eq ".rs" -or 
     $_.Extension -eq ".yaml" -or 
     $_.Extension -eq ".yml" -or 
     $_.Extension -eq ".json" -or 
     $_.Extension -eq ".md" -or 
     $_.Extension -eq ".bat" -or 
     $_.Extension -eq ".sh" -or 
     $_.Extension -eq ".ps1" -or 
     $_.Extension -eq ".html" -or 
     $_.Extension -eq ".css" -or 
     $_.Extension -eq ".svelte")
} | ForEach-Object {
    $relativePath = $_.FullName.Substring($sourceDir.Length + 1)
    $destPath = Join-Path $tempDir $relativePath
    $destDir = Split-Path $destPath -Parent
    
    if (-not (Test-Path $destDir)) {
        New-Item -ItemType Directory -Path $destDir -Force | Out-Null
    }
    
    Copy-Item $_.FullName -Destination $destPath -Force
}

# Create zip file
Compress-Archive -Path "$tempDir\*" -DestinationPath $outputZip -Force

# Clean up
Remove-Item -Path $tempDir -Recurse -Force

Write-Host "Successfully created o2.zip" -ForegroundColor Green
```

## Even Simpler: Just the O2 files we created

If you just want the 26 files we specifically created/modified:

```powershell
$kha = "${IRIS_ROOT}"
$temp = "$env:TEMP\o2_specific"
$zip = "C:\Users\jason\Desktop\o2.zip"

New-Item -ItemType Directory -Path $temp -Force | Out-Null

# List of specific files
$files = @(
    "python\core\hot_swap_laplacian.py",
    "python\core\memory_crystallization.py",
    "python\core\nightly_growth_engine.py",
    "python\core\blowup_harness.py",
    "python\core\topology_policy.py",
    "python\core\soliton_memory_integration_modified.py",
    "python\core\lattice_evolution_runner_modified.py",
    "concept-mesh\src\lattice_topology.rs",
    "concept-mesh\src\comfort_analysis.rs",
    "concept-mesh\src\soliton_memory_modified.rs",
    "frontend\ghost\GhostPhaseBalancer.ts",
    "tests\test_hot_swap_laplacian.py",
    "tests\test_dark_solitons.py",
    "tests\test_topology_morphing.py",
    "tests\test_memory_consolidation.py",
    "tests\run_integration_tests.py",
    "conf\soliton_memory_config.yaml",
    "conf\lattice_config_updated.yaml",
    "docs\HOT_SWAP_LAPLACIAN_GUIDE.md",
    "docs\SOLITON_ARCHITECTURE.md",
    "examples\hot_swap_o2_demo.py",
    "benchmarks\benchmark_soliton_performance.py",
    "integrate_hot_swap.py",
    "SOLITON_SELF_ORGANIZING_PATCH_BUNDLE.md",
    "SOLITON_SELF_ORGANIZING_PATCH_BUNDLE_PART2.md",
    "FILES_CREATED_AND_MODIFIED_SUMMARY.md"
)

foreach ($file in $files) {
    $src = Join-Path $kha $file
    if (Test-Path $src) {
        $dest = Join-Path $temp $file
        $destDir = Split-Path $dest -Parent
        if (-not (Test-Path $destDir)) {
            New-Item -ItemType Directory -Path $destDir -Force | Out-Null
        }
        Copy-Item $src -Destination $dest -Force
    }
}

Compress-Archive -Path "$temp\*" -DestinationPath $zip -Force
Remove-Item -Path $temp -Recurse -Force

Write-Host "Created o2.zip with O2 implementation files" -ForegroundColor Green
```

This last option will create a zip with just the 26 files from our O2 implementation work.
