# Find Programming Files Modified Since July 3rd, 2025

## PowerShell Command to Create o2.zip

```powershell
# Set variables
$sourceDir = "${IRIS_ROOT}"
$outputZip = "C:\Users\jason\Desktop\o2.zip"
$startDate = Get-Date "2025-07-03"

# Get all files modified since July 3rd, 2025, excluding non-programming files
$files = Get-ChildItem -Path $sourceDir -Recurse -File | Where-Object {
    $_.LastWriteTime -ge $startDate -and
    $_.FullName -notmatch "\\\.git\\" -and
    $_.FullName -notmatch "\\node_modules\\" -and
    $_.FullName -notmatch "\\\.yarn\\" -and
    $_.FullName -notmatch "\\__pycache__\\" -and
    $_.FullName -notmatch "\\\.pytest_cache\\" -and
    $_.FullName -notmatch "\\\.turbo\\" -and
    $_.FullName -notmatch "\\dist\\" -and
    $_.FullName -notmatch "\\build\\" -and
    $_.Extension -match "\.(py|js|ts|jsx|tsx|rs|yaml|yml|json|md|bat|sh|ps1|html|css|svelte)$"
}

# Create temp directory for filtered files
$tempDir = New-Item -ItemType Directory -Path "$env:TEMP\o2_filtered_$(Get-Date -Format 'yyyyMMddHHmmss')" -Force

# Copy files preserving directory structure
foreach ($file in $files) {
    $relativePath = $file.FullName.Substring($sourceDir.Length + 1)
    $destPath = Join-Path $tempDir $relativePath
    $destDir = Split-Path $destPath -Parent
    
    if (-not (Test-Path $destDir)) {
        New-Item -ItemType Directory -Path $destDir -Force | Out-Null
    }
    
    Copy-Item $file.FullName -Destination $destPath -Force
}

# Create the zip file
Compress-Archive -Path "$tempDir\*" -DestinationPath $outputZip -Force

# Clean up temp directory
Remove-Item -Path $tempDir -Recurse -Force

Write-Host "Created $outputZip with $($files.Count) files modified since July 3rd, 2025" -ForegroundColor Green
```

## Files Created/Modified Since July 3rd, 2025

### Python Core Files
- `python/core/hot_swap_laplacian.py`
- `python/core/memory_crystallization.py`
- `python/core/nightly_growth_engine.py`
- `python/core/blowup_harness.py`
- `python/core/topology_policy.py`
- `python/core/soliton_memory_integration_modified.py`
- `python/core/lattice_evolution_runner_modified.py`

### Rust Files
- `concept-mesh/src/lattice_topology.rs`
- `concept-mesh/src/comfort_analysis.rs`
- `concept-mesh/src/soliton_memory_modified.rs`

### TypeScript/Frontend Files
- `frontend/ghost/GhostPhaseBalancer.ts`

### Test Files
- `tests/test_hot_swap_laplacian.py`
- `tests/test_dark_solitons.py`
- `tests/test_topology_morphing.py`
- `tests/test_memory_consolidation.py`
- `tests/run_integration_tests.py`

### Configuration Files
- `conf/soliton_memory_config.yaml`
- `conf/lattice_config_updated.yaml`

### Documentation Files
- `docs/HOT_SWAP_LAPLACIAN_GUIDE.md`
- `docs/SOLITON_ARCHITECTURE.md`
- `SOLITON_SELF_ORGANIZING_PATCH_BUNDLE.md`
- `SOLITON_SELF_ORGANIZING_PATCH_BUNDLE_PART2.md`
- `FILES_CREATED_AND_MODIFIED_SUMMARY.md`

### Examples & Tools
- `examples/hot_swap_o2_demo.py`
- `benchmarks/benchmark_soliton_performance.py`
- `integrate_hot_swap.py`
