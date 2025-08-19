# O2 Files Download Command

## Quick One-Liner PowerShell Command

Copy and paste this entire command into PowerShell:

```powershell
$s="${IRIS_ROOT}";$z="C:\Users\jason\Desktop\o2.zip";$d=Get-Date "2025-07-03";$t=New-Item -ItemType Directory -Path "$env:TEMP\o2_$(Get-Date -Format 'yyyyMMddHHmmss')" -Force;Get-ChildItem -Path $s -Recurse -File|Where-Object{$_.LastWriteTime -ge $d -and $_.FullName -notmatch "\\(\.git|node_modules|\.yarn|__pycache__|\.pytest_cache|\.turbo|dist|build)\\" -and $_.Extension -match "\.(py|js|ts|jsx|tsx|rs|yaml|yml|json|md|bat|sh|ps1|html|css|svelte)$"}|ForEach-Object{$r=$_.FullName.Substring($s.Length+1);$p=Join-Path $t $r;$pd=Split-Path $p -Parent;if(-not(Test-Path $pd)){New-Item -ItemType Directory -Path $pd -Force|Out-Null};Copy-Item $_.FullName -Destination $p -Force};Compress-Archive -Path "$t\*" -DestinationPath $z -Force;Remove-Item -Path $t -Recurse -Force;Write-Host "Created $z" -ForegroundColor Green
```

## Alternative: Step-by-Step Commands

If the one-liner is too long, you can run these commands sequentially:

```powershell
# Step 1: Set variables
$sourceDir = "${IRIS_ROOT}"
$outputZip = "C:\Users\jason\Desktop\o2.zip"
$startDate = Get-Date "2025-07-03"

# Step 2: Create temp directory
$tempDir = New-Item -ItemType Directory -Path "$env:TEMP\o2_filtered" -Force

# Step 3: Get and copy files
Get-ChildItem -Path $sourceDir -Recurse -File | Where-Object {
    $_.LastWriteTime -ge $startDate -and
    $_.FullName -notmatch "\\\.git\\" -and
    $_.FullName -notmatch "\\node_modules\\" -and
    $_.Extension -match "\.(py|js|ts|jsx|tsx|rs|yaml|yml|json|md|bat|sh|ps1|html|css|svelte)$"
} | ForEach-Object {
    $relativePath = $_.FullName.Substring($sourceDir.Length + 1)
    $destPath = Join-Path $tempDir $relativePath
    $destDir = Split-Path $destPath -Parent
    if (-not (Test-Path $destDir)) {
        New-Item -ItemType Directory -Path $destDir -Force | Out-Null
    }
    Copy-Item $_.FullName -Destination $destPath -Force
}

# Step 4: Create zip and cleanup
Compress-Archive -Path "$tempDir\*" -DestinationPath $outputZip -Force
Remove-Item -Path $tempDir -Recurse -Force
Write-Host "Created o2.zip" -ForegroundColor Green
```

## Files That Will Be Included

Based on the July 3rd, 2025 modification date, the zip will include:

### Core Implementation (26 files from our work)
- All Python modules in `python/core/`
- All Rust modules in `concept-mesh/src/`
- TypeScript component in `frontend/ghost/`
- Test files in `tests/`
- Configuration files in `conf/`
- Documentation and examples

### Additional Recent Files
Any other programming files (.py, .js, .ts, .rs, .yaml, etc.) modified since July 3rd, 2025 will also be included.

The zip file will preserve the directory structure for easy deployment.
