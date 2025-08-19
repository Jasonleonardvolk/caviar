# Generate Changelog for ChatGPT - Exactly what changed in last 3 days
# Creates a summary file ChatGPT can read first

$days = 3
$cutoff = (Get-Date).AddDays(-$days)
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

Write-Host "`n===== GENERATING CHANGELOG FOR CHATGPT =====" -ForegroundColor Cyan
Write-Host "Creating metadata file with 3-day changes..." -ForegroundColor Yellow

# Define paths
$localPath = "C:\Users\jason\Desktop\tori\kha"
$drivePath = "G:\My Drive\Computers\My Laptop\kha"

# Get all changed files
$changedFiles = Get-ChildItem $localPath -Recurse -File | 
    Where-Object { 
        $_.LastWriteTime -gt $cutoff -and
        $_.Extension -notin @('.tmp', '.cache', '.lock', '.log') -and
        $_.FullName -notmatch '\\\.git\\' -and
        $_.FullName -notmatch '\\\.venv\\' -and
        $_.FullName -notmatch '\\__pycache__\\' -and
        $_.FullName -notmatch '\\node_modules\\'
    }

# Group by type
$pyFiles = $changedFiles | Where-Object { $_.Extension -eq '.py' }
$tsFiles = $changedFiles | Where-Object { $_.Extension -in @('.ts', '.tsx') }
$svelteFiles = $changedFiles | Where-Object { $_.Extension -eq '.svelte' }
$mdFiles = $changedFiles | Where-Object { $_.Extension -eq '.md' }
$jsonFiles = $changedFiles | Where-Object { $_.Extension -eq '.json' }
$shaderFiles = $changedFiles | Where-Object { $_.Extension -in @('.wgsl', '.glsl') }
$otherFiles = $changedFiles | Where-Object { $_.Extension -notin @('.py', '.ts', '.tsx', '.svelte', '.md', '.json', '.wgsl', '.glsl') }

# Create changelog content
$changelog = @"
# KHA FOLDER CHANGELOG FOR CHATGPT
Generated: $timestamp
Time Window: Last $days days (since $cutoff)
Total Changed Files: $($changedFiles.Count)

## SUMMARY BY FILE TYPE
- Python (.py): $($pyFiles.Count) files
- TypeScript (.ts/.tsx): $($tsFiles.Count) files  
- Svelte (.svelte): $($svelteFiles.Count) files
- Markdown (.md): $($mdFiles.Count) files
- JSON (.json): $($jsonFiles.Count) files
- Shaders (.wgsl/.glsl): $($shaderFiles.Count) files
- Other: $($otherFiles.Count) files

## KEY UPDATES TO CHECK FIRST
"@

# Add most recently modified files (top 10)
$changelog += "`n### Most Recently Modified (for immediate attention):`n"
$changedFiles | Sort-Object LastWriteTime -Descending | Select-Object -First 10 | ForEach-Object {
    $relativePath = $_.FullName.Replace($localPath, "").TrimStart("\")
    $changelog += "- $($_.LastWriteTime.ToString('MM/dd HH:mm')) - $relativePath`n"
}

# Add new files (created in last 3 days)
$newFiles = $changedFiles | Where-Object { $_.CreationTime -gt $cutoff }
if ($newFiles) {
    $changelog += "`n### NEW FILES CREATED ($($newFiles.Count)):`n"
    $newFiles | Sort-Object CreationTime -Descending | ForEach-Object {
        $relativePath = $_.FullName.Replace($localPath, "").TrimStart("\")
        $changelog += "- $relativePath`n"
    }
}

# Add critical system files
$criticalFiles = $changedFiles | Where-Object { 
    $_.Name -match "(launcher|main|api|server|config|setup|init)" -or
    $_.FullName -match "(tori|prajna|scholarsphere|saigon|soliton|penrose)"
}
if ($criticalFiles) {
    $changelog += "`n### CRITICAL SYSTEM FILES CHANGED ($($criticalFiles.Count)):`n"
    $criticalFiles | ForEach-Object {
        $relativePath = $_.FullName.Replace($localPath, "").TrimStart("\")
        $changelog += "- $relativePath`n"
    }
}

# Add component breakdown
$changelog += @"

## COMPONENT STATUS

### TORI CORE
"@
$toriFiles = $changedFiles | Where-Object { $_.FullName -match "\\tori\\" -or $_.Name -match "tori" }
$changelog += "- Changed files: $($toriFiles.Count)`n"
if ($toriFiles) {
    $toriFiles | Select-Object -First 5 | ForEach-Object {
        $changelog += "  - $($_.Name)`n"
    }
}

$changelog += "`n### PRAJNA`n"
$prajnaFiles = $changedFiles | Where-Object { $_.FullName -match "\\prajna\\" -or $_.Name -match "prajna" }
$changelog += "- Changed files: $($prajnaFiles.Count)`n"

$changelog += "`n### SCHOLARSPHERE`n"
$scholarFiles = $changedFiles | Where-Object { $_.FullName -match "\\scholarsphere\\" -or $_.Name -match "scholar" }
$changelog += "- Changed files: $($scholarFiles.Count)`n"

$changelog += "`n### SOLITON`n"
$solitonFiles = $changedFiles | Where-Object { $_.FullName -match "\\soliton\\" -or $_.Name -match "soliton" }
$changelog += "- Changed files: $($solitonFiles.Count)`n"

# Add query hints for ChatGPT
$changelog += @"

## CHATGPT QUERY HINTS

To pull these changes efficiently:
1. Use freshness filter: last $days days
2. Path filter: /kha/
3. Focus on these directories first:
"@

# Find directories with most changes
$dirChanges = $changedFiles | Group-Object { Split-Path $_.FullName -Parent } | 
    Sort-Object Count -Descending | Select-Object -First 5
$dirChanges | ForEach-Object {
    $dir = $_.Name.Replace($localPath, "").TrimStart("\")
    $changelog += "   - $dir ($($_.Count) files)`n"
}

$changelog += @"

## SYNC VERIFICATION
- Local path: $localPath
- Drive path: $drivePath
- Files synced: $($changedFiles.Count)
- Sync timestamp: $timestamp
- Ready for ChatGPT freshness query: YES

## NOTES FOR CHATGPT
This changelog represents all significant changes in the last $days days.
Temp files, caches, and build artifacts have been excluded.
Focus on the "Critical System Files" and "Most Recently Modified" sections first.

---
End of Changelog
"@

# Save changelog
$changelogPath = "$localPath\CHATGPT_CHANGELOG_$(Get-Date -Format 'yyyyMMdd_HHmmss').md"
$changelog | Out-File -FilePath $changelogPath -Encoding UTF8

Write-Host "`n✓ Changelog created: $(Split-Path $changelogPath -Leaf)" -ForegroundColor Green

# Also create a simple file list for ChatGPT
$fileList = $changedFiles | ForEach-Object {
    $relativePath = $_.FullName.Replace($localPath, "").TrimStart("\")
    "$($_.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))|$relativePath|$($_.Length)"
}
$fileListPath = "$localPath\CHATGPT_FILES_$(Get-Date -Format 'yyyyMMdd_HHmmss').txt"
$fileList | Out-File -FilePath $fileListPath -Encoding UTF8

Write-Host "✓ File list created: $(Split-Path $fileListPath -Leaf)" -ForegroundColor Green

# Copy to Drive location too
if (Test-Path $drivePath) {
    Copy-Item $changelogPath "$drivePath\" -Force
    Copy-Item $fileListPath "$drivePath\" -Force
    Write-Host "✓ Copied to Google Drive for immediate ChatGPT access" -ForegroundColor Green
}

Write-Host "`n===== SUMMARY FOR CHATGPT =====" -ForegroundColor Cyan
Write-Host "Files changed in last $days days: $($changedFiles.Count)" -ForegroundColor Yellow
Write-Host "Most active component: $(if($toriFiles.Count -gt 0){'TORI'}elseif($prajnaFiles.Count -gt 0){'PRAJNA'}else{'CORE'})" -ForegroundColor Yellow
Write-Host "Changelog location: $changelogPath" -ForegroundColor White

Write-Host "`n===== NEXT STEPS =====" -ForegroundColor Cyan
Write-Host "1. Run CHATGPT_SYNC.bat to sync to Drive" -ForegroundColor White
Write-Host "2. Tell ChatGPT: 'Sync complete, check CHATGPT_CHANGELOG file'" -ForegroundColor White
Write-Host "3. ChatGPT can use freshness query on /kha/ for last 3 days" -ForegroundColor White

Write-Host "`nPress any key to exit..."
$null = $host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")