# scrub_bom_and_recopy.ps1
# Removes BOM from WGSL files and recopies to public directory

$sourceDir = Join-Path $PSScriptRoot ".." ".." "frontend" "lib" "webgpu" "shaders"
$targetDir = Join-Path $PSScriptRoot ".." ".." "frontend" "public" "hybrid" "wgsl"

Write-Host "🧹 Scrubbing BOM from WGSL files..." -ForegroundColor Cyan

# Function to remove BOM
function Remove-BOM {
    param($FilePath)
    
    $content = Get-Content -Path $FilePath -Raw -Encoding UTF8
    
    # Check if file has BOM
    $bytes = [System.IO.File]::ReadAllBytes($FilePath)
    if ($bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF) {
        Write-Host "  Removing BOM from: $(Split-Path $FilePath -Leaf)" -ForegroundColor Yellow
        
        # Write without BOM
        $utf8NoBOM = New-Object System.Text.UTF8Encoding $false
        [System.IO.File]::WriteAllText($FilePath, $content, $utf8NoBOM)
        return $true
    }
    return $false
}

# Process all WGSL files
$wgslFiles = Get-ChildItem -Path $sourceDir -Filter "*.wgsl"
$bomCount = 0

foreach ($file in $wgslFiles) {
    if (Remove-BOM -FilePath $file.FullName) {
        $bomCount++
    }
}

if ($bomCount -gt 0) {
    Write-Host "✅ Removed BOM from $bomCount files" -ForegroundColor Green
} else {
    Write-Host "✅ No BOM found in any files" -ForegroundColor Green
}

# Copy to public directory
Write-Host "`n📋 Copying shaders to public directory..." -ForegroundColor Cyan

foreach ($file in $wgslFiles) {
    $targetPath = Join-Path $targetDir $file.Name
    Copy-Item -Path $file.FullName -Destination $targetPath -Force
    Write-Host "  Copied: $($file.Name)" -ForegroundColor Gray
}

Write-Host "✅ Successfully copied $($wgslFiles.Count) shader files!" -ForegroundColor Green
Write-Host "All shaders are now synchronized and BOM-free." -ForegroundColor Green
