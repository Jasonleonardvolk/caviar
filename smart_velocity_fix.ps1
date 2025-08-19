# Smart fix for velocityField.wgsl
Write-Host "`n=== Smart Fix for velocityField.wgsl ===" -ForegroundColor Cyan

$velocityPath = "C:\Users\jason\Desktop\tori\kha\frontend\lib\webgpu\shaders\velocityField.wgsl"

if (Test-Path $velocityPath) {
    $content = Get-Content $velocityPath -Raw
    
    # Pattern to find compute functions with storage buffers in parameters
    # Looking specifically around line 284
    $pattern = @'
(?ms)(.*?)(@compute\s+@workgroup_size\([^)]+\)\s*\n\s*fn\s+\w+\s*\()([^)]*?)(@group\(\d+\)\s*@binding\(\d+\)\s*var<[^>]+>\s*(\w+):\s*[^),]+)(,?[^)]*?\))(.*)
'@

    if ($content -match $pattern) {
        $before = $matches[1]
        $funcStart = $matches[2]
        $paramsBeforeStorage = $matches[3]
        $storageParam = $matches[4]
        $varName = $matches[5]
        $paramsAfter = $matches[6]
        $after = $matches[7]
        
        # Clean up the storage parameter to make a proper module-level declaration
        $storageDecl = $storageParam -replace '^\s*,\s*', '' -replace ',\s*$', ''
        $storageDecl = "`n// Storage buffer moved from function parameters`n" + $storageDecl + ";`n"
        
        # Remove the storage parameter from function params
        $cleanParams = $paramsBeforeStorage + $paramsAfter
        $cleanParams = $cleanParams -replace '^\s*,\s*', '' -replace ',\s*,', ',' -replace ',\s*$', ''
        
        # Reconstruct the file
        $fixedContent = $before + $storageDecl + "`n" + $funcStart + $cleanParams + $after
        
        # Save
        Copy-Item $velocityPath "$velocityPath.backup_smart" -Force
        $fixedContent | Set-Content $velocityPath -NoNewline
        
        Write-Host "Successfully moved storage buffer '$varName' to module level" -ForegroundColor Green
        Write-Host "Backup saved as velocityField.wgsl.backup_smart" -ForegroundColor DarkGray
    } else {
        Write-Host "Could not find the pattern. Showing context around line 284:" -ForegroundColor Yellow
        $lines = $content -split "`n"
        for ($i = 280; $i -lt 290 -and $i -lt $lines.Count; $i++) {
            Write-Host "$($i+1): $($lines[$i])"
        }
    }
}

# Validate the fix
Write-Host "`nValidating velocityField.wgsl..." -ForegroundColor Cyan
$result = & naga $velocityPath 2>&1
if ($result -match "error") {
    Write-Host "Still has errors:" -ForegroundColor Red
    $result | Select-Object -First 10
} else {
    Write-Host "✅ velocityField.wgsl is now valid!" -ForegroundColor Green
}
