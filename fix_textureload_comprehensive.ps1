# Comprehensive textureLoad fix
Write-Host "`n=== Comprehensive textureLoad Fix ===" -ForegroundColor Cyan

$lenticularPath = "C:\Users\jason\Desktop\tori\kha\frontend\lib\webgpu\shaders\lenticularInterlace.wgsl"

if (Test-Path $lenticularPath) {
    Write-Host "Searching for all textureLoad calls with 3 arguments..." -ForegroundColor Yellow
    
    $content = Get-Content $lenticularPath -Raw
    $originalContent = $content
    
    # Count occurrences
    $count = ([regex]::Matches($content, 'textureLoad\([^)]+,\s*[^)]+,\s*\d+\)')).Count
    Write-Host "Found $count textureLoad calls with 3 arguments" -ForegroundColor Yellow
    
    # Fix all patterns of textureLoad with 3 arguments
    # Pattern 1: textureLoad(texture, coord, 0)
    $content = $content -replace 'textureLoad\(([^,]+),\s*([^,]+),\s*0\)', 'textureLoad($1, $2)'
    
    # Pattern 2: textureLoad(texture, coord, mipLevel) where mipLevel is any number
    $content = $content -replace 'textureLoad\(([^,]+),\s*([^,]+),\s*\d+\)', 'textureLoad($1, $2)'
    
    # Pattern 3: textureLoad with .rgb or similar at the end
    $content = $content -replace 'textureLoad\(([^,]+),\s*([^,)]+),\s*[^)]+\)\.(rgb|rgba|xyz|xy)', 'textureLoad($1, $2).$3'
    
    if ($content -ne $originalContent) {
        # Create backup
        Copy-Item $lenticularPath "$lenticularPath.backup_textureLoad" -Force
        
        # Save fixed content
        $content | Set-Content $lenticularPath -NoNewline
        Write-Host "Fixed all textureLoad calls" -ForegroundColor Green
        
        # Show what was changed
        $newCount = ([regex]::Matches($content, 'textureLoad\([^)]+,\s*[^)]+,\s*\d+\)')).Count
        Write-Host "Remaining 3-arg textureLoad calls: $newCount" -ForegroundColor Cyan
    } else {
        Write-Host "No changes needed" -ForegroundColor Yellow
    }
    
    # Validate
    Write-Host "`nValidating lenticularInterlace.wgsl..." -ForegroundColor Cyan
    $result = & naga $lenticularPath 2>&1
    if ($result -match "error") {
        Write-Host "Still has errors:" -ForegroundColor Red
        $result | Select-Object -First 5
    } else {
        Write-Host "✅ lenticularInterlace.wgsl is now valid!" -ForegroundColor Green
    }
}
