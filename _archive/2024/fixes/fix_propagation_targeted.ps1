# Show exact content and fix propagation.wgsl
Write-Host "=== Examining propagation.wgsl Line 476 ===" -ForegroundColor Cyan

$file = "C:\Users\jason\Desktop\tori\kha\frontend\shaders\propagation.wgsl"

# First, let's see what's actually there
$lines = Get-Content $file
Write-Host "`nContent around line 476:" -ForegroundColor Yellow
for ($i = 470; $i -lt 480 -and $i -lt $lines.Count; $i++) {
    $lineNum = $i + 1
    if ($lineNum -eq 476) {
        Write-Host "${lineNum}: $($lines[$i])" -ForegroundColor Red -BackgroundColor DarkGray
    } else {
        Write-Host "${lineNum}: $($lines[$i])" -ForegroundColor DarkGray
    }
}

# Now let's fix it with a targeted approach
Write-Host "`nApplying targeted fix..." -ForegroundColor Yellow

$newLines = @()
$foundPrepareFunc = $false
$addedDeclaration = $false

for ($i = 0; $i -lt $lines.Count; $i++) {
    $line = $lines[$i]
    $lineNum = $i + 1
    
    # When we find prepare_for_multiview function
    if ($line -match 'fn prepare_for_multiview' -and -not $addedDeclaration) {
        # Add the storage declaration before the function
        $newLines += "@group(2) @binding(4) var multiview_buffer: texture_storage_2d<rg32float, write>;"
        $newLines += ""
        $addedDeclaration = $true
        $foundPrepareFunc = $true
    }
    
    # Skip line 476 if it contains @group
    if ($lineNum -eq 476 -and $line -match '@group.*multiview_buffer') {
        # Replace with proper function parameter closing
        $newLines += "                        @builtin(global_invocation_id) global_id: vec3<u32>) {"
        continue
    }
    
    # For all other lines, just add them
    $newLines += $line
}

if ($foundPrepareFunc) {
    # Save the fixed content
    $newLines | Set-Content $file
    Write-Host "`nFixed! Changes made:" -ForegroundColor Green
    Write-Host "- Added multiview_buffer declaration before function" -ForegroundColor Green
    Write-Host "- Removed @group from function parameters" -ForegroundColor Green
    
    # Validate
    $result = & naga $file 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n✅ propagation.wgsl is now valid!" -ForegroundColor Green
    } else {
        Write-Host "`n❌ Validation failed:" -ForegroundColor Red
        $result | Select-Object -First 5
    }
} else {
    Write-Host "`nCould not find prepare_for_multiview function!" -ForegroundColor Red
}
