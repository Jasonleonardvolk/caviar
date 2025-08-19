# Direct line replacement fixes

Write-Host "=== DIRECT LINE FIXES ===" -ForegroundColor Cyan

# Fix 1: lenticularInterlace.wgsl line 367
$file1 = "C:\Users\jason\Desktop\tori\kha\frontend\lib\webgpu\shaders\lenticularInterlace.wgsl"
Write-Host "`n1. Fixing $file1 line 367..." -ForegroundColor Yellow

$lines = Get-Content $file1
if ($lines[366] -like "*return textureLoad(quilt_texture, quilt_coord);*") {
    $lines[366] = $lines[366] -replace 'textureLoad\(quilt_texture, quilt_coord\)', 'textureLoad(quilt_texture, quilt_coord, 0)'
    $lines | Set-Content $file1
    Write-Host "Fixed line 367" -ForegroundColor Green
} else {
    Write-Host "Line 367 doesn't match expected pattern" -ForegroundColor Red
    Write-Host "Line 367: $($lines[366])" -ForegroundColor DarkGray
}

# Fix 2: propagation.wgsl line 476  
$file2 = "C:\Users\jason\Desktop\tori\kha\frontend\lib\webgpu\shaders\propagation.wgsl"
Write-Host "`n2. Fixing $file2..." -ForegroundColor Yellow

# First ensure it's not JSON
$content = Get-Content $file2 -Raw
if ($content -match '^\s*{') {
    Write-Host "Copying WGSL from frontend/shaders..." -ForegroundColor Yellow
    Copy-Item "C:\Users\jason\Desktop\tori\kha\frontend\shaders\propagation.wgsl" $file2 -Force
}

# Re-read and fix
$lines = Get-Content $file2
$newLines = @()
$i = 0

while ($i -lt $lines.Count) {
    # Look for prepare_for_multiview with @group parameter
    if ($lines[$i] -match 'fn prepare_for_multiview' -and $i+1 -lt $lines.Count -and $lines[$i+1] -match '@group.*multiview_buffer') {
        # Insert storage declaration before function
        $newLines += ""
        $newLines += "@group(2) @binding(4) var multiview_buffer: texture_storage_2d<rg32float, write>;"
        $newLines += ""
        $newLines += $lines[$i]
        $newLines += "                        @builtin(global_invocation_id) global_id: vec3<u32>) {"
        $i += 2  # Skip the @group line
    } else {
        $newLines += $lines[$i]
        $i++
    }
}

$newLines | Set-Content $file2
Write-Host "Fixed prepare_for_multiview function" -ForegroundColor Green

# Fix 3: velocityField.wgsl
$file3 = "C:\Users\jason\Desktop\tori\kha\frontend\lib\webgpu\shaders\velocityField.wgsl" 
Write-Host "`n3. Fixing $file3..." -ForegroundColor Yellow

# Read and check around line 371
$lines = Get-Content $file3
Write-Host "Lines around 371:" -ForegroundColor DarkGray
for ($i = 368; $i -lt 374 -and $i -lt $lines.Count; $i++) {
    Write-Host "$($i+1): $($lines[$i])" -ForegroundColor DarkGray
}

# Look for any function with @group in parameters
$content = Get-Content $file3 -Raw
$fixed = $false

# Pattern for any function with @group parameter
if ($content -match '(fn\s+\w+\s*\([^)]*?)(@group\(\d+\)\s*@binding\(\d+\)\s*var<[^>]+>\s*\w+:\s*[^)]+)(\)\s*{)') {
    $before = $matches[1]
    $storage = $matches[2]
    $after = $matches[3]
    
    # Extract storage details and create declaration
    if ($storage -match '@group\((\d+)\)\s*@binding\((\d+)\)\s*var<([^>]+)>\s*(\w+):\s*(.+)') {
        $decl = "@group($($matches[1])) @binding($($matches[2])) var<$($matches[3])> $($matches[4]): $($matches[5]);"
        
        # Find where to insert (before the function)
        $insertPoint = $content.IndexOf($matches[0])
        
        # Insert declaration before function
        $newContent = $content.Substring(0, $insertPoint) + "`n$decl`n`n" + $before + $after + $content.Substring($insertPoint + $matches[0].Length)
        
        $newContent | Set-Content $file3 -NoNewline
        Write-Host "Fixed storage buffer declaration" -ForegroundColor Green
        $fixed = $true
    }
}

if (-not $fixed) {
    Write-Host "Could not find pattern to fix" -ForegroundColor Red
}

# Final validation
Write-Host "`n=== VALIDATION ===" -ForegroundColor Cyan
@($file1, $file2, $file3) | ForEach-Object {
    $name = Split-Path $_ -Leaf
    $result = & naga $_ 2>&1 | Out-String
    if (-not ($result -match "error")) {
        Write-Host "✅ $name" -ForegroundColor Green
    } else {
        Write-Host "❌ $name" -ForegroundColor Red
        $firstError = ($result -split "`n" | Where-Object { $_ -match "error:" }) | Select-Object -First 1
        Write-Host "   $firstError" -ForegroundColor DarkRed
    }
}
