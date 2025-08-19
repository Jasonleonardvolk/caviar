# Manual Fix Guide for 3 Remaining Shaders
Write-Host @"
==================================================
🔧 MANUAL SHADER FIX GUIDE - 3 SHADERS REMAINING
==================================================
"@ -ForegroundColor Cyan

Write-Host "We'll fix each shader one by one. Follow the instructions carefully.`n" -ForegroundColor Yellow

# SHADER 1: lenticularInterlace.wgsl
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray
Write-Host "🔨 SHADER 1: lenticularInterlace.wgsl" -ForegroundColor Green
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray
Write-Host @"

ISSUE: textureLoad needs 3 arguments (missing mip level)

FIND these patterns:
  textureLoad(temp_buffer, coord)
  textureLoad(temp_buffer, sample_coord)

REPLACE with:
  textureLoad(temp_buffer, coord, 0)
  textureLoad(temp_buffer, sample_coord, 0)

Add ', 0' as the third argument to ALL textureLoad calls that only have 2 arguments.
"@ -ForegroundColor Cyan

$file1 = "C:\Users\jason\Desktop\tori\kha\frontend\lib\webgpu\shaders\lenticularInterlace.wgsl"
Read-Host "`nPress ENTER to open lenticularInterlace.wgsl in Notepad"
notepad $file1

Read-Host "`nPress ENTER after you've saved the changes"

# Validate
$result = & naga $file1 2>&1 | Out-String
if (-not ($result -match "error")) {
    Write-Host "✅ lenticularInterlace.wgsl FIXED!" -ForegroundColor Green
} else {
    Write-Host "❌ Still has errors. Check for more textureLoad calls with only 2 args." -ForegroundColor Red
}

# SHADER 2: propagation.wgsl
Write-Host "`n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray
Write-Host "🔨 SHADER 2: propagation.wgsl (frontend/shaders/)" -ForegroundColor Green
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray
Write-Host @"

ISSUE: @group inside function parameters (line 476)

FIND around line 475-476:
  fn prepare_for_multiview(@builtin(global_invocation_id) global_id: vec3<u32>,
                          @group(2) @binding(4) var multiview_buffer: texture_storage_2d<rg32float, write>) {

REPLACE with:
  @group(2) @binding(4) var multiview_buffer: texture_storage_2d<rg32float, write>;

  fn prepare_for_multiview(@builtin(global_invocation_id) global_id: vec3<u32>) {

Move the @group line ABOVE the function!
"@ -ForegroundColor Cyan

$file2 = "C:\Users\jason\Desktop\tori\kha\frontend\shaders\propagation.wgsl"
Read-Host "`nPress ENTER to open propagation.wgsl in Notepad"
notepad $file2

Read-Host "`nPress ENTER after you've saved the changes"

# Validate
$result = & naga $file2 2>&1 | Out-String
if (-not ($result -match "error")) {
    Write-Host "✅ propagation.wgsl FIXED!" -ForegroundColor Green
} else {
    Write-Host "❌ Still has errors at line 476" -ForegroundColor Red
}

# SHADER 3: velocityField.wgsl
Write-Host "`n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray
Write-Host "🔨 SHADER 3: velocityField.wgsl" -ForegroundColor Green
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray
Write-Host @"

ISSUE: @group inside function parameters (line 371)

FIND around line 370-371:
  fn visualize_flow(@builtin(global_invocation_id) global_id: vec3<u32>,
                   @group(0) @binding(6) var flow_vis_out: texture_storage_2d<rgba8unorm, write>) {

REPLACE with:
  @group(0) @binding(6) var flow_vis_out: texture_storage_2d<rgba8unorm, write>;

  fn visualize_flow(@builtin(global_invocation_id) global_id: vec3<u32>) {

Move the @group line ABOVE the function!
"@ -ForegroundColor Cyan

$file3 = "C:\Users\jason\Desktop\tori\kha\frontend\lib\webgpu\shaders\velocityField.wgsl"
Read-Host "`nPress ENTER to open velocityField.wgsl in Notepad"
notepad $file3

Read-Host "`nPress ENTER after you've saved the changes"

# Validate
$result = & naga $file3 2>&1 | Out-String
if (-not ($result -match "error")) {
    Write-Host "✅ velocityField.wgsl FIXED!" -ForegroundColor Green
} else {
    Write-Host "❌ Still has errors at line 371" -ForegroundColor Red
}

# Final validation
Write-Host "`n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray
Write-Host "📊 FINAL VALIDATION" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray

$shaderDir = "C:\Users\jason\Desktop\tori\kha\frontend\lib\webgpu\shaders"
$validCount = 0
$totalCount = 0

Get-ChildItem "$shaderDir\*.wgsl" | ForEach-Object {
    $totalCount++
    $result = & naga $_.FullName 2>&1 | Out-String
    if (-not ($result -match "error")) {
        Write-Host "✅ $($_.Name)" -ForegroundColor Green
        $validCount++
    } else {
        Write-Host "❌ $($_.Name)" -ForegroundColor Red
    }
}

# Also check main propagation.wgsl
$mainProp = "C:\Users\jason\Desktop\tori\kha\frontend\shaders\propagation.wgsl"
if (Test-Path $mainProp) {
    $totalCount++
    $result = & naga $mainProp 2>&1 | Out-String
    if (-not ($result -match "error")) {
        Write-Host "✅ propagation.wgsl (main)" -ForegroundColor Green
        $validCount++
    } else {
        Write-Host "❌ propagation.wgsl (main)" -ForegroundColor Red
    }
}

Write-Host "`n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray
if ($validCount -eq $totalCount) {
    Write-Host "🎉 ALL $totalCount SHADERS ARE NOW VALID! 🎉" -ForegroundColor Green -BackgroundColor DarkGreen
    Write-Host "You can now run TORI without shader compilation errors!" -ForegroundColor Green
} else {
    Write-Host "⚠️  $validCount out of $totalCount shaders are valid" -ForegroundColor Yellow
    Write-Host "Check the errors above and re-run this script." -ForegroundColor Yellow
}
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray
