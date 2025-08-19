# Simple manual fix for propagation.wgsl
Write-Host @"
=== MANUAL FIX FOR propagation.wgsl ===

The error is on line 476. Here's what you need to do:

1. Look for this code around line 475-476:

fn prepare_for_multiview(@builtin(global_invocation_id) global_id: vec3<u32>,
                        @group(2) @binding(4) var multiview_buffer: texture_storage_2d<rg32float, write>) {

2. Change it to:

@group(2) @binding(4) var multiview_buffer: texture_storage_2d<rg32float, write>;

fn prepare_for_multiview(@builtin(global_invocation_id) global_id: vec3<u32>) {

That's it! The @group declaration must be OUTSIDE the function, not inside the parameters.
"@ -ForegroundColor Cyan

$file = "C:\Users\jason\Desktop\tori\kha\frontend\shaders\propagation.wgsl"

Write-Host "`nOpening file in Notepad..." -ForegroundColor Yellow
Write-Host "1. Press Ctrl+G to go to line" -ForegroundColor Yellow
Write-Host "2. Type 476 and press Enter" -ForegroundColor Yellow
Write-Host "3. Make the changes shown above" -ForegroundColor Yellow
Write-Host "4. Save (Ctrl+S) and close" -ForegroundColor Yellow

notepad $file

Read-Host "`nPress ENTER after you've made the changes"

# Validate
Write-Host "`nValidating..." -ForegroundColor Yellow
$result = & naga $file 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ propagation.wgsl is now valid!" -ForegroundColor Green
} else {
    Write-Host "❌ Still has errors:" -ForegroundColor Red
    $result | Select-Object -First 5
}
