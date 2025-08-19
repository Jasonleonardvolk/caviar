# Simple manual fix instructions for velocityField.wgsl
Write-Host "`n=== Manual Fix for velocityField.wgsl ===" -ForegroundColor Cyan
Write-Host @"

The issue is on lines 283-284. The function has a storage buffer as a parameter.

CURRENT (BROKEN):
@compute @workgroup_size(128, 1, 1)
fn advect_particles(@builtin(global_invocation_id) global_id: vec3<u32>,
                   @group(0) @binding(5) var<storage, read_write> particles: array<vec4<f32>>) {

NEEDS TO BE:
@group(0) @binding(5) var<storage, read_write> particles: array<vec4<f32>>;

@compute @workgroup_size(128, 1, 1)
fn advect_particles(@builtin(global_invocation_id) global_id: vec3<u32>) {

"@ -ForegroundColor Yellow

$velocityPath = "C:\Users\jason\Desktop\tori\kha\frontend\lib\webgpu\shaders\velocityField.wgsl"

# Show current content around line 284
Write-Host "`nCurrent content:" -ForegroundColor Cyan
$lines = Get-Content $velocityPath
Write-Host "Line 282: $($lines[281])" -ForegroundColor DarkGray
Write-Host "Line 283: $($lines[282])" -ForegroundColor Red
Write-Host "Line 284: $($lines[283])" -ForegroundColor Red
Write-Host "Line 285: $($lines[284])" -ForegroundColor DarkGray

Write-Host "`n📝 TO FIX MANUALLY:" -ForegroundColor Yellow
Write-Host "1. Open in editor: notepad `"$velocityPath`"" -ForegroundColor Cyan
Write-Host "2. Go to line 283-284" -ForegroundColor Cyan
Write-Host "3. Move the @group line ABOVE the @compute line" -ForegroundColor Cyan
Write-Host "4. Remove it from the function parameters" -ForegroundColor Cyan
Write-Host "5. Save and close" -ForegroundColor Cyan

$response = Read-Host "`nPress ENTER to open in Notepad, or 'skip' to skip"
if ($response -ne 'skip') {
    notepad $velocityPath
}
