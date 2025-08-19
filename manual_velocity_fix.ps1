# Manual fix for velocityField.wgsl
Write-Host "`n=== Manual Fix for velocityField.wgsl ===" -ForegroundColor Cyan

$velocityPath = "C:\Users\jason\Desktop\tori\kha\frontend\lib\webgpu\shaders\velocityField.wgsl"

if (Test-Path $velocityPath) {
    Write-Host @"
The velocityField.wgsl shader has storage buffers in function parameters.
This is the problematic pattern around line 284:

fn some_function(@builtin(global_invocation_id) global_id: vec3<u32>,
                @group(0) @binding(5) var<storage, read_write> particles: array<vec4<f32>>) {

It needs to be changed to:

@group(0) @binding(5) var<storage, read_write> particles: array<vec4<f32>>;

fn some_function(@builtin(global_invocation_id) global_id: vec3<u32>) {

The storage buffer declaration must be moved OUTSIDE the function, at module level.

Opening the file in notepad for manual edit...
"@ -ForegroundColor Yellow

    # Open in notepad for manual edit
    notepad $velocityPath
    
    Write-Host "`nAfter editing, save and close Notepad, then re-run validation." -ForegroundColor Green
}
