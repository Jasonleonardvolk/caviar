# Direct fix for propagation.wgsl line 476
Write-Host "=== Direct Fix for propagation.wgsl Line 476 ===" -ForegroundColor Cyan

$file = "C:\Users\jason\Desktop\tori\kha\frontend\shaders\propagation.wgsl"

if (Test-Path $file) {
    $content = Get-Content $file -Raw
    
    # The problematic pattern around line 476
    # Function with @group in parameters
    $pattern = 'fn prepare_for_multiview\(@builtin\(global_invocation_id\) global_id: vec3<u32>,\s*\n\s*@group\(2\) @binding\(4\) var multiview_buffer: texture_storage_2d<rg32float, write>\)\s*{'
    
    # The fix - move @group above function
    $replacement = @'
@group(2) @binding(4) var multiview_buffer: texture_storage_2d<rg32float, write>;

fn prepare_for_multiview(@builtin(global_invocation_id) global_id: vec3<u32>) {
'@
    
    if ($content -match $pattern) {
        Write-Host "Found the pattern to fix!" -ForegroundColor Green
        $content = $content -replace $pattern, $replacement
        $content | Set-Content $file -NoNewline
        Write-Host "Fixed successfully!" -ForegroundColor Green
    } else {
        Write-Host "Pattern not found. Trying alternative fix..." -ForegroundColor Yellow
        
        # Alternative: Find and fix line by line
        $lines = Get-Content $file
        $newLines = @()
        
        for ($i = 0; $i -lt $lines.Count; $i++) {
            if ($i -eq 475 -and $lines[$i] -match '@group.*multiview_buffer') {
                # Insert declaration before the function (which should be a few lines up)
                # Find where to insert by looking backwards for the function declaration
                $insertIndex = $newLines.Count - 1
                while ($insertIndex -gt 0 -and -not ($newLines[$insertIndex] -match 'fn prepare_for_multiview')) {
                    $insertIndex--
                }
                
                if ($insertIndex -gt 0) {
                    # Insert before the function
                    $before = $newLines[0..($insertIndex-1)]
                    $after = $newLines[$insertIndex..($newLines.Count-1)]
                    
                    $newLines = $before
                    $newLines += ""
                    $newLines += "@group(2) @binding(4) var multiview_buffer: texture_storage_2d<rg32float, write>;"
                    $newLines += ""
                    $newLines += $after
                    
                    # Change the current line to close the function signature properly
                    continue
                }
            }
            
            $newLines += $lines[$i]
        }
        
        $newLines | Set-Content $file
        Write-Host "Applied alternative fix" -ForegroundColor Green
    }
    
    # Validate
    Write-Host "`nValidating..." -ForegroundColor Yellow
    $result = & naga $file 2>&1 | Out-String
    if (-not ($result -match "error")) {
        Write-Host "✅ propagation.wgsl is now valid!" -ForegroundColor Green
    } else {
        Write-Host "❌ Still has errors. Opening file for manual edit..." -ForegroundColor Red
        Write-Host "`nThe issue is at line 476. Change:" -ForegroundColor Yellow
        Write-Host @"
FROM:
fn prepare_for_multiview(@builtin(global_invocation_id) global_id: vec3<u32>,
                        @group(2) @binding(4) var multiview_buffer: texture_storage_2d<rg32float, write>) {

TO:
@group(2) @binding(4) var multiview_buffer: texture_storage_2d<rg32float, write>;

fn prepare_for_multiview(@builtin(global_invocation_id) global_id: vec3<u32>) {
"@ -ForegroundColor Cyan
        
        $response = Read-Host "`nPress ENTER to open in notepad"
        if ($response -eq '') {
            notepad $file
        }
    }
}
