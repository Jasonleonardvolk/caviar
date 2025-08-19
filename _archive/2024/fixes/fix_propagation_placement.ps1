# Fix propagation.wgsl - move @group to correct position
Write-Host "=== Fixing propagation.wgsl @group placement ===" -ForegroundColor Cyan

$file = "C:\Users\jason\Desktop\tori\kha\frontend\shaders\propagation.wgsl"

# Read the file
$lines = Get-Content $file

# Find the problematic section (around line 474-476)
$newLines = @()
$i = 0

while ($i -lt $lines.Count) {
    # Look for the pattern where @group is between @compute and fn
    if ($lines[$i] -match '@compute.*@workgroup_size' -and 
        $i+1 -lt $lines.Count -and $lines[$i+1] -match '@group.*multiview_buffer') {
        
        Write-Host "Found misplaced @group at line $($i+2)" -ForegroundColor Yellow
        
        # First, add the @group line BEFORE @compute
        $newLines += $lines[$i+1]  # The @group line
        $newLines += ""
        $newLines += $lines[$i]    # The @compute line
        
        # Skip both lines as we've already added them
        $i += 2
    } else {
        $newLines += $lines[$i]
        $i++
    }
}

# Save the fixed file
$newLines | Set-Content $file

Write-Host "`nFixed! Moved @group declaration before @compute directive" -ForegroundColor Green

# Validate
Write-Host "`nValidating..." -ForegroundColor Yellow
$result = & naga $file 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ propagation.wgsl is now valid!" -ForegroundColor Green
} else {
    Write-Host "Still has errors:" -ForegroundColor Red
    $result | Select-Object -First 10
}
