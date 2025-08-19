# Smart fix based on actual errors

Write-Host "=== SMART FIX BASED ON ACTUAL ERRORS ===" -ForegroundColor Cyan

# First, let's see what the actual errors are
Write-Host "`nChecking current errors..." -ForegroundColor Yellow

$errors = @{}

# Check lenticularInterlace
$shader = "C:\Users\jason\Desktop\tori\kha\frontend\lib\webgpu\shaders\lenticularInterlace.wgsl"
$result = & naga $shader 2>&1 | Out-String
if ($result -match "error: wrong number of arguments: expected (\d+), found (\d+)") {
    $expected = $matches[1]
    $found = $matches[2]
    Write-Host "lenticularInterlace: textureLoad expects $expected args, found $found" -ForegroundColor Red
    
    # Extract line number
    if ($result -match "(\d+):\d+\s*\n.*\n.*textureLoad") {
        $lineNum = [int]$matches[1]
        Write-Host "  Error at line $lineNum" -ForegroundColor DarkRed
        $errors[$shader] = @{Type="textureLoad"; Line=$lineNum; Expected=$expected; Found=$found}
    }
}

# Check velocityField
$shader = "C:\Users\jason\Desktop\tori\kha\frontend\lib\webgpu\shaders\velocityField.wgsl"
$result = & naga $shader 2>&1 | Out-String
if ($result -match "error: unknown attribute: ``group``.*:(\d+):") {
    $lineNum = [int]$matches[1]
    Write-Host "velocityField: @group in function params at line $lineNum" -ForegroundColor Red
    $errors[$shader] = @{Type="group"; Line=$lineNum}
}

# Check propagation
$shader = "C:\Users\jason\Desktop\tori\kha\frontend\lib\webgpu\shaders\propagation.wgsl"
$result = & naga $shader 2>&1 | Out-String
if ($result -match "error: unknown attribute: ``group``.*:(\d+):") {
    $lineNum = [int]$matches[1]
    Write-Host "propagation: @group in function params at line $lineNum" -ForegroundColor Red
    $errors[$shader] = @{Type="group"; Line=$lineNum}
}

# Now fix based on errors
Write-Host "`nApplying fixes..." -ForegroundColor Yellow

foreach ($errorEntry in $errors.GetEnumerator()) {
    $file = $errorEntry.Key
    $error = $errorEntry.Value
    $name = Split-Path $file -Leaf
    
    Write-Host "`nFixing $name..." -ForegroundColor Cyan
    
    if ($error.Type -eq "textureLoad") {
        # Fix textureLoad argument count
        $lines = Get-Content $file
        $lineIndex = $error.Line - 1
        
        Write-Host "  Line $($error.Line): $($lines[$lineIndex])" -ForegroundColor DarkGray
        
        # If it needs 3 args and has 2, add , 0
        if ($error.Expected -eq 3 -and $error.Found -eq 2) {
            $lines[$lineIndex] = $lines[$lineIndex] -replace 'textureLoad\(([^,]+),\s*([^)]+)\)', 'textureLoad($1, $2, 0)'
            Write-Host "  Fixed: Added mip level 0" -ForegroundColor Green
        }
        
        $lines | Set-Content $file
    }
    elseif ($error.Type -eq "group") {
        # Fix @group in function parameters
        $lines = Get-Content $file
        $lineIndex = $error.Line - 1
        
        # Find the function and extract storage declaration
        $funcStart = -1
        for ($i = $lineIndex; $i -ge 0 -and $i -ge $lineIndex - 10; $i--) {
            if ($lines[$i] -match 'fn\s+\w+') {
                $funcStart = $i
                break
            }
        }
        
        if ($funcStart -ge 0) {
            # Extract the @group line
            $groupLine = $lines[$lineIndex]
            if ($groupLine -match '@group\((\d+)\)\s*@binding\((\d+)\)\s*var(?:<[^>]+>)?\s+(\w+):\s*(.+)') {
                $decl = "@group($($matches[1])) @binding($($matches[2])) var $($matches[3]): $($matches[4]);"
                
                # Insert declaration before function
                $newLines = @()
                for ($i = 0; $i -lt $lines.Count; $i++) {
                    if ($i -eq $funcStart) {
                        $newLines += ""
                        $newLines += "// Storage moved from function parameters"
                        $newLines += $decl
                        $newLines += ""
                    }
                    
                    if ($i -eq $lineIndex) {
                        # Skip the @group line
                        continue
                    }
                    
                    $newLines += $lines[$i]
                }
                
                $newLines | Set-Content $file
                Write-Host "  Fixed: Moved storage declaration to module level" -ForegroundColor Green
            }
        }
    }
}

# Final validation
Write-Host "`n=== VALIDATION ===" -ForegroundColor Cyan
$allValid = $true
foreach ($file in $errors.Keys) {
    $name = Split-Path $file -Leaf
    $result = & naga $file 2>&1 | Out-String
    if (-not ($result -match "error")) {
        Write-Host "✅ $name" -ForegroundColor Green
    } else {
        Write-Host "❌ $name - Still has errors" -ForegroundColor Red
        $allValid = $false
    }
}

if (-not $allValid) {
    Write-Host "`nSome fixes didn't work. Running comprehensive fix..." -ForegroundColor Yellow
    & ".\comprehensive_fix_3_shaders.ps1"
}
