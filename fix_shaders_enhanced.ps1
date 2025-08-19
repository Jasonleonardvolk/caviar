# Enhanced WGSL Shader Auto-Fixer
# Fixes common WGSL syntax errors automatically

$shaderDir = "C:\Users\jason\Desktop\tori\kha\frontend\lib\webgpu\shaders"
$naga = "naga"

Write-Host "`n=== WGSL Shader Auto-Fixer ===" -ForegroundColor Cyan
Write-Host "Fixing common shader compilation errors automatically`n" -ForegroundColor Yellow

# Function to fix swizzle assignments
function Fix-SwizzleAssignment {
    param($content)
    
    # Replace color.rgb = expr with individual assignments
    $content = $content -replace '(\s*)(\w+)\.rgb\s*=\s*([^;]+);', @'
$1var temp_rgb = $3;
$1$2.r = temp_rgb.r;
$1$2.g = temp_rgb.g;
$1$2.b = temp_rgb.b;
'@
    
    # Replace other common swizzle patterns
    $content = $content -replace '(\s*)(\w+)\.xy\s*=\s*([^;]+);', @'
$1var temp_xy = $3;
$1$2.x = temp_xy.x;
$1$2.y = temp_xy.y;
'@
    
    return $content
}

# Function to fix storage buffers in function parameters
function Fix-StorageBufferParams {
    param($content)
    
    # Find function declarations with @group parameters
    if ($content -match '(@compute[^\n]+\n)fn\s+(\w+)\s*\([^)]*@group[^)]+\)\s*{') {
        Write-Host "  - Moving storage buffer declarations to module level" -ForegroundColor Green
        
        # Extract storage buffer declarations from function parameters
        $pattern = 'fn\s+(\w+)\s*\(([^)]*)\)\s*{'
        $content = $content -replace $pattern, {
            $funcName = $matches[1]
            $params = $matches[2]
            
            # Extract and move storage declarations
            $storageDecls = @()
            $cleanParams = @()
            
            $params -split ',' | ForEach-Object {
                $param = $_.Trim()
                if ($param -match '@group') {
                    # Extract the storage declaration
                    if ($param -match '@group\((\d+)\)\s*@binding\((\d+)\)\s*var<([^>]+)>\s*(\w+):\s*(.+)') {
                        $storageDecls += "@group($($matches[1])) @binding($($matches[2])) var<$($matches[3])> $($matches[4]): $($matches[5]);"
                    }
                } elseif ($param -match '@builtin') {
                    $cleanParams += $param
                }
            }
            
            # Place storage declarations before the function
            $result = ""
            if ($storageDecls.Count -gt 0) {
                $result = "`n// Storage buffers moved from function parameters`n"
                $result += ($storageDecls -join "`n") + "`n"
            }
            $result += "fn $funcName(" + ($cleanParams -join ', ') + ") {"
            
            return $result
        }
    }
    
    return $content
}

# Function to check if content starts with invalid character
function Fix-InvalidStart {
    param($content)
    
    # Remove leading { or other invalid characters
    if ($content -match '^\s*{') {
        Write-Host "  - Removing invalid leading '{' character" -ForegroundColor Green
        $content = $content -replace '^\s*{\s*', ''
    }
    
    return $content
}

# Function to fix missing struct fields
function Fix-MissingStructField {
    param($content, $structName, $fieldName)
    
    # Find the struct and add the missing field
    if ($content -match "(struct\s+$structName\s*{[^}]+)(})") {
        $structContent = $matches[1]
        
        # Add the field before the closing brace
        if ($fieldName -eq "aberration_strength") {
            Write-Host "  - Adding missing field 'aberration_strength' to struct '$structName'" -ForegroundColor Green
            $newField = ",`n    aberration_strength: f32"
            $content = $content -replace "(struct\s+$structName\s*{[^}]+)(})", "`$1$newField`n`$2"
        }
    }
    
    return $content
}

# Process each shader file
Get-ChildItem "$shaderDir\*.wgsl" | ForEach-Object {
    $file = $_.FullName
    $name = $_.Name
    Write-Host "`nProcessing: $name" -ForegroundColor Cyan
    
    # Read the file
    $content = Get-Content $file -Raw
    $originalContent = $content
    $modified = $false
    
    # Run initial validation
    $result = & $naga $file 2>&1 | Out-String
    
    # Apply fixes based on errors
    if ($result -match 'expected global item.*found') {
        $content = Fix-InvalidStart -content $content
        $modified = $true
    }
    
    if ($result -match 'cannot assign to this expression') {
        Write-Host "  - Fixing swizzle assignment" -ForegroundColor Green
        $content = Fix-SwizzleAssignment -content $content
        $modified = $true
    }
    
    if ($result -match 'invalid field accessor') {
        Write-Host "  - Adding missing aberration_strength field" -ForegroundColor Green
        # Find which struct needs the field
        if ($result -match 'render_params\.aberration_strength') {
            # Assuming RenderParams struct needs the field
            $content = Fix-MissingStructField -content $content -structName "RenderParams" -fieldName "aberration_strength"
            $modified = $true
        }
    }
    
    if ($result -match 'unknown attribute') {
        $content = Fix-StorageBufferParams -content $content
        $modified = $true
    }
    
    if ($result -match 'reserved keyword') {
        Write-Host "  - Replacing reserved keyword 'filter'" -ForegroundColor Green
        $content = $content -replace '\bfilter\b', 'filter_val'
        $modified = $true
    }
    
    # Save if modified
    if ($modified) {
        # Create backup
        $backupFile = "$file.backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
        Copy-Item $file $backupFile
        Write-Host "  - Created backup: $(Split-Path $backupFile -Leaf)" -ForegroundColor DarkGray
        
        # Save fixed content
        $content | Set-Content $file -NoNewline
        Write-Host "  - Saved fixes to $name" -ForegroundColor Green
        
        # Re-validate
        Write-Host "  - Re-validating..." -ForegroundColor Yellow
        $result2 = & $naga $file 2>&1 | Out-String
        
        if ($result2 -match "error") {
            Write-Host "  ! Still has errors (may need manual fix):" -ForegroundColor Red
            Write-Host ($result2 -split "`n" | Select-Object -First 5 | Out-String)
        } else {
            Write-Host "  ✓ $name is now valid!" -ForegroundColor Green
        }
    } else {
        if ($result -match "error") {
            Write-Host "  ! Has errors that need manual fixing" -ForegroundColor Red
        } else {
            Write-Host "  ✓ Already valid" -ForegroundColor Green
        }
    }
}

Write-Host "`n=== Auto-Fix Complete ===" -ForegroundColor Cyan
Write-Host "Check the results above. Some shaders may still need manual fixes." -ForegroundColor Yellow
Write-Host "Backups were created for all modified files." -ForegroundColor DarkGray
