# Intelligent Auto-Fix for WGSL Shaders
# Applies fixes identified by intelligent_diagnostic.mjs
# 
# Unlike the fired gentleman's work, this ACTUALLY fixes things

param(
    [string]$ReportPath = ".\build\intelligent_diagnostic.json",
    [switch]$DryRun,
    [switch]$Backup
)

$ErrorActionPreference = "Stop"

Write-Host "🔧 Intelligent WGSL Auto-Fixer" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor DarkGray

# Load the diagnostic report
if (-not (Test-Path $ReportPath)) {
    Write-Host "❌ No diagnostic report found. Run intelligent_diagnostic.mjs first." -ForegroundColor Red
    exit 1
}

$report = Get-Content $ReportPath -Raw | ConvertFrom-Json

Write-Host "📊 Found $($report.summary.realIssues) real issues to fix" -ForegroundColor Yellow
Write-Host "   - Errors: $($report.summary.errors)" -ForegroundColor Red
Write-Host "   - Warnings: $($report.summary.warnings)" -ForegroundColor Yellow
Write-Host ""

# Backup function
function Backup-File($filePath) {
    if ($Backup) {
        $backupPath = "$filePath.backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
        Copy-Item $filePath $backupPath
        Write-Host "   📁 Backed up to: $(Split-Path -Leaf $backupPath)" -ForegroundColor Gray
    }
}

# Fix application functions
function Fix-SwizzleAssignment($file, $line, $fix) {
    $content = Get-Content $file
    $originalLine = $content[$line - 1]
    
    # Extract the variable and operator
    if ($originalLine -match '(\w+)\.([rgba]{2,4}|[xyzw]{2,4})\s*([+\-*/%])?=\s*(.+);') {
        $varName = $matches[1]
        $swizzle = $matches[2]
        $operator = $matches[3]
        $value = $matches[4]
        
        # Generate individual assignments
        $newLines = @()
        foreach ($component in $swizzle.ToCharArray()) {
            $newLines += "    $varName.$component $operator= $value;"
        }
        
        # Replace the line
        $content[$line - 1] = $newLines -join "`n"
        
        if (-not $DryRun) {
            Set-Content $file $content -NoNewline
            Write-Host "   ✅ Fixed swizzle assignment at line $line" -ForegroundColor Green
        } else {
            Write-Host "   [DRY RUN] Would fix swizzle at line $line" -ForegroundColor Cyan
            Write-Host "      FROM: $originalLine" -ForegroundColor DarkGray
            Write-Host "      TO:   $($newLines -join ' | ')" -ForegroundColor Green
        }
    }
}

function Fix-TypeCast($file, $line, $fix) {
    $content = Get-Content $file
    $originalLine = $content[$line - 1]
    
    # Fix vec2<i32>(u32, u32) patterns
    if ($originalLine -match 'vec2<i32>\(([^,]+),\s*([^)]+)\)') {
        $arg1 = $matches[1].Trim()
        $arg2 = $matches[2].Trim()
        
        # Check if arguments need casting
        $needsCast1 = $arg1 -match '^\w+$' -and $arg1 -notmatch '^i32\('
        $needsCast2 = $arg2 -match '^\w+$' -and $arg2 -notmatch '^i32\('
        
        if ($needsCast1 -or $needsCast2) {
            $newArg1 = if ($needsCast1) { "i32($arg1)" } else { $arg1 }
            $newArg2 = if ($needsCast2) { "i32($arg2)" } else { $arg2 }
            
            $newLine = $originalLine -replace 'vec2<i32>\([^)]+\)', "vec2<i32>($newArg1, $newArg2)"
            $content[$line - 1] = $newLine
            
            if (-not $DryRun) {
                Set-Content $file $content -NoNewline
                Write-Host "   ✅ Fixed type cast at line $line" -ForegroundColor Green
            } else {
                Write-Host "   [DRY RUN] Would fix type cast at line $line" -ForegroundColor Cyan
            }
        }
    }
}

function Fix-TextureLoadArgs($file, $line, $fix) {
    $content = Get-Content $file
    $originalLine = $content[$line - 1]
    
    # Count commas to determine argument count
    if ($originalLine -match 'textureLoad\s*\(([^)]+)\)') {
        $args = $matches[1]
        $argCount = ($args.ToCharArray() | Where-Object {$_ -eq ','}).Count + 1
        
        # Add missing mip level argument
        if ($argCount -eq 3) {
            # texture_2d_array needs 4 args
            $newLine = $originalLine -replace '\)', ', 0)'
            $content[$line - 1] = $newLine
            
            if (-not $DryRun) {
                Set-Content $file $content -NoNewline
                Write-Host "   ✅ Added mip level argument at line $line" -ForegroundColor Green
            } else {
                Write-Host "   [DRY RUN] Would add mip level at line $line" -ForegroundColor Cyan
            }
        }
    }
}

function Fix-StoragePadding($file, $structName, $fixes) {
    $content = Get-Content $file -Raw
    
    # Find the struct
    if ($content -match "(struct\s+$structName\s*\{[^}]+\})") {
        $structBlock = $matches[1]
        $modifiedStruct = $structBlock
        
        foreach ($fix in $fixes) {
            # Add padding after vec3 fields
            $pattern = "($($fix.after)\s*:\s*vec3<f32>)"
            $replacement = "`$1,`n    _pad_$($fix.after): f32"
            $modifiedStruct = $modifiedStruct -replace $pattern, $replacement
        }
        
        $newContent = $content.Replace($structBlock, $modifiedStruct)
        
        if (-not $DryRun) {
            Set-Content $file $newContent -NoNewline
            Write-Host "   ✅ Added padding to struct $structName" -ForegroundColor Green
        } else {
            Write-Host "   [DRY RUN] Would add padding to $structName" -ForegroundColor Cyan
        }
    }
}

# Process each issue
$fixedCount = 0
$groupedByFile = $report.realIssues | Group-Object -Property file

foreach ($fileGroup in $groupedByFile) {
    $fileName = $fileGroup.Name
    $filePath = Join-Path ".\frontend\hybrid\wgsl" $fileName
    
    if (-not (Test-Path $filePath)) {
        # Try other locations
        $filePath = Join-Path ".\frontend\lib\webgpu\shaders" $fileName
    }
    
    if (-not (Test-Path $filePath)) {
        Write-Host "⚠️  Cannot find file: $fileName" -ForegroundColor Yellow
        continue
    }
    
    Write-Host "`n📁 Processing $fileName..." -ForegroundColor White
    
    if ($Backup -and -not $DryRun) {
        Backup-File $filePath
    }
    
    foreach ($issue in $fileGroup.Group) {
        Write-Host "   Line $($issue.line): $($issue.problem)" -ForegroundColor DarkYellow
        
        if ($issue.fix) {
            switch ($issue.rule) {
                'SWIZZLE_ASSIGNMENT' {
                    Fix-SwizzleAssignment $filePath $issue.line $issue.fix
                    $fixedCount++
                }
                'TYPE_MISMATCH_VEC_CONSTRUCTOR' {
                    Fix-TypeCast $filePath $issue.line $issue.fix
                    $fixedCount++
                }
                'TEXTURE_LOAD_ARGS' {
                    Fix-TextureLoadArgs $filePath $issue.line $issue.fix
                    $fixedCount++
                }
                'STORAGE_VEC3_PADDING' {
                    # Handle struct padding specially
                    if ($issue.problem -match "struct '(\w+)'") {
                        Fix-StoragePadding $filePath $matches[1] $issue.fixes
                        $fixedCount++
                    }
                }
                default {
                    Write-Host "   ⚠️  No auto-fix available for $($issue.rule)" -ForegroundColor Yellow
                }
            }
        } else {
            Write-Host "   ℹ️  Manual fix required" -ForegroundColor Cyan
        }
    }
}

Write-Host "`n================================" -ForegroundColor DarkGray
if ($DryRun) {
    Write-Host "🔍 DRY RUN COMPLETE" -ForegroundColor Cyan
    Write-Host "   Would fix $fixedCount issues" -ForegroundColor White
    Write-Host "   Run without -DryRun to apply fixes" -ForegroundColor Gray
} else {
    Write-Host "✅ AUTO-FIX COMPLETE" -ForegroundColor Green
    Write-Host "   Fixed $fixedCount issues" -ForegroundColor White
    Write-Host "   Run validator to confirm all errors resolved" -ForegroundColor Gray
}

# Generate validation command
Write-Host "`n🎯 Next step:" -ForegroundColor Magenta
Write-Host "   node tools/shaders/shader_quality_gate_v2.mjs --dir=frontend/ --strict" -ForegroundColor White
