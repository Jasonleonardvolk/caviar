param(
  [string]$RepoRoot = "D:\Dev\kha",
  [switch]$SkipClean,
  [switch]$SkipBuild,
  [switch]$Verbose,
  [switch]$StopOnFirstError
)

$ErrorActionPreference = "Stop"
Set-Location $RepoRoot

# Color configuration
$colors = @{
  Header = "Cyan"
  Pass = "Green"
  Fail = "Red"
  Warn = "Yellow"
  Info = "White"
  Debug = "Gray"
}

# Timestamp for reports
$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$reportDir = Join-Path $RepoRoot "tools\release\ship-ready-reports"
New-Item -ItemType Directory -Force -Path $reportDir | Out-Null

# Results tracking
$global:testResults = @()
$global:criticalErrors = @()
$global:warnings = @()
$global:passCount = 0
$global:failCount = 0

function Write-Header {
  param([string]$Text)
  Write-Host ""
  Write-Host ("=" * 80) -ForegroundColor $colors.Header
  Write-Host $Text -ForegroundColor $colors.Header
  Write-Host ("=" * 80) -ForegroundColor $colors.Header
}

function Write-Step {
  param([string]$Step, [string]$Description)
  Write-Host ""
  Write-Host "[$Step]" -ForegroundColor $colors.Header -NoNewline
  Write-Host " $Description" -ForegroundColor $colors.Info
}

function Test-Result {
  param(
    [string]$Test,
    [bool]$Pass,
    [string]$Message = "",
    [bool]$Critical = $true
  )
  
  $result = @{
    Test = $Test
    Pass = $Pass
    Message = $Message
    Critical = $Critical
    Timestamp = Get-Date -Format "HH:mm:ss"
  }
  
  $global:testResults += $result
  
  if ($Pass) {
    Write-Host "  [PASS] " -ForegroundColor $colors.Pass -NoNewline
    Write-Host "$Test" -ForegroundColor $colors.Info
    $global:passCount++
  } else {
    Write-Host "  [FAIL] " -ForegroundColor $colors.Fail -NoNewline
    Write-Host "$Test" -ForegroundColor $colors.Info
    if ($Message) {
      Write-Host "    --> $Message" -ForegroundColor $colors.Warn
    }
    $global:failCount++
    
    if ($Critical) {
      $global:criticalErrors += "$Test: $Message"
      if ($StopOnFirstError) {
        Write-Host ""
        Write-Host "CRITICAL ERROR - Stopping validation" -ForegroundColor $colors.Fail
        Show-Summary
        exit 1
      }
    } else {
      $global:warnings += "$Test: $Message"
    }
  }
}

function Run-Command {
  param(
    [string]$Name,
    [scriptblock]$Command,
    [string]$WorkingDir = $null,
    [bool]$Critical = $true
  )
  
  $logFile = Join-Path $reportDir "$Name-$timestamp.log"
  $errorFile = Join-Path $reportDir "$Name-$timestamp.error"
  
  try {
    if ($WorkingDir) {
      Push-Location $WorkingDir
    }
    
    if ($Verbose) {
      Write-Host "    Running: $Command" -ForegroundColor $colors.Debug
    }
    
    # Execute and capture output
    $output = & $Command 2>&1
    $exitCode = $LASTEXITCODE
    
    # Save output
    $output | Out-File -FilePath $logFile -Encoding UTF8
    
    if ($exitCode -eq 0) {
      Test-Result -Test $Name -Pass $true -Critical $Critical
      return $true
    } else {
      # Extract error info
      $errors = $output | Where-Object { $_ -match "error" -or $_ -match "fail" }
      $errors | Out-File -FilePath $errorFile -Encoding UTF8
      
      $errorSummary = if ($errors.Count -gt 0) { 
        "$($errors.Count) errors (see $errorFile)" 
      } else { 
        "Exit code $exitCode" 
      }
      
      Test-Result -Test $Name -Pass $false -Message $errorSummary -Critical $Critical
      return $false
    }
  }
  catch {
    $_.Exception.Message | Out-File -FilePath $errorFile -Encoding UTF8
    Test-Result -Test $Name -Pass $false -Message $_.Exception.Message -Critical $Critical
    return $false
  }
  finally {
    if ($WorkingDir) {
      Pop-Location
    }
  }
}

function Test-FileExists {
  param(
    [string]$Path,
    [string]$Description,
    [bool]$Critical = $true
  )
  
  $exists = Test-Path $Path
  $message = if (-not $exists) { "File not found: $Path" } else { "" }
  Test-Result -Test $Description -Pass $exists -Message $message -Critical $Critical
  return $exists
}

function Test-DirectoryExists {
  param(
    [string]$Path,
    [string]$Description,
    [int]$MinFiles = 0,
    [bool]$Critical = $true
  )
  
  if (Test-Path $Path) {
    if ($MinFiles -gt 0) {
      $fileCount = (Get-ChildItem -Path $Path -Recurse -File | Measure-Object).Count
      $pass = $fileCount -ge $MinFiles
      $message = if (-not $pass) { "Only $fileCount files (expected >= $MinFiles)" } else { "$fileCount files found" }
      Test-Result -Test $Description -Pass $pass -Message $message -Critical $Critical
      return $pass
    } else {
      Test-Result -Test $Description -Pass $true -Critical $Critical
      return $true
    }
  } else {
    Test-Result -Test $Description -Pass $false -Message "Directory not found: $Path" -Critical $Critical
    return $false
  }
}

function Show-Summary {
  Write-Header "VALIDATION SUMMARY"
  
  Write-Host ""
  Write-Host "Results: " -NoNewline
  Write-Host "$global:passCount PASSED" -ForegroundColor $colors.Pass -NoNewline
  Write-Host " / " -NoNewline
  Write-Host "$global:failCount FAILED" -ForegroundColor $colors.Fail
  
  if ($global:criticalErrors.Count -gt 0) {
    Write-Host ""
    Write-Host "CRITICAL ERRORS:" -ForegroundColor $colors.Fail
    foreach ($error in $global:criticalErrors) {
      Write-Host "  - $error" -ForegroundColor $colors.Fail
    }
  }
  
  if ($global:warnings.Count -gt 0) {
    Write-Host ""
    Write-Host "WARNINGS:" -ForegroundColor $colors.Warn
    foreach ($warning in $global:warnings) {
      Write-Host "  - $warning" -ForegroundColor $colors.Warn
    }
  }
  
  # Save report
  $reportFile = Join-Path $reportDir "ship-ready-report-$timestamp.json"
  $report = @{
    Timestamp = $timestamp
    RepoRoot = $RepoRoot
    PassCount = $global:passCount
    FailCount = $global:failCount
    CriticalErrors = $global:criticalErrors
    Warnings = $global:warnings
    Results = $global:testResults
    ShipReady = ($global:criticalErrors.Count -eq 0)
  }
  $report | ConvertTo-Json -Depth 5 | Out-File -FilePath $reportFile -Encoding UTF8
  
  Write-Host ""
  Write-Host "Full report: $reportFile" -ForegroundColor $colors.Info
  
  Write-Host ""
  if ($global:criticalErrors.Count -eq 0) {
    Write-Host "+=========================================+" -ForegroundColor $colors.Pass
    Write-Host "|         SHIP READY: YES [OK]           |" -ForegroundColor $colors.Pass
    Write-Host "+=========================================+" -ForegroundColor $colors.Pass
    return $true
  } else {
    Write-Host "+=========================================+" -ForegroundColor $colors.Fail
    Write-Host "|         SHIP READY: NO [FAIL]          |" -ForegroundColor $colors.Fail
    Write-Host "+=========================================+" -ForegroundColor $colors.Fail
    return $false
  }
}

# ============================================================================
# MAIN VALIDATION SEQUENCE
# ============================================================================

Write-Header "IRIS SHIP-READY VALIDATION"
Write-Host "Repository: $RepoRoot"
Write-Host "Timestamp: $timestamp"

# ----------------------------------------------------------------------------
# STEP 1: ENVIRONMENT VALIDATION
# ----------------------------------------------------------------------------
Write-Step "STEP 1" "Environment Validation"

# Check Node.js
$nodeVersion = node -v 2>$null
Test-Result -Test "Node.js installed" -Pass ($null -ne $nodeVersion) -Message "Node.js is required" -Critical $true

# Check npm
$npmVersion = npm -v 2>$null
Test-Result -Test "npm installed" -Pass ($null -ne $npmVersion) -Message "npm is required" -Critical $true

# Check Git
$gitVersion = git --version 2>$null
Test-Result -Test "Git installed" -Pass ($null -ne $gitVersion) -Critical $false

# Check for clean working tree
$gitStatus = git status --porcelain 2>$null
$isClean = [string]::IsNullOrWhiteSpace($gitStatus)
Test-Result -Test "Git working tree clean" -Pass $isClean -Message "Uncommitted changes present" -Critical $false

# ----------------------------------------------------------------------------
# STEP 2: DEPENDENCIES CHECK
# ----------------------------------------------------------------------------
Write-Step "STEP 2" "Dependencies Validation"

# Check package.json exists
Test-FileExists -Path (Join-Path $RepoRoot "package.json") -Description "Root package.json exists"

# Check for node_modules
Test-DirectoryExists -Path (Join-Path $RepoRoot "node_modules") -Description "Root node_modules exists" -MinFiles 100

# Check tori_ui_svelte dependencies
$toriPath = Join-Path $RepoRoot "tori_ui_svelte"
if (Test-Path $toriPath) {
  Test-FileExists -Path (Join-Path $toriPath "package.json") -Description "tori_ui_svelte package.json"
  Test-DirectoryExists -Path (Join-Path $toriPath "node_modules") -Description "tori_ui_svelte node_modules" -MinFiles 100
}

# ----------------------------------------------------------------------------
# STEP 3: TYPESCRIPT COMPILATION
# ----------------------------------------------------------------------------
Write-Step "STEP 3" "TypeScript Compilation"

# Check for TypeScript config
$tsconfigPath = Join-Path $RepoRoot "tsconfig.json"
$tsconfigFrontend = Join-Path $RepoRoot "frontend\tsconfig.json"
$tsconfigTori = Join-Path $toriPath "tsconfig.json"

$hasTsConfig = (Test-Path $tsconfigPath) -or (Test-Path $tsconfigFrontend) -or (Test-Path $tsconfigTori)
Test-Result -Test "TypeScript config exists" -Pass $hasTsConfig -Critical $true

# Run TypeScript check
if (Test-Path $tsconfigTori) {
  Run-Command -Name "TypeScript tori_ui_svelte" -Command { npx tsc --noEmit } -WorkingDir $toriPath
} elseif (Test-Path $tsconfigFrontend) {
  Run-Command -Name "TypeScript frontend" -Command { npx tsc -p frontend\tsconfig.json --noEmit }
} else {
  Run-Command -Name "TypeScript root" -Command { npx tsc --noEmit }
}

# ----------------------------------------------------------------------------
# STEP 4: CRITICAL FILE VALIDATION
# ----------------------------------------------------------------------------
Write-Step "STEP 4" "Critical File Validation"

# Check QuiltGenerator path fix
$realGhostPath = Join-Path $toriPath "src\lib\realGhostEngine.js"
if (Test-Path $realGhostPath) {
  $content = Get-Content $realGhostPath -Raw
  $hasOldImport = $content -match "frontend/lib/webgpu/quiltGenerator"
  $hasNewImport = $content -match "tools/quilt/WebGPU/QuiltGenerator"
  
  Test-Result -Test "QuiltGenerator import fixed" -Pass (-not $hasOldImport -and $hasNewImport) `
    -Message "Import path needs correction" -Critical $true
}

# Check QuiltGenerator file exists
$quiltPath = Join-Path $RepoRoot "tools\quilt\WebGPU\QuiltGenerator.ts"
Test-FileExists -Path $quiltPath -Description "QuiltGenerator.ts exists" -Critical $false

# Check for Svelte files
$svelteFiles = Get-ChildItem -Path $toriPath -Filter "*.svelte" -Recurse -ErrorAction SilentlyContinue
Test-Result -Test "Svelte components exist" -Pass ($svelteFiles.Count -gt 0) `
  -Message "No .svelte files found" -Critical $true

# ----------------------------------------------------------------------------
# STEP 5: SHADER VALIDATION
# ----------------------------------------------------------------------------
Write-Step "STEP 5" "Shader Validation"

$shaderPath = Join-Path $RepoRoot "frontend\shaders"
$wgslFiles = @()

if (Test-Path $shaderPath) {
  $wgslFiles = Get-ChildItem -Path $shaderPath -Filter "*.wgsl" -Recurse -ErrorAction SilentlyContinue
}

# Also check tools/quilt for shaders
$quiltShaderPath = Join-Path $RepoRoot "tools\quilt\WebGPU\shaders"
if (Test-Path $quiltShaderPath) {
  $wgslFiles += Get-ChildItem -Path $quiltShaderPath -Filter "*.wgsl" -Recurse -ErrorAction SilentlyContinue
}

Test-Result -Test "WGSL shader files found" -Pass ($wgslFiles.Count -gt 0) `
  -Message "$($wgslFiles.Count) shader files" -Critical $false

# Run shader validation if validator exists
$shaderValidator = Join-Path $RepoRoot "tools\shaders\validate-wgsl.js"
if (Test-Path $shaderValidator) {
  Run-Command -Name "Shader validation" -Command { node tools\shaders\validate-wgsl.js --dir=frontend --strict } -Critical $false
}

# ----------------------------------------------------------------------------
# STEP 6: API SMOKE TEST
# ----------------------------------------------------------------------------
Write-Step "STEP 6" "API Smoke Test"

$apiSmokeScript = Join-Path $RepoRoot "tools\release\api-smoke.js"
if (Test-Path $apiSmokeScript) {
  $envFile = Join-Path $RepoRoot ".env.production"
  if (-not (Test-Path $envFile)) {
    $envFile = Join-Path $RepoRoot ".env"
  }
  
  if (Test-Path $envFile) {
    Run-Command -Name "API smoke test" -Command { node tools\release\api-smoke.js --env $envFile } -Critical $false
  } else {
    Test-Result -Test "API smoke test" -Pass $false -Message "No .env file found" -Critical $false
  }
} else {
  Test-Result -Test "API smoke test" -Pass $false -Message "api-smoke.js not found" -Critical $false
}

# ----------------------------------------------------------------------------
# STEP 7: BUILD PROCESS
# ----------------------------------------------------------------------------
if (-not $SkipBuild) {
  Write-Step "STEP 7" "Build Process"
  
  # Clean previous builds if requested
  if (-not $SkipClean) {
    Write-Host "  Cleaning previous build artifacts..." -ForegroundColor $colors.Info
    $buildDirs = @(
      "tori_ui_svelte\build",
      "tori_ui_svelte\dist",
      "tori_ui_svelte\.svelte-kit",
      "dist",
      "build"
    )
    
    foreach ($dir in $buildDirs) {
      $fullPath = Join-Path $RepoRoot $dir
      if (Test-Path $fullPath) {
        Remove-Item -Path $fullPath -Recurse -Force -ErrorAction SilentlyContinue
      }
    }
  }
  
  # Attempt build
  $buildSuccess = Run-Command -Name "npm run build" -Command { npm run build } -WorkingDir $toriPath
  
  # Check build outputs
  $buildOutput = Join-Path $toriPath "build"
  $distOutput = Join-Path $toriPath "dist"
  $svelteKitOutput = Join-Path $toriPath ".svelte-kit\output"
  
  $hasOutput = (Test-Path $buildOutput) -or (Test-Path $distOutput) -or (Test-Path $svelteKitOutput)
  Test-Result -Test "Build output generated" -Pass $hasOutput -Message "No build output found" -Critical $true
  
  if ($hasOutput) {
    # Count files in output
    $outputPath = if (Test-Path $buildOutput) { $buildOutput } `
      elseif (Test-Path $distOutput) { $distOutput } `
      else { $svelteKitOutput }
    
    $fileCount = (Get-ChildItem -Path $outputPath -Recurse -File | Measure-Object).Count
    Test-Result -Test "Build contains files" -Pass ($fileCount -gt 10) `
      -Message "$fileCount files generated" -Critical $true
      
    # Check for index.html
    $indexFile = Get-ChildItem -Path $outputPath -Filter "index.html" -Recurse | Select-Object -First 1
    Test-Result -Test "index.html exists" -Pass ($null -ne $indexFile) -Critical $true
    
    # Check for JavaScript bundles
    $jsFiles = Get-ChildItem -Path $outputPath -Filter "*.js" -Recurse
    Test-Result -Test "JavaScript bundles exist" -Pass ($jsFiles.Count -gt 0) `
      -Message "$($jsFiles.Count) JS files" -Critical $true
  }
} else {
  Write-Host "  Skipping build step (--SkipBuild specified)" -ForegroundColor $colors.Warn
}

# ----------------------------------------------------------------------------
# STEP 8: RELEASE STRUCTURE
# ----------------------------------------------------------------------------
Write-Step "STEP 8" "Release Structure Validation"

$releasesDir = Join-Path $RepoRoot "releases"
Test-DirectoryExists -Path $releasesDir -Description "Releases directory exists" -Critical $false

# Check for latest release
if (Test-Path $releasesDir) {
  $latestRelease = Get-ChildItem -Path $releasesDir -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  
  if ($latestRelease) {
    $releasePath = $latestRelease.FullName
    $manifestPath = Join-Path $releasePath "manifest.json"
    $distPath = Join-Path $releasePath "dist"
    
    Test-FileExists -Path $manifestPath -Description "Release manifest.json" -Critical $false
    Test-DirectoryExists -Path $distPath -Description "Release dist folder" -MinFiles 1 -Critical $false
    
    # Verify manifest content
    if (Test-Path $manifestPath) {
      try {
        $manifest = Get-Content $manifestPath | ConvertFrom-Json
        Test-Result -Test "Manifest version field" -Pass ($null -ne $manifest.version) -Critical $false
      } catch {
        Test-Result -Test "Manifest JSON valid" -Pass $false -Message $_.Exception.Message -Critical $false
      }
    }
  } else {
    Test-Result -Test "Release folder exists" -Pass $false -Message "No releases found" -Critical $false
  }
}

# ----------------------------------------------------------------------------
# STEP 9: FINAL CHECKS
# ----------------------------------------------------------------------------
Write-Step "STEP 9" "Final Ship-Ready Checks"

# Check for README
Test-FileExists -Path (Join-Path $RepoRoot "README.md") -Description "README.md exists" -Critical $false

# Check for LICENSE
$hasLicense = (Test-Path (Join-Path $RepoRoot "LICENSE")) -or (Test-Path (Join-Path $RepoRoot "LICENSE.md"))
Test-Result -Test "LICENSE file exists" -Pass $hasLicense -Critical $false

# Check for .gitignore
Test-FileExists -Path (Join-Path $RepoRoot ".gitignore") -Description ".gitignore exists" -Critical $false

# Check package.json has required fields
$packageJsonPath = Join-Path $RepoRoot "package.json"
if (Test-Path $packageJsonPath) {
  $packageJson = Get-Content $packageJsonPath | ConvertFrom-Json
  Test-Result -Test "package.json has name" -Pass ($null -ne $packageJson.name) -Critical $false
  Test-Result -Test "package.json has version" -Pass ($null -ne $packageJson.version) -Critical $false
  Test-Result -Test "package.json has scripts.build" -Pass ($null -ne $packageJson.scripts.build) -Critical $true
}

# Check for test files
$testFiles = Get-ChildItem -Path $RepoRoot -Filter "*.test.*" -Recurse -ErrorAction SilentlyContinue
$specFiles = Get-ChildItem -Path $RepoRoot -Filter "*.spec.*" -Recurse -ErrorAction SilentlyContinue
$hasTests = ($testFiles.Count -gt 0) -or ($specFiles.Count -gt 0)
Test-Result -Test "Test files exist" -Pass $hasTests `
  -Message "No test files found" -Critical $false

# ----------------------------------------------------------------------------
# SUMMARY
# ----------------------------------------------------------------------------
$shipReady = Show-Summary

# Exit with appropriate code
if ($shipReady) {
  exit 0
} else {
  exit 1
}
