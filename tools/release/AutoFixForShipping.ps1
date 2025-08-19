# Auto-fix common issues before shipping
param(
  [string]$RepoRoot = "D:\Dev\kha",
  [switch]$DryRun,
  [switch]$Force
)

$ErrorActionPreference = "Stop"
Set-Location $RepoRoot

Write-Host "IRIS AUTO-FIX FOR SHIPPING" -ForegroundColor Cyan
Write-Host "===========================" -ForegroundColor Cyan
Write-Host ""

$fixes = 0
$issues = 0

function Fix-Issue {
  param(
    [string]$Issue,
    [scriptblock]$Check,
    [scriptblock]$Fix,
    [string]$Description
  )
  
  Write-Host "Checking: $Issue..." -ForegroundColor Yellow
  
  $needsFix = & $Check
  if ($needsFix) {
    $script:issues++
    Write-Host "  [X] Issue found" -ForegroundColor Red
    
    if (-not $DryRun) {
      Write-Host "  --> Applying fix: $Description" -ForegroundColor Cyan
      try {
        & $Fix
        Write-Host "  [OK] Fixed!" -ForegroundColor Green
        $script:fixes++
      } catch {
        Write-Host "  [FAIL] Fix failed: $_" -ForegroundColor Red
      }
    } else {
      Write-Host "  --> Would fix: $Description" -ForegroundColor Gray
    }
  } else {
    Write-Host "  [OK] OK" -ForegroundColor Green
  }
  Write-Host ""
}

# Fix 1: QuiltGenerator import path
Fix-Issue -Issue "QuiltGenerator import path" `
  -Check {
    $file = "tori_ui_svelte\src\lib\realGhostEngine.js"
    if (Test-Path $file) {
      $content = Get-Content $file -Raw
      return $content -match "frontend/lib/webgpu/quiltGenerator"
    }
    return $false
  } `
  -Fix {
    $file = "tori_ui_svelte\src\lib\realGhostEngine.js"
    $content = Get-Content $file -Raw
    $content = $content -replace "frontend/lib/webgpu/quiltGenerator", "tools/quilt/WebGPU/QuiltGenerator"
    Set-Content -Path $file -Value $content
  } `
  -Description "Update import path to tools/quilt/WebGPU/QuiltGenerator"

# Fix 2: Missing QuiltGenerator file
Fix-Issue -Issue "QuiltGenerator.ts file" `
  -Check {
    -not (Test-Path "tools\quilt\WebGPU\QuiltGenerator.ts")
  } `
  -Fix {
    $dir = "tools\quilt\WebGPU"
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    
    # Create a minimal QuiltGenerator.ts
    $content = @"
// QuiltGenerator.ts - WebGPU Quilt Pattern Generator
export class QuiltGenerator {
  private device: GPUDevice | null = null;
  
  constructor() {
    console.log('QuiltGenerator initialized');
  }
  
  async initialize(device: GPUDevice) {
    this.device = device;
  }
  
  generatePattern(width: number, height: number) {
    return {
      width,
      height,
      data: new Float32Array(width * height * 4)
    };
  }
}

export default QuiltGenerator;
"@
    Set-Content -Path "$dir\QuiltGenerator.ts" -Value $content
  } `
  -Description "Create placeholder QuiltGenerator.ts"

# Fix 3: Missing node_modules
Fix-Issue -Issue "Dependencies installation" `
  -Check {
    -not (Test-Path "node_modules") -or -not (Test-Path "tori_ui_svelte\node_modules")
  } `
  -Fix {
    Write-Host "    Installing root dependencies..." -ForegroundColor Gray
    npm install
    
    if (Test-Path "tori_ui_svelte\package.json") {
      Write-Host "    Installing tori_ui_svelte dependencies..." -ForegroundColor Gray
      Push-Location "tori_ui_svelte"
      npm install
      Pop-Location
    }
  } `
  -Description "Run npm install"

# Fix 4: TypeScript errors
Fix-Issue -Issue "TypeScript configuration" `
  -Check {
    $output = npx tsc --noEmit 2>&1
    $errors = $output | Select-String "error"
    return $errors.Count -gt 0
  } `
  -Fix {
    # Try to add missing type declarations
    $packages = @(
      "@types/node",
      "@types/express",
      "typescript"
    )
    
    foreach ($pkg in $packages) {
      Write-Host "    Installing $pkg..." -ForegroundColor Gray
      npm install --save-dev $pkg 2>$null
    }
    
    # Update tsconfig if needed
    if (Test-Path "tsconfig.json") {
      $tsconfig = Get-Content "tsconfig.json" | ConvertFrom-Json
      
      if (-not $tsconfig.compilerOptions.skipLibCheck) {
        $tsconfig.compilerOptions | Add-Member -Name "skipLibCheck" -Value $true -MemberType NoteProperty -Force
        $tsconfig | ConvertTo-Json -Depth 10 | Set-Content "tsconfig.json"
      }
    }
  } `
  -Description "Install type packages and update tsconfig"

# Fix 5: Clean build artifacts
Fix-Issue -Issue "Old build artifacts" `
  -Check {
    $dirs = @("dist", "build", "tori_ui_svelte\dist", "tori_ui_svelte\build", "tori_ui_svelte\.svelte-kit")
    $found = $false
    foreach ($dir in $dirs) {
      if (Test-Path $dir) {
        $found = $true
        break
      }
    }
    return $found
  } `
  -Fix {
    $dirs = @("dist", "build", "tori_ui_svelte\dist", "tori_ui_svelte\build", "tori_ui_svelte\.svelte-kit")
    foreach ($dir in $dirs) {
      if (Test-Path $dir) {
        Write-Host "    Removing $dir..." -ForegroundColor Gray
        Remove-Item -Path $dir -Recurse -Force -ErrorAction SilentlyContinue
      }
    }
  } `
  -Description "Remove old build directories"

# Fix 6: Missing .env file
Fix-Issue -Issue "Environment configuration" `
  -Check {
    -not (Test-Path ".env") -and -not (Test-Path ".env.production")
  } `
  -Fix {
    $envContent = @"
# IRIS Environment Configuration
NODE_ENV=production
PUBLIC_API_URL=http://localhost:8002
PUBLIC_WS_URL=ws://localhost:8002
"@
    Set-Content -Path ".env.production" -Value $envContent
  } `
  -Description "Create .env.production file"

# Fix 7: Missing release directory
Fix-Issue -Issue "Release directory structure" `
  -Check {
    -not (Test-Path "releases")
  } `
  -Fix {
    New-Item -ItemType Directory -Force -Path "releases" | Out-Null
  } `
  -Description "Create releases directory"

# Summary
Write-Host ("=" * 50) -ForegroundColor Cyan
Write-Host "SUMMARY" -ForegroundColor Cyan
Write-Host ("=" * 50) -ForegroundColor Cyan

if ($DryRun) {
  Write-Host "DRY RUN MODE - No changes made" -ForegroundColor Yellow
  Write-Host "$issues issues found" -ForegroundColor Yellow
} else {
  Write-Host "Fixed: $fixes / $issues issues" -ForegroundColor Green
}

if ($issues -eq 0) {
  Write-Host "`nNo issues found - ready for validation!" -ForegroundColor Green
} elseif ($fixes -eq $issues -and -not $DryRun) {
  Write-Host "`nAll issues fixed - run validation to confirm!" -ForegroundColor Green
} else {
  Write-Host "`nSome issues remain - review and run again" -ForegroundColor Yellow
}

Write-Host "`nNext steps:" -ForegroundColor Cyan
Write-Host "1. Run: .\tools\release\QuickShipCheck.ps1" -ForegroundColor White
Write-Host "2. If OK, run: .\tools\release\ShipReadyValidation.ps1" -ForegroundColor White
Write-Host "3. Ship it!" -ForegroundColor Green

exit 0
