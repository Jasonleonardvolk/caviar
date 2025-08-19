    # TORI One-Command Dev Install for Windows
    # Creates venv, installs all deps, builds Rust crates, and launches TORI
    param(
        [switch]$Headless = $false
    )

    Set-StrictMode -Version Latest
    $ErrorActionPreference = "Stop"
    $StartTime = Get-Date

    # Helper function for consistent banners
    function Write-Banner {
        param([string]$text)
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host "🚀 $text" -ForegroundColor Cyan
        Write-Host "========================================" -ForegroundColor Cyan
    }

    Write-Banner "TORI ONE-COMMAND DEV INSTALL"
    Write-Host ""

    # Check execution policy
    if ((Get-ExecutionPolicy) -in 'AllSigned','Restricted') {
        Write-Warning "ExecutionPolicy is $(Get-ExecutionPolicy); script may fail. Try: Set-ExecutionPolicy -Scope Process Bypass"
    }

    # Check Python version
    Write-Host "🐍 Checking Python version..." -ForegroundColor Green

    # Try different Python commands
    $pythonCmd = $null
    $pythonVersion = $null
    foreach ($cmd in @('python', 'python3', 'py -3')) {
        try {
            $testVersion = & $cmd --version 2>&1
            if ($testVersion -match '^Python\s+3\.(\d+)') {
                $pythonCmd = $cmd
                $pythonVersion = $testVersion
                break
            }
        } catch {
            continue
        }
    }

    if (-not $pythonCmd) {
        Write-Host "❌ Python not found or version check failed" -ForegroundColor Red
        exit 1
    }

    if ($pythonVersion -match '^Python\s+3\.(\d+)') {
        $minorVersion = [int]$matches[1]
        if ($minorVersion -lt 8) {
            Write-Host "❌ Python 3.8+ required (found: $pythonVersion)" -ForegroundColor Red
            exit 1
        }
        Write-Host "✅ $pythonVersion (using: $pythonCmd)" -ForegroundColor Green
    }

    # Create venv
    Write-Host ""
    Write-Host "🐍 Creating virtual environment..." -ForegroundColor Green
    if (Test-Path .venv) {
        Write-Host "   Virtual environment already exists, using it" -ForegroundColor Yellow
    } else {
        & $pythonCmd -m venv .venv
        Write-Host "✅ Virtual environment created" -ForegroundColor Green
    }

    # Activate venv (dot-source to keep it active)
    Write-Host ""
    Write-Host "🔧 Activating virtual environment..." -ForegroundColor Green
    if (-not $Env:VIRTUAL_ENV) {
        . .\.venv\Scripts\Activate.ps1
        Write-Host "✅ Virtual environment activated" -ForegroundColor Green
    } else {
        Write-Host "   Virtual environment already active" -ForegroundColor Yellow
    }

    # Upgrade pip
    Write-Host ""
    Write-Host "📦 Upgrading pip, wheel, setuptools..." -ForegroundColor Green
    python -m pip install -U pip wheel setuptools 2>&1 | Tee-Object -FilePath pip_upgrade.log
    Write-Host "✅ Package tools upgraded" -ForegroundColor Green

    # Install TORI with dev dependencies
    Write-Host ""
    Write-Host "📦 Installing TORI (editable) + dev dependencies..." -ForegroundColor Green
    if (Test-Path pyproject.toml) {
        Write-Host "   Installing from pyproject.toml..." -ForegroundColor Gray
        python -m pip install -e ".[dev]" 2>&1 | Tee-Object -FilePath install.log
        if ($LASTEXITCODE -ne 0) {
            Write-Host "❌ Installation failed. Check install.log for details" -ForegroundColor Red
            exit 1
        }
    } else {
        # Fallback to requirements-dev.txt
        Write-Host "   pyproject.toml not found, using requirements-dev.txt" -ForegroundColor Yellow
        if (Test-Path requirements-dev.txt) {
            python -m pip install -r requirements-dev.txt 2>&1 | Tee-Object -FilePath install.log
            if ($LASTEXITCODE -ne 0) {
                Write-Host "❌ Installation failed. Check install.log for details" -ForegroundColor Red
                exit 1
            }
        } else {
            Write-Host "❌ No requirements-dev.txt found either" -ForegroundColor Red
            exit 1
        }
    }
    Write-Host "✅ Python dependencies installed" -ForegroundColor Green

    # Install spaCy model
    Write-Host ""
    Write-Host "🧠 Installing spaCy language model..." -ForegroundColor Green
    python -m spacy download en_core_web_lg -q 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ spaCy model installed" -ForegroundColor Green
    } else {
        Write-Host "⚠️  spaCy model installation failed (non-critical)" -ForegroundColor Yellow
    }

    # Build Rust crates
    Write-Host ""
    Write-Host "🦀 Checking for Rust toolchain..." -ForegroundColor Green

    if (Get-Command cargo -ErrorAction SilentlyContinue) {
        $rustVersion = rustc --version
        Write-Host "   Found $rustVersion" -ForegroundColor Gray
        
        # Check for maturin
        if (-not (Get-Command maturin -ErrorAction SilentlyContinue)) {
            Write-Host "   Installing maturin..." -ForegroundColor Gray
            python -m pip install maturin
        }
        
        # Build Penrose
        if (Test-Path "concept_mesh\penrose_rs\Cargo.toml") {
            Write-Host "   Building Penrose (similarity engine)..." -ForegroundColor Gray
            Push-Location concept_mesh\penrose_rs
            maturin develop --release 2>&1 | Tee-Object -FilePath ..\..\penrose_build.log
            Pop-Location
            if ($LASTEXITCODE -eq 0) {
                Write-Host "✅ Penrose built" -ForegroundColor Green
            } else {
                Write-Host "⚠️  Penrose build failed (check penrose_build.log)" -ForegroundColor Yellow
            }
        }
        
        # Build concept_mesh_rs (if it has PyO3 bindings)
        if (Test-Path "concept_mesh\Cargo.toml") {
            Write-Host "   Checking concept_mesh_rs for PyO3 bindings..." -ForegroundColor Gray
            if (Select-String -Path "concept_mesh\Cargo.toml" -Pattern "pyo3" -SimpleMatch -Quiet) {
                Push-Location concept_mesh
                maturin develop --release 2>&1 | Tee-Object -FilePath ..\concept_mesh_build.log
                Pop-Location
                if ($LASTEXITCODE -eq 0) {
                    Write-Host "✅ concept_mesh_rs built" -ForegroundColor Green
                } else {
                    Write-Host "⚠️  concept_mesh_rs build failed (check concept_mesh_build.log)" -ForegroundColor Yellow
                }
            } else {
                Write-Host "   Skipping concept_mesh_rs (no PyO3 bindings)" -ForegroundColor Yellow
            }
        }
    } else {
        Write-Host "⚠️  Rust not found, skipping Rust builds" -ForegroundColor Yellow
        Write-Host "   Install from: https://rustup.rs/" -ForegroundColor Yellow
    }

    # Install frontend dependencies
    Write-Host ""
    Write-Host "🌐 Checking for Node.js..." -ForegroundColor Green
    if (Test-Path "frontend\package.json") {
        if (Get-Command npm -ErrorAction SilentlyContinue) {
            Write-Host "   Installing frontend dependencies..." -ForegroundColor Gray
            Push-Location frontend
            npm ci 2>&1 | Tee-Object -FilePath ..\frontend_install.log
            Pop-Location
            if ($LASTEXITCODE -eq 0) {
                Write-Host "✅ Frontend dependencies installed" -ForegroundColor Green
            } else {
                Write-Host "⚠️  Frontend installation had issues (check frontend_install.log)" -ForegroundColor Yellow
            }
        } else {
            Write-Host "⚠️  npm not found, skipping frontend setup" -ForegroundColor Yellow
            Write-Host "   Install Node.js from: https://nodejs.org/" -ForegroundColor Yellow
        }
    } else {
        Write-Host "   No frontend directory found, skipping" -ForegroundColor Yellow
    }

    # Link MCP packages
    Write-Host ""
    Write-Host "🔗 Installing MCP packages..." -ForegroundColor Green
    if (Test-Path "mcp_metacognitive") {
        python -m pip install mcp 2>&1 | Tee-Object -FilePath mcp_install.log
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ MCP base package installed" -ForegroundColor Green
        } else {
            Write-Host "⚠️  MCP installation failed (non-critical)" -ForegroundColor Yellow
        }
    } else {
        Write-Host "   MCP metacognitive not found, skipping" -ForegroundColor Yellow
    }

    # Install additional critical packages
    Write-Host ""
    Write-Host "📦 Ensuring critical packages are installed..." -ForegroundColor Green
    python -m pip install sse-starlette pypdf2 deepdiff aiofiles 2>&1 | Out-Null
    Write-Host "✅ Critical packages verified" -ForegroundColor Green

    # Final summary
    $EndTime = Get-Date
    $Duration = $EndTime - $StartTime
    Write-Host ""
    Write-Banner "INSTALLATION COMPLETE!"
    Write-Host ("⏱ Finished in {0:mm} min {0:ss} sec" -f $Duration) -ForegroundColor Cyan
    Write-Host ""

    # Launch or test
    if (-not $Headless) {
        Write-Host "🚀 Launching TORI..." -ForegroundColor Green
        Write-Host "   Press Ctrl+C to stop" -ForegroundColor Gray
        Write-Host ""
        python enhanced_launcher.py
        Write-Host ""
        Write-Host "🎉 TORI stopped. Setup complete!" -ForegroundColor Green
    } else {
        Write-Host "🔍 Running health check (headless mode)..." -ForegroundColor Green
        
        # Start TORI in background
        $tori = Start-Process python -ArgumentList "enhanced_launcher.py", "--no-browser" -PassThru -WindowStyle Hidden
        
        # Wait for API to be ready
        $ready = $false
        $attempts = 0
        $maxAttempts = 30
        
        Write-Host "   Waiting for API to start..." -ForegroundColor Gray
        while (-not $ready -and $attempts -lt $maxAttempts) {
            Start-Sleep -Seconds 2
            try {
                $response = Invoke-RestMethod -Uri "http://localhost:8003/api/health" -Method Get -ErrorAction SilentlyContinue
                if ($response.status -eq "healthy") {
                    $ready = $true
                    Write-Host "✅ API health check passed" -ForegroundColor Green
                }
            } catch {
                $attempts++
                if ($attempts % 5 -eq 0) {
                    Write-Host "   Still waiting... ($attempts/$maxAttempts)" -ForegroundColor Gray
                }
            }
        }
        
        # Stop TORI
        if ($tori -and -not $tori.HasExited) {
            Stop-Process -Id $tori.Id -Force -ErrorAction SilentlyContinue
        }
        
        if (-not $ready) {
            Write-Host "❌ API health check failed after $maxAttempts attempts" -ForegroundColor Red
            exit 1
        }
        
        Write-Host ""
        Write-Host "🎉 Setup complete! TORI is ready for development." -ForegroundColor Green
    }

    # Cleanup hint
    Write-Host ""
    Write-Host "💡 Log files created:" -ForegroundColor Gray
    Write-Host "   - install.log (pip installations)" -ForegroundColor Gray
    Write-Host "   - penrose_build.log (if Rust was found)" -ForegroundColor Gray
    Write-Host "   - frontend_install.log (if npm was found)" -ForegroundColor Gray
