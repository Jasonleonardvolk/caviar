# Verify-Iris.ps1
# Dev/Prod smoke checks for iRis hologram system

param(
    [Parameter(Position=0)]
    [ValidateSet('dev', 'prod', 'both')]
    [string]$Environment = 'both'
)

Write-Host "`n=== iRis Verification Suite ===" -ForegroundColor Cyan

function Test-Endpoint {
    param(
        [string]$Url,
        [string]$Name
    )
    
    try {
        $response = Invoke-WebRequest -Uri $Url -TimeoutSec 5 -ErrorAction Stop
        Write-Host "[OK] $Name : $Url" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "[FAIL] $Name : $Url" -ForegroundColor Red
        Write-Host "  Error: $_" -ForegroundColor Yellow
        return $false
    }
}

function Test-DevEnvironment {
    Write-Host "`nChecking Dev Environment..." -ForegroundColor Cyan
    
    $devChecks = @(
        @{Url = "http://localhost:5173"; Name = "Frontend Dev Server"},
        @{Url = "http://localhost:5173/hologram?show=wow"; Name = "Hologram Route"},
        @{Url = "http://localhost:8002/health"; Name = "Backend API Health"}
    )
    
    $passed = 0
    foreach ($check in $devChecks) {
        if (Test-Endpoint -Url $check.Url -Name $check.Name) {
            $passed++
        }
    }
    
    Write-Host "`nDev Results: $passed/$($devChecks.Count) passed" -ForegroundColor $(if ($passed -eq $devChecks.Count) {'Green'} else {'Yellow'})
}

function Test-ProdEnvironment {
    Write-Host "`nChecking Prod Environment (SSR)..." -ForegroundColor Cyan
    
    $prodChecks = @(
        @{Url = "http://localhost:3000"; Name = "SSR Frontend"},
        @{Url = "http://localhost:3000/hologram?show=wow"; Name = "SSR Hologram Route"},
        @{Url = "http://localhost:8002/api/health"; Name = "Production API"}
    )
    
    $passed = 0
    foreach ($check in $prodChecks) {
        if (Test-Endpoint -Url $check.Url -Name $check.Name) {
            $passed++
        }
    }
    
    Write-Host "`nProd Results: $passed/$($prodChecks.Count) passed" -ForegroundColor $(if ($passed -eq $prodChecks.Count) {'Green'} else {'Yellow'})
}

# Main execution
switch ($Environment) {
    'dev'  { Test-DevEnvironment }
    'prod' { Test-ProdEnvironment }
    'both' { 
        Test-DevEnvironment
        Test-ProdEnvironment
    }
}

Write-Host "`n=== Verification Complete ===" -ForegroundColor Cyan