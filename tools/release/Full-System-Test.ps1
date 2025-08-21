param([string]$ProjectRoot = "D:\Dev\kha")

$ErrorActionPreference="Stop"
function Ok($m){Write-Host "[OK] $m" -f Green}
function Info($m){Write-Host "[i] $m" -f Cyan}
function Warn($m){Write-Host "[!] $m" -f Yellow}
function Fail($m){Write-Host "[X] $m" -f Red}

Info "========================================="
Info "  CAVIAR/KHA Complete System Test Suite"
Info "========================================="

$ReportDir = Join-Path $ProjectRoot "verification_reports"
New-Item -ItemType Directory -Force -Path $ReportDir | Out-Null
$Report = Join-Path $ReportDir ("full_system_test_{0}.json" -f (Get-Date -Format "yyyyMMdd_HHmmss"))

# Find dev port
function Up($port){ 
  try { 
    (Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$port/health/ping" -TimeoutSec 3).StatusCode -eq 200 
  } catch { $false } 
}

$ports=@($env:IRIS_DEV_PORT,5173,3000,3310) | Where-Object { $_ }
$port=$null
foreach($p in $ports){ if(Up $p){ $port=$p; break } }

if(-not $port){ 
  Fail "No dev server running. Start with: cd frontend && pnpm.cmd dev"
  exit 1
}

$base = "http://127.0.0.1:$port"
Info "Testing server at $base"
Info ""

# Test function
function TestEndpoint($method, $path, $body=$null, $contentType="application/json") {
  $uri = "$base$path"
  try {
    $params = @{
      Uri = $uri
      Method = $method
      TimeoutSec = 10
      UseBasicParsing = $true
    }
    if ($body) {
      $params.Body = $body
      $params.ContentType = $contentType
    }
    $response = Invoke-WebRequest @params
    return @{ok=$true; status=$response.StatusCode; content=$response.Content}
  } catch {
    $status = if ($_.Exception.Response) { [int]$_.Exception.Response.StatusCode } else { 0 }
    return @{ok=$false; status=$status; error=$_.Exception.Message}
  }
}

$testResults = @{}

# 1. Core Routes
Info "1. Testing Core Routes..."
$routes = @(
  @{name="Homepage"; path="/"},
  @{name="Hologram"; path="/hologram"},
  @{name="Templates"; path="/templates"},
  @{name="Pricing"; path="/pricing"}
)

$routeResults = @()
foreach($r in $routes) {
  $result = TestEndpoint -method "GET" -path $r.path
  if ($result.ok) {
    Ok "  $($r.name) ($($r.path)) → $($result.status)"
  } else {
    Fail "  $($r.name) ($($r.path)) → $($result.status): $($result.error)"
  }
  $routeResults += @{name=$r.name; path=$r.path; result=$result}
}
$testResults.routes = $routeResults

# 2. API Endpoints
Info ""
Info "2. Testing API Endpoints..."

# Health check
$health = TestEndpoint -method "GET" -path "/health"
if ($health.ok) {
  Ok "  Health endpoint → $($health.status)"
  try {
    $healthData = $health.content | ConvertFrom-Json
    Info "    Status: $($healthData.status)"
    Info "    Demo Ready: $($healthData.demoReady)"
  } catch {}
} else {
  Fail "  Health endpoint → Failed"
}
$testResults.health = $health

# Export API with concept graph
$exportBody = '{"input":"data\\concept_graph.json","layout":"grid","scale":0.12}'
$export = TestEndpoint -method "POST" -path "/api/templates/export" -body $exportBody
if ($export.ok) {
  Ok "  Export API → $($export.status)"
  try {
    $exportData = $export.content | ConvertFrom-Json
    if ($exportData.ok) {
      Ok "    Input validated: $($exportData.input)"
    }
  } catch {}
} else {
  Fail "  Export API → Failed: $($export.error)"
}
$testResults.export = $export

# 3. Static Assets
Info ""
Info "3. Testing Static Assets..."
$assets = @(
  "/favicon.ico",
  "/robots.txt"
)

$assetResults = @()
foreach($a in $assets) {
  $result = TestEndpoint -method "GET" -path $a
  if ($result.ok) {
    Ok "  $a → $($result.status)"
  } else {
    Warn "  $a → Not found (optional)"
  }
  $assetResults += @{path=$a; result=$result}
}
$testResults.assets = $assetResults

# 4. WebGL/GPU Capabilities
Info ""
Info "4. Checking WebGL/GPU Support..."
$hologram = TestEndpoint -method "GET" -path "/hologram"
if ($hologram.ok -and $hologram.content -match "webgpu|WebGPU|webgl|WebGL") {
  Ok "  WebGL/WebGPU references found in hologram page"
  $testResults.webgl = $true
} else {
  Warn "  WebGL/WebGPU references not detected"
  $testResults.webgl = $false
}

# 5. File System Check
Info ""
Info "5. Verifying Project Structure..."
$criticalPaths = @(
  "frontend\src\routes\hologram\+page.svelte",
  "frontend\src\lib\components\HologramRecorder.svelte",
  "data\concept_graph.json",
  "tools\release\Quick-Verify.ps1"
)

$fileResults = @()
foreach($p in $criticalPaths) {
  $fullPath = Join-Path $ProjectRoot $p
  if (Test-Path $fullPath) {
    Ok "  ✓ $p"
    $fileResults += @{path=$p; exists=$true}
  } else {
    Fail "  ✗ $p"
    $fileResults += @{path=$p; exists=$false}
  }
}
$testResults.files = $fileResults

# 6. Performance Metrics
Info ""
Info "6. Performance Metrics..."
$perfTests = @()
foreach($i in 1..3) {
  $start = Get-Date
  $result = TestEndpoint -method "GET" -path "/hologram"
  $duration = ((Get-Date) - $start).TotalMilliseconds
  $perfTests += $duration
}
$avgTime = ($perfTests | Measure-Object -Average).Average
Info "  Average response time: $([math]::Round($avgTime, 2))ms"
if ($avgTime -lt 500) {
  Ok "  Performance: Excellent (<500ms)"
} elseif ($avgTime -lt 1000) {
  Warn "  Performance: Good (<1000ms)"
} else {
  Fail "  Performance: Slow (>1000ms)"
}
$testResults.performance = @{tests=$perfTests; average=$avgTime}

# Summary
Info ""
Info "========================================="
$totalTests = 0
$passedTests = 0

foreach($r in $routeResults) {
  $totalTests++
  if ($r.result.ok) { $passedTests++ }
}
foreach($f in $fileResults) {
  $totalTests++
  if ($f.exists) { $passedTests++ }
}
if ($health.ok) { $passedTests++ }
$totalTests++
if ($export.ok) { $passedTests++ }
$totalTests++

$successRate = [math]::Round(($passedTests / $totalTests) * 100, 1)

if ($successRate -eq 100) {
  Ok "✅ PERFECT! All $totalTests tests passed (100%)"
} elseif ($successRate -ge 80) {
  Ok "✅ PASSED: $passedTests/$totalTests tests ($successRate%)"
} elseif ($successRate -ge 60) {
  Warn "⚠️ PARTIAL: $passedTests/$totalTests tests ($successRate%)"
} else {
  Fail "❌ FAILED: $passedTests/$totalTests tests ($successRate%)"
}

# Save report
$fullReport = @{
  timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
  server = $base
  summary = @{
    total = $totalTests
    passed = $passedTests
    failed = $totalTests - $passedTests
    successRate = $successRate
  }
  results = $testResults
}

$fullReport | ConvertTo-Json -Depth 10 | Set-Content $Report
Info ""
Info "Full report saved: $Report"
Info "========================================="

# Exit code based on success
if ($successRate -ge 80) { exit 0 } else { exit 1 }