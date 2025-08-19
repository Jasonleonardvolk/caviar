# Fix the shader bundle generation issue
param(
    [string]$RepoRoot = "D:\Dev\kha"
)

Push-Location $RepoRoot

Write-Host "Fixing shader bundle generation..." -ForegroundColor Cyan
Write-Host ""

# First, let's backup the current broken file
$brokenFile = ".\frontend\lib\webgpu\generated\shaderSources.ts"
$backupFile = ".\frontend\lib\webgpu\generated\shaderSources.ts.broken"

if (Test-Path $brokenFile) {
    Write-Host "Backing up broken file..." -ForegroundColor Yellow
    Copy-Item $brokenFile $backupFile -Force
}

# Run the bundler script
Write-Host "Running shader bundler..." -ForegroundColor Cyan
try {
    npx tsx scripts/bundleShaders.ts
    $bundlerResult = $LASTEXITCODE
} catch {
    Write-Host "Error running bundler with tsx, trying ts-node..." -ForegroundColor Yellow
    try {
        npx ts-node scripts/bundleShaders.ts
        $bundlerResult = $LASTEXITCODE
    } catch {
        Write-Host "Both tsx and ts-node failed. Let me create a quick fix..." -ForegroundColor Red
        $bundlerResult = 1
    }
}

# If bundler failed, create a minimal valid file
if ($bundlerResult -ne 0) {
    Write-Host "Bundler failed. Creating minimal valid shader sources file..." -ForegroundColor Yellow
    
    $minimalContent = @'
// Auto-generated minimal shader sources (bundler failed)
// Generated: ' + (Get-Date -Format "yyyy-MM-ddTHH:mm:ss.fffZ") + '

// Temporary empty exports to fix TypeScript errors
export const shaderSources = {};
export const shaderMetadata = {
  generated: "' + (Get-Date -Format "yyyy-MM-ddTHH:mm:ss.fffZ") + '",
  totalShaders: 0,
  validShaders: 0,
  shaderDir: "",
  shaders: {}
};

export type ShaderName = keyof typeof shaderSources;
export type ShaderMap = typeof shaderSources;

export function getShader(name: ShaderName): string {
  throw new Error(`Shader bundle not properly generated. Run: npx tsx scripts/bundleShaders.ts`);
}

export default shaderSources;
'@

    Set-Content -Path $brokenFile -Value $minimalContent -Encoding UTF8
    Write-Host "Created minimal valid file to fix TypeScript errors" -ForegroundColor Green
}

# Check if TypeScript still has errors
Write-Host ""
Write-Host "Checking TypeScript compilation..." -ForegroundColor Cyan
npx tsc -p .\frontend\tsconfig.json --noEmit 2>&1 | Select-String "shaderSources" | Select-Object -First 5

Pop-Location

if ($bundlerResult -eq 0) {
    Write-Host ""
    Write-Host "✅ Shader bundle regenerated successfully!" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "⚠️ Created temporary fix for TypeScript errors" -ForegroundColor Yellow
    Write-Host "To properly fix, ensure you have tsx or ts-node installed:" -ForegroundColor Yellow
    Write-Host "  npm install -D tsx" -ForegroundColor Gray
    Write-Host "  npx tsx scripts/bundleShaders.ts" -ForegroundColor Gray
}
