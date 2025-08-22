# Update-PackageForIris.ps1
# Updates package.json with iRis repository information

$ErrorActionPreference = "Stop"

Write-Host "`n========================================" -ForegroundColor Yellow
Write-Host "Updating package.json for iRis" -ForegroundColor Yellow
Write-Host "========================================`n" -ForegroundColor Yellow

# Read current package.json
$packagePath = "package.json"
$packageContent = Get-Content $packagePath -Raw | ConvertFrom-Json

# Update package information
Write-Host "[INFO] Updating package metadata..." -ForegroundColor Cyan

# Update name to iRis
$packageContent.name = "iris"

# Add or update repository field
if (-not $packageContent.PSObject.Properties['repository']) {
    $packageContent | Add-Member -NotePropertyName repository -NotePropertyValue @{} -Force
}

$packageContent.repository = @{
    type = "git"
    url = "git+https://github.com/Jasonleonardvolk/iRis.git"
}

# Add or update homepage
$packageContent | Add-Member -NotePropertyName homepage -NotePropertyValue "https://github.com/Jasonleonardvolk/iRis#readme" -Force

# Add or update bugs URL
if (-not $packageContent.PSObject.Properties['bugs']) {
    $packageContent | Add-Member -NotePropertyName bugs -NotePropertyValue @{} -Force
}
$packageContent.bugs = @{
    url = "https://github.com/Jasonleonardvolk/iRis/issues"
}

# Add description if missing
if (-not $packageContent.PSObject.Properties['description']) {
    $packageContent | Add-Member -NotePropertyName description -NotePropertyValue "iRis - Holographic UI with memory synthesis" -Force
}

# Add keywords
if (-not $packageContent.PSObject.Properties['keywords']) {
    $packageContent | Add-Member -NotePropertyName keywords -NotePropertyValue @() -Force
}
$packageContent.keywords = @(
    "iris",
    "holographic",
    "ui",
    "svelte",
    "sveltekit",
    "webgpu",
    "memory-synthesis"
)

# Add author
if (-not $packageContent.PSObject.Properties['author']) {
    $packageContent | Add-Member -NotePropertyName author -NotePropertyValue "Jason Leonard Volk" -Force
}

# Add license
if (-not $packageContent.PSObject.Properties['license']) {
    $packageContent | Add-Member -NotePropertyName license -NotePropertyValue "MIT" -Force
}

# Convert back to JSON with proper formatting
$jsonContent = $packageContent | ConvertTo-Json -Depth 10

# Write back to file
$jsonContent | Set-Content $packagePath -Encoding UTF8

Write-Host "[OK] package.json updated successfully" -ForegroundColor Green

# Show what was updated
Write-Host "`nUpdated fields:" -ForegroundColor Cyan
Write-Host "  - name: iris" -ForegroundColor White
Write-Host "  - repository: https://github.com/Jasonleonardvolk/iRis.git" -ForegroundColor White
Write-Host "  - homepage: https://github.com/Jasonleonardvolk/iRis#readme" -ForegroundColor White
Write-Host "  - bugs: https://github.com/Jasonleonardvolk/iRis/issues" -ForegroundColor White
Write-Host "  - description: iRis - Holographic UI with memory synthesis" -ForegroundColor White
Write-Host "  - keywords: [iris, holographic, ui, ...]" -ForegroundColor White

Write-Host "`nNext step: Commit this change" -ForegroundColor Yellow
Write-Host "  git add package.json" -ForegroundColor Gray
Write-Host "  git commit -m `"chore(config): update package.json for iRis repository`"" -ForegroundColor Gray

Write-Host ""
