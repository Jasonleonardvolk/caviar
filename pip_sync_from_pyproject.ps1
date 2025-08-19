# Save as: .\pip_sync_from_pyproject.ps1
# Activates in the current venv and matches only main dependencies (not dev/extras)

$pyproject = Get-Content ".\pyproject.toml"

# Find the dependencies section
$depsBlock = $pyproject | Select-String "^\[tool.poetry.dependencies\]" -Context 0,1000
$startIdx = $depsBlock.LineNumber
$lines = $pyproject[$startIdx..($pyproject.Length - 1)]

$pipInstallCmd = "pip install"

foreach ($line in $lines) {
    if ($line -match "^\[.*\]") { break } # Stop at the next section
    if ($line -match "^\s*#") { continue } # Skip comments
    if ($line -match "^\s*$") { continue } # Skip blanks
    if ($line -match "^\s*python") { continue } # Skip python itself

    # Match "package = "version""
    if ($line -match '^\s*([^ ]+)\s*=\s*["'']([^"'']+)["'']') {
        $pkg = $matches[1]
        $ver = $matches[2]
        # Remove extras (e.g., requests = { version = "2.32.3", ... })
        if ($ver -match "^{") { continue }
        # Remove poetry specifiers like ^, ~, >=, <=
        $verClean = $ver -replace '[\^~>=< ]', ''
        $pipInstallCmd += " $pkg==$verClean"
    }
}

Write-Host "`n⏳ Installing pinned dependencies:"
Write-Host $pipInstallCmd
Invoke-Expression $pipInstallCmd
Write-Host "`n✅ All main dependencies from pyproject.toml are now pip-pinned."
