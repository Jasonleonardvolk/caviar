# Fix PATH safely without truncation
$tintPath = "C:\Users\jason\Desktop\tori\kha\tools\tint"

# Get current USER PATH (not truncated)
$currentPath = [Environment]::GetEnvironmentVariable("Path", "User")

# Check if tint path already exists
if ($currentPath -notlike "*$tintPath*") {
    # Add tint to user PATH safely
    [Environment]::SetEnvironmentVariable("Path", "$currentPath;$tintPath", "User")
    Write-Host "SUCCESS: Added Tint to PATH safely (no truncation)" -ForegroundColor Green
} else {
    Write-Host "Tint path already in PATH" -ForegroundColor Yellow
}

# Verify
Write-Host ""
Write-Host "Current User PATH length: $($currentPath.Length) characters" -ForegroundColor Cyan
Write-Host ""

# Test if tint works
$tintExe = "$tintPath\tint.exe"
if (Test-Path $tintExe) {
    & $tintExe --version
    Write-Host "Tint is ready to use!" -ForegroundColor Green
} else {
    Write-Host "WARNING: tint.exe not found at $tintExe" -ForegroundColor Yellow
    Write-Host "Download from: https://github.com/google/dawn/releases" -ForegroundColor Yellow
}
