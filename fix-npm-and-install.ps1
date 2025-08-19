Write-Host "Fixing NPM permission issues and installing dependencies..." -ForegroundColor Cyan

# Kill any processes that might be locking files
Write-Host "Stopping any Node processes..." -ForegroundColor Yellow
Get-Process node -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Get-Process npm -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue

# Remove problematic esbuild directory
Write-Host "Removing problematic directories..." -ForegroundColor Yellow
$problemDir = "C:\Users\jason\Desktop\tori\kha\node_modules\.esbuild-S1lfBe73"
if (Test-Path $problemDir) {
    Remove-Item -Path $problemDir -Recurse -Force -ErrorAction SilentlyContinue
}

# Clear npm cache
Write-Host "Clearing npm cache..." -ForegroundColor Yellow
npm cache clean --force

# Navigate to frontend directory
cd frontend

# Remove node_modules and package-lock if they exist
Write-Host "Cleaning old installations..." -ForegroundColor Yellow
if (Test-Path "node_modules") {
    Remove-Item -Path "node_modules" -Recurse -Force -ErrorAction SilentlyContinue
}
if (Test-Path "package-lock.json") {
    Remove-Item -Path "package-lock.json" -Force -ErrorAction SilentlyContinue
}

# Install dependencies
Write-Host "`nInstalling dependencies..." -ForegroundColor Green
npm install

# Install @webgpu/types specifically
Write-Host "`nInstalling @webgpu/types..." -ForegroundColor Green
npm install --save-dev @webgpu/types

Write-Host "`nDependencies installed successfully!" -ForegroundColor Green

cd ..
