# scripts\bootstrap.ps1
Set-StrictMode -Version Latest
$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Definition)

# 1) Node & Yarn via Volta
if (-not (Get-Command volta -ErrorAction SilentlyContinue)) {
  Write-Host "Installing Volta for Node & Yarn version management..." -ForegroundColor Cyan
  iwr https://get.volta.sh -useb | Invoke-Expression
}
Write-Host "Setting up Node 18.20.3 and Yarn 3.7.1 via Volta..." -ForegroundColor Cyan
volta install node@18.20.3 yarn@3.7.1   # pins binaries in %USERPROFILE%\.volta

# 2) Python 3.11
if (-not (Get-Command pyenv -ErrorAction SilentlyContinue)) {
  Write-Host "pyenv not found. Please install pyenv-win first:" -ForegroundColor Yellow
  Write-Host "https://github.com/pyenv-win/pyenv-win#installation" -ForegroundColor Yellow
  Write-Host "Then run this script again." -ForegroundColor Yellow
  exit 1
}

Write-Host "Setting up Python 3.11.9 via pyenv..." -ForegroundColor Cyan
if (-not (pyenv versions | Select-String '3.11.9')) {
  pyenv install 3.11.9
}
pyenv local 3.11.9
python -m pip install --upgrade pip virtualenv
python -m virtualenv .venv
& .\.venv\Scripts\Activate.ps1

if (Test-Path "$repo\alan_core") {
  Write-Host "Installing alan_core package..." -ForegroundColor Cyan
  pip install -e "$repo\alan_core\"
}

# 3) JavaScript deps (zero-install is basically a checksum)
Write-Host "Installing JavaScript dependencies..." -ForegroundColor Cyan
yarn install --immutable --immutable-cache

Write-Host "`n✅ Environment ready. Run 'yarn dev' to launch the ALAN IDE." -ForegroundColor Green
