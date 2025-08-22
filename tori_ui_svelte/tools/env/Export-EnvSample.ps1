New-Item -ItemType Directory -Force "$PSScriptRoot" | Out-Null
$uiRoot = (Resolve-Path "$PSScriptRoot\..\..").Path
$dest   = Join-Path $uiRoot ".env.sample"

$Port     = if ($env:PORT) { $env:PORT } else { "3000" }
$Mocks    = if ($env:IRIS_USE_MOCKS) { $env:IRIS_USE_MOCKS } else { "0" }
$PdfUrl   = if ($env:IRIS_PDF_SERVICE_URL) { $env:IRIS_PDF_SERVICE_URL } else { "http://127.0.0.1:7401" }
$VaultUrl = if ($env:IRIS_MEMORY_VAULT_URL) { $env:IRIS_MEMORY_VAULT_URL } else { "http://127.0.0.1:7501" }
$Storage  = if ($env:IRIS_STORAGE_TYPE) { $env:IRIS_STORAGE_TYPE } else { "local" }

@"
# iRIS runtime config (copy to .env and edit)
PORT=$Port
IRIS_USE_MOCKS=$Mocks
IRIS_PDF_SERVICE_URL=$PdfUrl
IRIS_MEMORY_VAULT_URL=$VaultUrl
IRIS_STORAGE_TYPE=$Storage
"@ | Set-Content -Encoding UTF8 $dest

Write-Host "[env] Wrote $dest"
