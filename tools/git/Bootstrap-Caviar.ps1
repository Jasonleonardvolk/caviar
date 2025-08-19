# Bootstrap-Caviar.ps1 (PS 5.1 compatible)
[CmdletBinding()]
param(
  [string]$RepoRoot = "D:\Dev\kha",
  [string]$OrgOrUser = "Jasonleonardvolk",
  [string]$RepoName = "caviar",
  [switch]$Private
)

$ErrorActionPreference = "Stop"
Set-Location $RepoRoot

Write-Host "=== Bootstrap NEW clean repo ===" -ForegroundColor Cyan
Write-Host "Local root: $RepoRoot" -ForegroundColor Cyan
Write-Host "GitHub: $OrgOrUser/$RepoName" -ForegroundColor Cyan

# 0) Pause Drive sync manually if possible

# 1) Remove existing .git (clear attributes first; then delete)
if (Test-Path ".git") {
  cmd /c "attrib -r -h -s /s /d .git"
  Remove-Item -Recurse -Force ".git"
}

# 2) Ensure .gitignore (keep existing; else write minimal)
if (-not (Test-Path ".gitignore")) {
@"
conversations/
data/
docs/
.venv/
venv/
__pycache__/
*.pyc
node_modules/
tori_ui_svelte/.svelte-kit/
tori_ui_svelte/build/
tori_ui_svelte/dist/
**/target/
tools/ffmpeg/*.exe
tools/ffmpeg/*.dll
.tmp.driveupload/
*.log
*.tmp
*.temp
_NONREPO_LARGE/
"@ | Set-Content -Encoding UTF8 -NoNewline ".gitignore"
}

# 3) Evict >100MB files so push won't be rejected
$threshold = 100MB
$largeDir = Join-Path $RepoRoot "_NONREPO_LARGE"
$big = Get-ChildItem -Recurse -File -ErrorAction SilentlyContinue | Where-Object { $_.Length -ge $threshold }
if ($big -and $big.Count -gt 0) {
  Write-Host "Moving $($big.Count) file(s) >= 100MB to: $largeDir" -ForegroundColor Yellow
  New-Item -ItemType Directory -Force -Path $largeDir | Out-Null
  foreach ($f in $big) {
    $rel = $f.FullName.Substring($RepoRoot.Length).TrimStart('\','/')
    $dest = Join-Path $largeDir $rel
    New-Item -ItemType Directory -Force -Path (Split-Path $dest -Parent) | Out-Null
    Move-Item -Force $f.FullName $dest
  }
  Add-Content ".gitignore" "`r`n_NONREPO_LARGE/`r`n"
}

# 4) Create new repo with gh if available; else expect pre-created remote
$slug = "$OrgOrUser/$RepoName"
$vis = if ($Private.IsPresent) { "--private" } else { "--public" }

$gh = Get-Command gh -ErrorAction SilentlyContinue
if ($gh) {
  try { gh repo view $slug 2>$null | Out-Null } catch {
    Write-Host "Creating repo on GitHub: $slug" -ForegroundColor Green
    gh repo create $slug $vis --source "." --disable-issues --disable-wiki --homepage "https://github.com/$slug" --push
    Write-Host "✅ Clean repo pushed to https://github.com/$slug" -ForegroundColor Green
    exit 0
  }
}

# 5) Init, commit, push to existing remote (manual case)
git init | Out-Null
git symbolic-ref HEAD refs/heads/main
git remote add origin "https://github.com/$slug.git"
git config core.longpaths true

git add -A
git commit -m "initial clean slate (bootstrap caviar)"
git push -u origin main

Write-Host "✅ Clean repo pushed to https://github.com/$slug" -ForegroundColor Green
