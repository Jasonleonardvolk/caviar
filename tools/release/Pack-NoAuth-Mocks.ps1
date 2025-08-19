Param(
  [switch]$DryRun,
  [switch]$NoBackup
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Info($m){ Write-Host "[Pack-NoAuth-Mocks] $m" -ForegroundColor Cyan }
function Ok($m){ Write-Host "[Pack-NoAuth-Mocks] OK: $m" -ForegroundColor Green }
function Warn($m){ Write-Host "[Pack-NoAuth-Mocks] WARN: $m" -ForegroundColor Yellow }
function Err($m){ Write-Host "[Pack-NoAuth-Mocks] ERROR: $m" -ForegroundColor Red }

# --- locate project root (expects ...\tori_ui_svelte\src\...) ---
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$candidates = @(
  (Join-Path $RepoRoot "tori_ui_svelte"),
  (Join-Path $RepoRoot "frontend"),
  $RepoRoot
)
$ProjectRoot = $null
foreach($c in $candidates){
  if(Test-Path (Join-Path $c "src")){ $ProjectRoot = $c; break }
}
if(-not $ProjectRoot){ Err "Could not find a project containing 'src' under: $($candidates -join '; ')" ; exit 2 }
Info "ProjectRoot: $ProjectRoot"

# --- helpers ---
function Ensure-Dir([string]$Path){
  if(-not (Test-Path $Path)){
    if(-not $DryRun){ New-Item -ItemType Directory -Path $Path -Force | Out-Null }
    Ok "mkdir $Path"
  }
}
function ReadText($p){ if(Test-Path $p){ return [IO.File]::ReadAllText($p) } else { return $null } }
function WriteText($p, $t){
  Ensure-Dir ([IO.Path]::GetDirectoryName($p))
  if(-not $DryRun){ [IO.File]::WriteAllText($p, $t, (New-Object System.Text.UTF8Encoding($false))) }
  Ok "write $p"
}
function Backup($p){
  if($NoBackup){ return }
  if(Test-Path $p){
    $b = "$p.bak"
    if(-not (Test-Path $b)){ if(-not $DryRun){ Copy-Item $p $b -Force } ; Info "backup $b" }
  }
}
function Upsert($relPath, $content, $label){
  $full = Join-Path $ProjectRoot $relPath
  $cur = ReadText $full
  if($cur -eq $content){ Info "nochange $label ($relPath)"; return }
  if($cur -ne $null){ Backup $full }
  WriteText $full $content
  Ok $label
}

# --- file contents ---

$home_redirect = @'
import { redirect } from '@sveltejs/kit';

export const load = () => {
  // iRis-first landing
  throw redirect(307, '/renderer');
};
'@

$api_health = @'
export const GET = async () =>
  new Response(JSON.stringify({ ok: true, ts: Date.now() }), {
    status: 200,
    headers: { "content-type": "application/json" },
  });
'@

$auth_ts = @'
import { dev } from '$app/environment';

/**
 * If IRIS_ALLOW_UNAUTH=1 (or we're in dev), allow all.
 * If INTERNAL_API_KEY is set, require header x-api-key to match it.
 * Otherwise allow.
 */
export function requireApiKey(request: Request) {
  if (dev || process.env.IRIS_ALLOW_UNAUTH === '1') return null;

  const expected = process.env.INTERNAL_API_KEY;
  if (!expected) return null;

  const got = request.headers.get('x-api-key');
  if (got !== expected) {
    return new Response('Unauthorized', { status: 401 });
  }
  return null;
}
'@

$safe_fetch = @'
export async function fetchWithTimeout(url: string, ms: number, init: RequestInit = {}) {
  const ctrl = new AbortController();
  const id = setTimeout(() => ctrl.abort(), ms);
  try {
    return await fetch(url, { ...init, signal: ctrl.signal });
  } finally {
    clearTimeout(id);
  }
}
'@

$api_list = @'
import { json } from '@sveltejs/kit';
import { fetchWithTimeout } from '$lib/server/safeFetch';
import * as fs from 'node:fs/promises';
import * as path from 'node:path';

const EXTRACTOR_BASE_URL = process.env.EXTRACTOR_BASE_URL; // e.g. http://localhost:8001
const LOCAL_UPLOAD_DIR = process.env.LOCAL_UPLOAD_DIR || 'var/uploads';

async function listLocalUploads() {
  await fs.mkdir(LOCAL_UPLOAD_DIR, { recursive: true });
  const entries = await fs.readdir(LOCAL_UPLOAD_DIR, { withFileTypes: true });
  return entries
    .filter((e) => e.isFile())
    .map((e) => ({ key: e.name, href: `/uploads/${encodeURIComponent(e.name)}` }));
}

export const GET = async () => {
  if (EXTRACTOR_BASE_URL) {
    try {
      const r = await fetchWithTimeout(`${EXTRACTOR_BASE_URL}/list`, 1500);
      if (r.ok) {
        const text = await r.text();
        return new Response(text, {
          status: 200,
          headers: { 'content-type': r.headers.get('content-type') || 'application/json' }
        });
      }
      console.warn('Extractor /list returned', r.status);
    } catch (err) {
      console.warn('Extractor /list failed; using local fallback', err);
    }
  }
  const items = await listLocalUploads();
  return json({ source: 'local', items });
};
'@

$upload_server = @'
import { json } from '@sveltejs/kit';
import * as fs from 'node:fs/promises';
import * as path from 'node:path';

const LOCAL_UPLOAD_DIR = process.env.LOCAL_UPLOAD_DIR || 'var/uploads';

export const POST = async ({ request }) => {
  await fs.mkdir(LOCAL_UPLOAD_DIR, { recursive: true });

  const ctype = request.headers.get('content-type') || '';
  let buf: Buffer;
  let filename = `upload-${Date.now()}.bin`;

  if (ctype.startsWith('multipart/form-data')) {
    const form = await request.formData();
    const file: any = form.get('file');
    if (!file?.arrayBuffer) return new Response('No file', { status: 400 });
    buf = Buffer.from(await file.arrayBuffer());
    filename = file.name || filename;
  } else {
    buf = Buffer.from(await request.arrayBuffer());
  }

  const safe = filename.replace(/[^\w.\-]+/g, '_');
  const full = path.join(LOCAL_UPLOAD_DIR, safe);
  await fs.writeFile(full, buf);

  return json({ ok: true, key: safe, size: buf.length });
};

export const GET = async () => {
  await fs.mkdir(LOCAL_UPLOAD_DIR, { recursive: true });
  const entries = await fs.readdir(LOCAL_UPLOAD_DIR, { withFileTypes: true });
  const files = entries.filter(e => e.isFile()).map(e => e.name);
  return json({ ok: true, files });
};
'@

$uploads_route = @'
import type { RequestHandler } from './$types';
import * as fs from 'node:fs/promises';
import * as path from 'node:path';

const LOCAL_UPLOAD_DIR = process.env.LOCAL_UPLOAD_DIR || 'var/uploads';

export const GET: RequestHandler = async ({ params }) => {
  const file = params.file;
  const full = path.join(LOCAL_UPLOAD_DIR, file);
  try {
    const data = await fs.readFile(full);
    const ext = path.extname(file).toLowerCase();
    const mime =
      ext === '.pdf' ? 'application/pdf' :
      ext === '.txt' ? 'text/plain; charset=utf-8' :
      (ext === '.png' ? 'image/png' :
      ext === '.jpg' || ext === '.jpeg' ? 'image/jpeg' :
      'application/octet-stream');
    return new Response(data, { status: 200, headers: { 'content-type': mime } });
  } catch {
    return new Response('Not found', { status: 404 });
  }
};
'@

$pdf_stats = @'
import { json } from '@sveltejs/kit';
import { dev } from '$app/environment';
import { requireApiKey } from '$lib/server/auth';

export const GET = async ({ request }) => {
  const unauthorized = requireApiKey(request);
  if (unauthorized) return unauthorized;

  const useMocks = dev || process.env.IRIS_USE_MOCKS === '1';
  if (useMocks) {
    return json({ ok: true, stats: { pages: 0, docs: 0, updated: Date.now() }, mock: true });
  }

  // TODO: implement real stats
  return new Response('Not configured', { status: 503 });
};
'@

$memory_state = @'
import { json } from '@sveltejs/kit';
import { dev } from '$app/environment';
import { requireApiKey } from '$lib/server/auth';

export const GET = async ({ request }) => {
  const unauthorized = requireApiKey(request);
  if (unauthorized) return unauthorized;

  const useMocks = dev || process.env.IRIS_USE_MOCKS === '1';
  if (useMocks) {
    return json({ ok: true, state: { personas: [], metrics: {} }, mock: true });
  }

  // TODO: implement real state
  return new Response('Not configured', { status: 503 });
};
'@

# --- apply files ---

Upsert "src\routes\+page.server.ts"                $home_redirect "root redirect → /renderer"
Upsert "src\routes\api\health\+server.ts"          $api_health    "api/health endpoint"
Upsert "src\lib\server\auth.ts"                    $auth_ts       "auth helper (optional key)"
Upsert "src\lib\server\safeFetch.ts"               $safe_fetch    "safe fetch with timeout"
Upsert "src\routes\api\list\+server.ts"            $api_list      "api/list with local fallback"
Upsert "src\routes\upload\+server.ts"              $upload_server "upload handler (local dir)"
Upsert "src\routes\uploads\[file]\+server.ts"      $uploads_route "serve /uploads/<file> from disk"
Upsert "src\routes\api\pdf\stats\+server.ts"       $pdf_stats     "api/pdf/stats (mockable)"
Upsert "src\routes\api\memory\state\+server.ts"    $memory_state  "api/memory/state (mockable)"

# --- .env.local ensure keys ---
$envPath = Join-Path $ProjectRoot ".env.local"
$envBody = if(Test-Path $envPath){ [IO.File]::ReadAllText($envPath) } else { "" }

function EnsureEnvLine([string]$body, [string]$key, [string]$value){
  $pattern = "^\s*$([regex]::Escape($key))\s*="
  if($body -match $pattern){
    # replace line
    return [Regex]::Replace($body, "$pattern.*$", "$key=$value", [Text.RegularExpressions.RegexOptions]::Multiline)
  } else {
    if($body -and -not $body.EndsWith("`r`n")){ $body += "`r`n" }
    return $body + "$key=$value`r`n"
  }
}

$envBody = EnsureEnvLine $envBody "IRIS_ALLOW_UNAUTH" "1"
$envBody = EnsureEnvLine $envBody "IRIS_USE_MOCKS" "1"
$envBody = EnsureEnvLine $envBody "LOCAL_UPLOAD_DIR" "var/uploads"

if(Test-Path $envPath){
  $current = [IO.File]::ReadAllText($envPath)
  if($current -ne $envBody){ if(-not $NoBackup){ Copy-Item $envPath "$envPath.bak" -Force ; Info "backup $envPath.bak" } }
}
if(-not $DryRun){ [IO.File]::WriteAllText($envPath, $envBody, (New-Object System.Text.UTF8Encoding($false))) }
Ok ".env.local updated"

Ok "Done. Next:
  1) cd $ProjectRoot
  2) pnpm run build
  3) node .\build\index.js
  4) Run smoke: D:\Dev\kha\tools\release\Smoke-Test.ps1 -BaseUrl 'http://localhost:3000'"
