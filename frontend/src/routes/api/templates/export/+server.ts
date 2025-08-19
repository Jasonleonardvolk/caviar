import type { RequestHandler } from './$types';
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import archiver from 'archiver';

const ROOT = path.resolve(process.cwd(), '..');
const EXPORTS_DIR  = path.join(ROOT, 'exports');
const TEMPLATES_DIR = path.join(EXPORTS_DIR, 'templates');
const TEX_OUT_DIR   = path.join(EXPORTS_DIR, 'textures_ktx2');

function ensureDir(dir: string) { if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true }); }

function run(cmd: string, args: string[], cwd: string) {
  const p = spawnSync(cmd, args, { cwd, shell: true, stdio: 'pipe', env: process.env });
  if (p.status !== 0) {
    throw new Error(`${cmd} ${args.join(' ')} failed: ${p.stderr?.toString() || p.stdout?.toString()}`);
  }
  return String(p.stdout || '');
}

function timestamp() {
  return new Date().toISOString().replace(/[:.]/g, '-');
}

export const POST: RequestHandler = async ({ request, url }) => {
  // Accept body (JSON) or form (www-form / multipart) or query (?zip=1)
  const contentType = request.headers.get('content-type') || '';
  const isForm = contentType.includes('application/x-www-form-urlencoded') || contentType.includes('multipart/form-data');

  let input = 'D:\\Dev\\kha\\data\\concept_graph.json';
  let layout = 'grid';
  let scale = '0.12';
  let zip = url.searchParams.get('zip') === '1';

  if (isForm) {
    const form = await request.formData();
    input  = String(form.get('input')  ?? input);
    layout = String(form.get('layout') ?? layout);
    scale  = String(form.get('scale')  ?? scale);
  } else if (contentType.includes('application/json')) {
    try {
      const j = await request.json();
      input  = String(j?.input  ?? input);
      layout = String(j?.layout ?? layout);
      scale  = String(j?.scale  ?? scale);
      zip    = Boolean(j?.zip ?? zip);
    } catch {}
  }

  ensureDir(TEMPLATES_DIR);
  ensureDir(TEX_OUT_DIR);

  const outGlb = path.join(TEMPLATES_DIR, `concept-${timestamp()}.glb`);

  // 1) GLB from concept mesh (via PACK-M2 exporter).
  run('pnpm', [
    'dlx', 'tsx', 'tools/exporters/glb-from-conceptmesh.ts',
    '-i', input,
    '-o', outGlb,
    '--layout', layout,
    '--scale', scale
  ], ROOT);

  // 2) KTX2 encode (BasisU PowerShell wrapper from PACK-M2).
  //    You can override InputDir via JSON body if needed.
  const texIn = path.join(ROOT, 'assets', 'textures');
  run('powershell', [
    '-ExecutionPolicy', 'Bypass',
    '-File', 'tools/exporters/encode-ktx2.ps1',
    '-InputDir', texIn,
    '-OutDir', TEX_OUT_DIR
  ], ROOT);

  // 3) If zip requested, stream a bundle (GLB + textures_ktx2/*) back to caller.
  if (zip) {
    const zipName = `template-bundle-${timestamp()}.zip`;
    const headers = new Headers({
      'Content-Type': 'application/zip',
      'Content-Disposition': `attachment; filename="${zipName}"`
    });

    // Stream zip
    const stream = new ReadableStream({
      start(controller) {
        const archive = archiver('zip', { zlib: { level: 9 } });
        archive.on('data', (chunk) => controller.enqueue(chunk));
        archive.on('warning', (err) => console.warn('archiver warn:', err));
        archive.on('error', (err) => controller.error(err));
        archive.on('end', () => controller.close());

        archive.file(outGlb, { name: path.basename(outGlb) });
        if (fs.existsSync(TEX_OUT_DIR)) {
          archive.directory(TEX_OUT_DIR, 'textures_ktx2');
        }
        archive.finalize().catch((e) => controller.error(e));
      }
    });

    return new Response(stream, { headers });
  }

  // Otherwise return JSON manifest
  return new Response(JSON.stringify({
    ok: true,
    glb: outGlb,
    texturesDir: TEX_OUT_DIR
  }), { headers: { 'content-type': 'application/json' }});
};