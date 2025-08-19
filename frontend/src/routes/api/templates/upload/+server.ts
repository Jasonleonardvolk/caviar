import type { RequestHandler } from './$types';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import path from 'node:path';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { spawnSync } from 'node:child_process';

const ROOT = path.resolve(process.cwd(), '..'); // → D:\Dev\kha (dev)
const TMP_DIR = path.join(ROOT, 'tmp', 'uploads');
const TEMPLATES_DIR = path.join(ROOT, 'exports', 'templates');

function ensureDir(p: string) {
  if (!fs.existsSync(p)) fs.mkdirSync(p, { recursive: true });
}

function sanitizeBase(name: string) {
  return name.replace(/[^\w\.\-]+/g, '_'); // no path traversal
}

function runNode(cmd: string, args: string[], cwd: string) {
  const p = spawnSync(cmd, args, { cwd, shell: true, stdio: 'pipe', env: process.env });
  if (p.status !== 0) {
    throw new Error(`${cmd} ${args.join(' ')} failed: ${p.stderr?.toString() || p.stdout?.toString()}`);
  }
  return String(p.stdout || '');
}

async function writeSidecar(glbPath: string, meta: Record<string, unknown>) {
  const sidecar = glbPath.replace(/\.glb$/i, '.template.json');
  await fsp.writeFile(sidecar, JSON.stringify(meta, null, 2), 'utf8');
  return sidecar;
}

export const POST: RequestHandler = async ({ request }) => {
  const ct = request.headers.get('content-type') || '';
  if (!ct.includes('multipart/form-data')) {
    return new Response('Expected multipart/form-data', { status: 400 });
  }

  ensureDir(TMP_DIR);
  ensureDir(TEMPLATES_DIR);

  const form = await request.formData();
  const file = form.get('file') as unknown as File | null;
  const explicitName = String(form.get('name') || '').trim();
  const mode = String(form.get('mode') || '').trim().toLowerCase(); // 'glb' | 'concept'
  const description = String(form.get('description') || '').trim();
  const tags = String(form.get('tags') || '').split(',').map(s => s.trim()).filter(Boolean);

  if (!file) return new Response('file required', { status: 400 });

  const origName = sanitizeBase((file as any).name || 'upload.bin');
  const ext = path.extname(origName).toLowerCase();
  const isGLB = ext === '.glb' || mode === 'glb';
  const isJSON = ext === '.json' || mode === 'concept';

  if (!isGLB && !isJSON) {
    return new Response('Only .glb or .json (concept mesh) allowed', { status: 415 });
  }

  // Save upload to tmp
  const tmpName = sanitizeBase(explicitName || origName);
  const tmpPath = path.join(TMP_DIR, tmpName);
  const nodeReadable = Readable.fromWeb(file.stream() as any);
  await pipeline(nodeReadable, fs.createWriteStream(tmpPath));

  let glbOut: string;
  if (isGLB) {
    // Move GLB into exports/templates
    const base = tmpName.toLowerCase().endsWith('.glb') ? tmpName : `${tmpName}.glb`;
    glbOut = path.join(TEMPLATES_DIR, base);
    await fsp.copyFile(tmpPath, glbOut);
  } else {
    // Convert concept JSON → GLB via PACK-M2 exporter
    const base = (tmpName.toLowerCase().endsWith('.json') ? tmpName.slice(0, -5) : tmpName) || 'concept';
    glbOut = path.join(TEMPLATES_DIR, `${base}.glb`);
    runNode('pnpm', ['dlx', 'tsx', 'tools/exporters/glb-from-conceptmesh.ts',
      '-i', tmpPath, '-o', glbOut, '--layout', String(form.get('layout') || 'grid'),
      '--scale', String(form.get('scale') || '0.12')], ROOT);
  }

  // Optional sidecar metadata
  const sidecar = await writeSidecar(glbOut, {
    name: path.basename(glbOut),
    description,
    tags,
    source: isJSON ? 'concept_json' : 'glb_upload',
    uploadedAt: new Date().toISOString()
  });

  // Rebuild templates index (PACK-M3 tool)
  try {
    runNode('node', ['../tools/release/build-templates-index.mjs'], path.join(ROOT, 'frontend'));
  } catch (e) {
    // fallback PS on Windows-only if Node script missing
    try { runNode('powershell', ['-ExecutionPolicy', 'Bypass', '-File', '../tools/release/Build-Templates-Index.ps1'], path.join(ROOT, 'frontend')); } catch {}
  }

  return new Response(JSON.stringify({ ok: true, glb: glbOut, meta: sidecar }), {
    headers: { 'content-type': 'application/json' }
  });
};