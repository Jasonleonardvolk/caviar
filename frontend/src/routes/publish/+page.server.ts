import type { PageServerLoad } from './$types';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import path from 'node:path';

const ROOT = path.resolve(process.cwd(), '..');
const EXPORTS_DIR = path.join(ROOT, 'exports');
const TEMPLATES_DIR = path.join(EXPORTS_DIR, 'templates');
const TEX_DIR = path.join(EXPORTS_DIR, 'textures_ktx2');
const SNAP_GUIDES = path.join(ROOT, 'integrations', 'snap', 'guides');
const TIKTOK_GUIDES = path.join(ROOT, 'integrations', 'tiktok', 'guides');
const ENV_FILE = path.join(ROOT, 'frontend', '.env');
const PLANS_SRC = path.join(ROOT, 'config', 'plans.json');
const PLANS_DST = path.join(ROOT, 'frontend', 'static', 'config', 'plans.json');

async function listFiles(dir: string, ext?: string) {
  try {
    const out: any[] = [];
    const entries = await fsp.readdir(dir, { withFileTypes: true });
    for (const e of entries) {
      if (!e.isFile()) continue;
      if (ext && !e.name.toLowerCase().endsWith(ext)) continue;
      const abs = path.join(dir, e.name);
      const stat = await fsp.stat(abs);
      out.push({ name: e.name, path: abs, size: stat.size, mtime: stat.mtime.toISOString() });
    }
    return out.sort((a,b)=>a.name.localeCompare(b.name));
  } catch { return []; }
}

async function readGuides(dir: string) {
  const items = await listFiles(dir, '.md');
  const guides: { name: string; path: string; content: string }[] = [];
  for (const it of items) {
    try {
      guides.push({ name: it.name, path: it.path, content: await fsp.readFile(it.path, 'utf8') });
    } catch {}
  }
  return guides;
}

function hasStripeKey() {
  try { const t = fs.readFileSync(ENV_FILE, 'utf8'); return /STRIPE_SECRET_KEY\s*=\s*\S+/.test(t); } catch { return false; }
}

function plansInSync() {
  try {
    const A = fs.readFileSync(PLANS_SRC); const B = fs.readFileSync(PLANS_DST);
    return require('node:crypto').createHash('sha256').update(A).digest('hex') ===
           require('node:crypto').createHash('sha256').update(B).digest('hex');
  } catch { return false; }
}

export const load: PageServerLoad = async () => {
  const glbs = await listFiles(TEMPLATES_DIR, '.glb');
  const textures = await listFiles(TEX_DIR, '.ktx2');
  const snapGuides = await readGuides(SNAP_GUIDES);
  const ttGuides = await readGuides(TIKTOK_GUIDES);
  const stripeOk = hasStripeKey();
  const plansOk = plansInSync();

  // Heuristic "ready" flag
  const readyForSnap = glbs.length > 0 && textures.length > 0 && snapGuides.length > 0;
  const readyForTikTok = glbs.length > 0 && textures.length > 0 && ttGuides.length > 0;

  return {
    env: { stripeOk, plansOk, envFile: ENV_FILE, plansSrc: PLANS_SRC, plansDst: PLANS_DST },
    artifacts: { glbs, textures, exportsDir: EXPORTS_DIR },
    guides: { snap: snapGuides, tiktok: ttGuides },
    readiness: { snap: readyForSnap, tiktok: readyForTikTok }
  };
};