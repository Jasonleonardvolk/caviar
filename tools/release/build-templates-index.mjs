import { readdir, readFile, stat, mkdir, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';

const ROOT = resolve(process.cwd());                 // run from repo root or /frontend
const FRONTEND = ROOT.endsWith('frontend') ? ROOT : join(ROOT, 'frontend');
const PROJECT = ROOT.endsWith('frontend') ? resolve(FRONTEND, '..') : ROOT;

const TEMPLATES_DIR = join(PROJECT, 'exports', 'templates');
const OUT = join(FRONTEND, 'static', 'templates');
await mkdir(OUT, { recursive: true });

async function listGlb(dir) {
  try {
    const entries = await readdir(dir, { withFileTypes: true });
    const out = [];
    for (const e of entries) {
      if (!e.isFile() || !e.name.toLowerCase().endsWith('.glb')) continue;
      const abs = join(dir, e.name);
      const st = await stat(abs);
      const metaPath = abs.replace(/\.glb$/i, '.template.json');
      let meta = {};
      try { meta = JSON.parse(await readFile(metaPath, 'utf-8')); } catch {}
      out.push({ name: e.name, size: st.size, mtime: st.mtime.toISOString(), meta });
    }
    return out.sort((a,b) => a.name.localeCompare(b.name));
  } catch {
    return [];
  }
}

const items = await listGlb(TEMPLATES_DIR);
await writeFile(join(OUT, 'index.json'), JSON.stringify({ items }, null, 2));
console.log(`✅ templates index → ${join(OUT, 'index.json')}`);