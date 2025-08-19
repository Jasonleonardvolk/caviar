// @ts-nocheck
import type { PageServerLoad } from './$types';
import fs from 'node:fs';
import path from 'node:path';

const ROOT = path.resolve(process.cwd(), '..');                       // → D:\Dev\kha (dev)
const TEMPLATES_DIR = path.join(ROOT, 'exports', 'templates');        // → D:\Dev\kha\exports\templates

function listTemplates() {
  const items: Array<{
    file: string;
    name: string;
    size: number;
    mtime: string;
    meta: Record<string, unknown>;
  }> = [];

  if (!fs.existsSync(TEMPLATES_DIR)) return items;

  for (const entry of fs.readdirSync(TEMPLATES_DIR, { withFileTypes: true })) {
    if (!entry.isFile()) continue;
    if (!entry.name.toLowerCase().endsWith('.glb')) continue;

    const abs = path.join(TEMPLATES_DIR, entry.name);
    const stat = fs.statSync(abs);

    const metaPath = abs.replace(/\.glb$/i, '.template.json');
    let meta: Record<string, unknown> = {};
    if (fs.existsSync(metaPath)) {
      try { meta = JSON.parse(fs.readFileSync(metaPath, 'utf-8')); } catch {}
    }

    items.push({
      file: abs,
      name: entry.name,
      size: stat.size,
      mtime: stat.mtime.toISOString(),
      meta
    });
  }
  return items.sort((a, b) => a.name.localeCompare(b.name));
}

export const load = async () => {
  return { items: listTemplates(), root: ROOT, dir: TEMPLATES_DIR };
};;null as any as PageServerLoad;