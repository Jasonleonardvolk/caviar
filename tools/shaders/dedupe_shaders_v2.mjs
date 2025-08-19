#!/usr/bin/env node
/**
 * dedupe_shaders_v2.mjs
 * Always KEEP canonical (frontend/lib/webgpu/shaders) and remove/move legacy.
 * - Identical copies: delete all non-canonical duplicates.
 * - Conflicts (different hashes): keep canonical; move legacy to backup; do NOT touch canonical.
 */
import fs from 'fs';
import path from 'path';
import crypto from 'crypto';

const args = new Map(process.argv.slice(2).map(a => a.includes('=') ? a.split('=') : [a, true]));
const REPO = path.resolve(String(args.get('--repo') || process.cwd()));
const APPLY = !!args.get('--apply');
const BACKUP = !!args.get('--backup');
const REPORT = args.get('--report') ? path.resolve(String(args.get('--report'))) : null;

// Load config if present
const CFG_PATH = path.resolve('tools/shaders/shader_roots.json');
const CFG = fs.existsSync(CFG_PATH) ? JSON.parse(fs.readFileSync(CFG_PATH, 'utf-8')) : {
  canonical: "frontend/lib/webgpu/shaders",
  legacy_preferred_order: ["frontend/shaders","frontend/hybrid","frontend/public/hybrid/wgsl"],
  exclude_patterns: ["node_modules/","dist/","build/",".git/","frontend/shaders.bak/","conversations/"]
};

const CANONICAL_ROOT = path.resolve(REPO, CFG.canonical.replace(/\//g, path.sep));
const LEGACY_ORDER = CFG.legacy_preferred_order.map(p => path.resolve(REPO, p.replace(/\//g, path.sep)));
const EXCLUDES = CFG.exclude_patterns || [];

const BACKUP_DIR = path.join(REPO, 'frontend', 'shaders.bak', 'duplicates', new Date().toISOString().replace(/[:.]/g,'-'));

function isExcluded(rel) {
  return EXCLUDES.some(ex => rel.includes(ex));
}
function walk(dir) {
  const out = [];
  const st = [dir];
  while (st.length) {
    const d = st.pop();
    for (const e of fs.readdirSync(d, {withFileTypes:true})) {
      const p = path.join(d, e.name);
      const rel = path.relative(REPO, p);
      if (isExcluded(rel)) continue;
      if (e.isDirectory()) st.push(p);
      else if (e.isFile() && e.name.endsWith('.wgsl')) out.push(p);
    }
  }
  return out;
}
function shaFile(p) {
  return crypto.createHash('sha1').update(fs.readFileSync(p)).digest('hex');
}
function rankPath(p) {
  const rp = path.resolve(p);
  if (rp.startsWith(CANONICAL_ROOT + path.sep) || rp === CANONICAL_ROOT) return 0; // best
  const idx = LEGACY_ORDER.findIndex(root => rp.startsWith(root + path.sep) || rp === root);
  return idx >= 0 ? (idx + 1) : 999;
}

function chooseCanonical(entries) {
  // entries: [{path, hash}]
  // Group by hash to detect identical sets
  const byHash = new Map();
  for (const e of entries) {
    const list = byHash.get(e.hash) || [];
    list.push(e);
    byHash.set(e.hash, list);
  }
  // If there is a canonical-path entry, that is the pick regardless.
  const canonCandidates = entries.filter(e => rankPath(e.path) === 0);
  const pick = canonCandidates.length ? canonCandidates[0] :
               entries.slice().sort((a,b) => rankPath(a.path) - rankPath(b.path) || a.path.length - b.path.length)[0];
  const conflict = byHash.size > 1;
  return {pick, conflict, byHash};
}

const files = walk(REPO);
const groups = new Map(); // basename -> [{path,hash}]
for (const f of files) {
  const name = path.basename(f);
  const list = groups.get(name) || [];
  list.push({path:f, hash:shaFile(f)});
  groups.set(name, list);
}

const manifest = {repo: REPO, canonical_root: path.relative(REPO, CANONICAL_ROOT), items: [], conflicts: [], removed: [], backups: []};

if (APPLY && BACKUP) fs.mkdirSync(BACKUP_DIR, {recursive:true});

for (const [name, entries] of groups.entries()) {
  const {pick, conflict, byHash} = chooseCanonical(entries);
  manifest.items.push({
    name,
    canonical: path.relative(REPO, pick.path),
    count: entries.length,
    hashes: Array.from(new Set(entries.map(e=>e.hash)))
  });
  if (conflict) {
    manifest.conflicts.push({
      name, canonical: path.relative(REPO, pick.path),
      variants: entries.map(e => ({path: path.relative(REPO, e.path), hash: e.hash}))
    });
  }
  if (APPLY) {
    for (const e of entries) {
      if (e.path === pick.path) continue;
      // identical hash -> remove non-canonical copy
      if (e.hash === pick.hash) {
        if (BACKUP) {
          const dest = path.join(BACKUP_DIR, path.relative(REPO, e.path).replace(/[\\/:]/g,'__'));
          fs.mkdirSync(path.dirname(dest), {recursive:true});
          fs.copyFileSync(e.path, dest);
          manifest.backups.push({from: path.relative(REPO, e.path), to: path.relative(REPO, dest)});
        }
        fs.rmSync(e.path);
        manifest.removed.push(path.relative(REPO, e.path));
      } else {
        // different content -> ALWAYS keep canonical pick, backup legacy variant and remove it
        if (BACKUP) {
          const dest = path.join(BACKUP_DIR, path.relative(REPO, e.path).replace(/[\\/:]/g,'__'));
          fs.mkdirSync(path.dirname(dest), {recursive:true});
          fs.copyFileSync(e.path, dest);
          manifest.backups.push({from: path.relative(REPO, e.path), to: path.relative(REPO, dest)});
        }
        fs.rmSync(e.path);
        manifest.removed.push(path.relative(REPO, e.path));
      }
    }
  }
}

if (REPORT) {
  fs.mkdirSync(path.dirname(REPORT), {recursive:true});
  fs.writeFileSync(REPORT, JSON.stringify(manifest, null, 2));
}

console.log(`Scanned ${files.length} WGSL files.`);
console.log(`Groups: ${groups.size}; Conflicts: ${manifest.conflicts.length}`);
console.log(`Canonical root: ${manifest.canonical_root}`);
if (APPLY) {
  console.log(`Removed: ${manifest.removed.length}; Backups: ${manifest.backups.length}`);
  if (BACKUP) console.log(`Backups at: ${path.relative(REPO, BACKUP_DIR)}`);
}
