#!/usr/bin/env node
/**
 * Merge Enhanced into canonical lightFieldComposer and update imports project-wide.
 *
 * Usage:
 *  node tools/shaders/merge_lightFieldComposer.mjs --repo=. \
 *    --srcA=frontend/hybrid/wgsl/lightFieldComposer.wgsl \
 *    --srcB=frontend/hybrid/wgsl/lightFieldComposerEnhanced.wgsl \
 *    --canonical=frontend/lib/webgpu/shaders/lightFieldComposer.wgsl
 */
import fs from 'fs';
import path from 'path';
import crypto from 'crypto';

const args = new Map(process.argv.slice(2).map(a => a.includes('=') ? a.split('=') : [a, true]));
const REPO = path.resolve(String(args.get('--repo') || '.'));
const A = path.resolve(String(args.get('--srcA') || 'frontend/hybrid/wgsl/lightFieldComposer.wgsl'));
const B = path.resolve(String(args.get('--srcB') || 'frontend/hybrid/wgsl/lightFieldComposerEnhanced.wgsl'));
const CANON = path.resolve(String(args.get('--canonical') || 'frontend/lib/webgpu/shaders/lightFieldComposer.wgsl'));

function sha(s) { return crypto.createHash('sha1').update(s).digest('hex'); }

function merge(a, b) {
  // Prefer Enhanced (B) body, preserve header comments from A.
  const headerA = (a.match(/^(\/\/.*\n)+/m) || [''])[0];
  const bodyB = b.replace(/^(\/\/.*\n)+/m, '');
  const banner = `// Canonical lightFieldComposer.wgsl (merged)\n// A: ${sha(a).slice(0,8)}  B: ${sha(b).slice(0,8)}\n`;
  return banner + headerA + bodyB;
}

fs.mkdirSync(path.dirname(CANON), {recursive:true});
const a = fs.readFileSync(A, 'utf-8');
const b = fs.readFileSync(B, 'utf-8');
const out = merge(a,b);
fs.writeFileSync(CANON, out, 'utf-8');
console.log(`Wrote canonical: ${path.relative(REPO, CANON)}`);

// Remove the Enhanced file to avoid ambiguity (backup first)
const backupDir = path.join(REPO, 'frontend', 'shaders.bak', 'merged');
fs.mkdirSync(backupDir, {recursive:true});
const Bbackup = path.join(backupDir, 'lightFieldComposerEnhanced.wgsl');
fs.copyFileSync(B, Bbackup);
fs.rmSync(B);
console.log(`Backed up & removed Enhanced: ${path.relative(REPO, B)} -> ${path.relative(REPO, Bbackup)}`);

// Update imports everywhere
const exts = ['.ts','.tsx','.js','.jsx','.svelte','.wgsl'];
function walk(dir) {
  const out = [];
  const entries = fs.readdirSync(dir, {withFileTypes:true});
  for (const e of entries) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) {
      if (/node_modules|dist|build|\.git|shaders\.bak|conversations/i.test(p)) continue;
      out.push(...walk(p));
    } else {
      if (exts.includes(path.extname(e.name))) out.push(p);
    }
  }
  return out;
}

const files = walk(REPO);
const patterns = [
  /frontend\/hybrid\/wgsl\/lightFieldComposer(?:Enhanced)?\.wgsl/g,
  /frontend\\hybrid\\wgsl\\lightFieldComposer(?:Enhanced)?\.wgsl/g,
  /frontend\/hybrid\/lightFieldComposer(?:Enhanced)?\.wgsl/g,
  /frontend\\hybrid\\lightFieldComposer(?:Enhanced)?\.wgsl/g
];
let changed = 0;
for (const f of files) {
  let s = fs.readFileSync(f, 'utf-8');
  let s2 = s;
  for (const pat of patterns) s2 = s2.replace(pat, 'frontend/lib/webgpu/shaders/lightFieldComposer.wgsl');
  if (s2 !== s) {
    fs.writeFileSync(f, s2, 'utf-8');
    changed++;
    console.log(`Updated import: ${path.relative(REPO, f)}`);
  }
}
console.log(`Updated ${changed} files.`);
