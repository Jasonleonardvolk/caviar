#!/usr/bin/env node
/**
 * tools/shaders/tint_emit.mjs
 * Emit backend shader artifacts for parity-only checks.
 * For each WGSL in --src, run Tint to {msl,hlsl,spirv} and write to build/shaders
 * with a content hash: name.<sha8>.{msl,hlsl,spv}
 *
 * Usage:
 *   node tools/shaders/tint_emit.mjs --src=frontend/lib/webgpu/shaders --targets=msl,hlsl,spirv
 */
import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import {exec} from 'child_process';
import {promisify} from 'util';

const run = promisify(exec);
const args = new Map(process.argv.slice(2).map(a => a.includes('=') ? a.split('=') : [a, true]));
const SRC = path.resolve(String(args.get('--src') || 'frontend/lib/webgpu/shaders'));
const TARGETS = String(args.get('--targets') || 'msl,hlsl').split(',').filter(Boolean);
const OUTDIR = path.resolve('build/shaders');

fs.mkdirSync(OUTDIR, {recursive:true});

async function hasBin(bin) {
  try {
    const {stdout} = await run(`${bin} --version`);
    return !!stdout;
  } catch { return false; }
}

async function tint(file, fmt, outFile) {
  const cmd = `tint --format=${fmt} "${file}" -o "${outFile}"`;
  const {stdout, stderr} = await run(cmd).catch(e => e);
  if (stderr && /error/i.test(String(stderr))) throw new Error(String(stderr));
  return {stdout, stderr};
}

function hashFile(p) {
  const buf = fs.readFileSync(p);
  return crypto.createHash('sha1').update(buf).digest('hex');
}

function listWgsl(dir) {
  return fs.readdirSync(dir).filter(f => f.endsWith('.wgsl')).map(f => path.join(dir, f));
}

(async () => {
  if (!(await hasBin('tint'))) {
    console.error('ERROR: tint not found on PATH.');
    process.exit(1);
  }
  const files = listWgsl(SRC);
  let ok = 0, fail = 0;
  for (const f of files) {
    const base = path.basename(f, '.wgsl');
    const sha8 = hashFile(f).slice(0,8);
    for (const t of TARGETS) {
      const ext = (t === 'spirv' ? 'spv' : t);
      const out = path.join(OUTDIR, `${base}.${sha8}.${ext}`);
      try {
        await tint(f, t, out);
        console.log(`✓ ${path.relative('.', f)} -> ${path.relative('.', out)}`);
        ok++;
      } catch (e) {
        console.error(`✗ ${path.relative('.', f)} (${t})\n${String(e.message || e)}`);
        fail++;
      }
    }
  }
  if (fail) process.exit(1);
  process.exit(0);
})().catch(e => { console.error(e); process.exit(1); });
