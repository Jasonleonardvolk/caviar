/* frontend/scripts/validate_shaders.mjs */
import { promises as fs } from 'node:fs';
import path from 'node:path';
import { globby } from 'globby';

const ROOT = process.cwd();
const SHADERS_DIR = path.join(ROOT, 'frontend', 'lib', 'webgpu', 'shaders');

const LIMITS = {
  maxWorkgroupSizeX: 256,
  maxWorkgroupSizeY: 256,
  maxWorkgroupSizeZ: 64,
  maxWorkgroupInvocations: 256,
  // heuristic storage size per binding in bytes (rough sanity)
  maxStaticStorageBytes: 16 * 1024 * 1024,
};

function checkHeuristics(source, file) {
  const errs = [];

  // Simple regex checks (best-effort)
  const wgSize = /@workgroup_size\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)/.exec(source);
  if (wgSize) {
    const [ , x, y, z ] = wgSize.map(Number);
    if (x > LIMITS.maxWorkgroupSizeX) errs.push(`workgroup_size.x ${x} > ${LIMITS.maxWorkgroupSizeX}`);
    if (y > LIMITS.maxWorkgroupSizeY) errs.push(`workgroup_size.y ${y} > ${LIMITS.maxWorkgroupSizeY}`);
    if (z > LIMITS.maxWorkgroupSizeZ) errs.push(`workgroup_size.z ${z} > ${LIMITS.maxWorkgroupSizeZ}`);
    if (x*y*z > LIMITS.maxWorkgroupInvocations) errs.push(`invocations ${x*y*z} > ${LIMITS.maxWorkgroupInvocations}`);
  }

  // Storage array literal allocations hint
  const bigArr = /var<\s*(?:workgroup|storage)[^>]*>\s+[a-zA-Z_]\w*\s*:\s*array<[^>]*,\s*(\d+)\s*>/g;
  let m;
  while ((m = bigArr.exec(source))) {
    const n = Number(m[1]);
    if (n * 16 > LIMITS.maxStaticStorageBytes) {
      errs.push(`large static array (~${n*16} bytes) may exceed storage heuristics`);
    }
  }

  if (errs.length) {
    console.error(`[${file}]`);
    errs.forEach(e => console.error('  - ' + e));
  }
  return errs;
}

(async function main() {
  const files = await globby([`${SHADERS_DIR}/**/*.wgsl`]);
  let fail = 0;
  for (const f of files) {
    const src = await fs.readFile(f, 'utf8');
    const errs = checkHeuristics(src, f);
    if (errs.length) fail++;
  }
  if (fail) {
    console.error(`Shader validation failed in ${fail} file(s).`);
    process.exit(1);
  } else {
    console.log(`All shaders passed heuristic validation (${files.length} files).`);
  }
})();
