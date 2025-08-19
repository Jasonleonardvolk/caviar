import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { createHash } from 'node:crypto';

const ROOT = 'D:\\Dev\\kha';
const REPORT_DIR = join(ROOT, 'verification_reports');
const STAMP = new Date().toISOString().replace(/[:.]/g, '-');
const REPORT = join(REPORT_DIR, `mobile_claims_check_${STAMP}.json`);

// Inputs
const CONFIG = join(ROOT, 'config', 'mobile_support.json');
const README = join(ROOT, 'README.md');
const MATRIX = join(ROOT, 'docs', 'MOBILE_SUPPORT_MATRIX.md');
const CAPS   = join(ROOT, 'frontend', 'src', 'lib', 'device', 'capabilities.ts');
const HOLO   = join(ROOT, 'frontend', 'src', 'routes', 'hologram', '+page.svelte');

// Helpers
const ok = (b) => Boolean(b);
const sha = (buf) => createHash('sha256').update(buf).digest('hex');

async function readTxt(p, allowMissing = false) {
  try { return await readFile(p, 'utf8'); } catch (e) { if (allowMissing) return ''; throw e; }
}

function has(regex, text) {
  return new RegExp(regex, 'i').test(text || '');
}

function fail(msg, meta) { return { ok: false, msg, ...(meta||{}) }; }

(async () => {
  await mkdir(REPORT_DIR, { recursive: true });

  const cfg = JSON.parse(await readTxt(CONFIG));
  const readme = existsSync(README) ? await readTxt(README) : '';
  const matrix = existsSync(MATRIX) ? await readTxt(MATRIX) : '';
  const caps   = existsSync(CAPS) ? await readTxt(CAPS) : '';
  const holo   = existsSync(HOLO) ? await readTxt(HOLO) : '';

  const issues = [];

  // 1) README must assert major iOS and preferred beta string
  if (!has(`iOS\\s*${cfg.iosMajor}`, readme)) issues.push(fail('README missing required iOS major', { need: cfg.iosMajor }));
  if (!has(cfg.preferredBeta.replace('+','\\+'), readme)) issues.push(fail('README missing preferred beta string', { need: cfg.preferredBeta }));

  // 2) README or MATRIX must mention minimum iPhone model
  const minPhoneOK = has(cfg.minIphoneModel, readme) || has(cfg.minIphoneModel, matrix);
  if (!minPhoneOK) issues.push(fail('Minimum iPhone model not documented', { need: cfg.minIphoneModel }));

  // 3) Recommended iPads should appear in README or MATRIX
  for (const rec of cfg.recommendedIpads) {
    const found = has(rec, readme) || has(rec, matrix);
    if (!found) issues.push(fail('Recommended iPad missing', { need: rec }));
  }

  // 4) Capabilities code should actually prefer WebGPU when available
  if (cfg.preferWebGPU) {
    const preferOK =
      has('navigator\\.gpu', caps) &&
      has('prefersWebGPUHint', caps) &&
      has('return\\s+caps\\.webgpu\\s*===\\s*true', caps);
    if (!preferOK) issues.push(fail('capabilities.ts does not prefer WebGPU as configured', { file: CAPS }));
  }

  // 5) Hologram route must exist and reference #hologram-canvas
  const holoOK = existsSync(HOLO) && has('#hologram-canvas', holo);
  if (!holoOK) issues.push(fail('Hologram route or canvas id missing', { file: HOLO, need: '#hologram-canvas' }));

  // 6) Optional: confirm routes are referenced somewhere in README (soft check)
  for (const r of (cfg.requiredRoutes || [])) {
    if (!has(r.replace(/[\/]/g,'\\/'), readme)) {
      issues.push({ ok:false, msg:'Route reference not found in README (soft)', route:r });
    }
  }

  // 7) Record doc/code hashes for traceability
  const artifacts = {};
  if (existsSync(README)) artifacts.readmeHash = sha(await readFile(README));
  if (existsSync(MATRIX)) artifacts.matrixHash = sha(await readFile(MATRIX));
  if (existsSync(CAPS))   artifacts.capabilitiesHash = sha(await readFile(CAPS));

  const result = {
    ok: issues.length === 0,
    timestamp: new Date().toISOString(),
    config: cfg,
    files: { CONFIG, README, MATRIX, CAPS, HOLO },
    issues,
    artifacts
  };

  await writeFile(REPORT, JSON.stringify(result, null, 2));
  console.log(result.ok ? '[OK] Docs/claims consistent' : '[X] Docs/claims drift detected');
  if (!result.ok) {
    console.error('See report:', REPORT);
    process.exit(2);
  }
})();