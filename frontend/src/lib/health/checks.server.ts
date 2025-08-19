import fs from 'node:fs';
import fsp from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';

const ROOT = path.resolve(process.cwd(), '..');                // → D:\Dev\kha
const FRONT = path.join(ROOT, 'frontend');                     // → D:\Dev\kha\frontend
const EXPORTS = path.join(ROOT, 'exports');
const TEMPLATES_DIR = path.join(EXPORTS, 'templates');
const TEX_DIR = path.join(EXPORTS, 'textures_ktx2');
const SNAP_GUIDES = path.join(ROOT, 'integrations', 'snap', 'guides');
const TIKTOK_GUIDES = path.join(ROOT, 'integrations', 'tiktok', 'guides');
const PLANS_SRC = path.join(ROOT, 'config', 'plans.json');
const PLANS_DST = path.join(FRONT, 'static', 'config', 'plans.json');
const ENV_FILE = path.join(FRONT, '.env');

const REQUIRED_FILES = [
  // Hologram route & UI (PACK-HOLO-ROUTE)
  'frontend/src/routes/hologram/+page.svelte',
  'frontend/src/lib/hologram/engineShim.ts',
  'frontend/src/lib/device/capabilities.ts',
  'frontend/src/lib/stores/userPlan.ts',
  'frontend/src/lib/utils/exportVideo.ts',
  'frontend/src/lib/components/HologramRecorder.svelte',
  // Pricing & billing (PACK-M1/M2)
  'frontend/src/routes/pricing/+page.svelte',
  'frontend/src/lib/components/PricingTable.svelte',
  'frontend/src/routes/api/billing/checkout/+server.ts',
  'frontend/src/routes/api/billing/portal/+server.ts',
  // Templates catalog & export API (PACK-M3)
  'frontend/src/routes/templates/+page.svelte',
  'frontend/src/routes/templates/+page.server.ts',
  'frontend/src/routes/api/templates/export/+server.ts',
  // Exporters (PACK-M2)
  'tools/exporters/glb-from-conceptmesh.ts',
  'tools/exporters/encode-ktx2.ps1',
  // Upload + file streaming + publish checklist (PACK-M4)
  'frontend/src/routes/templates/upload/+page.svelte',
  'frontend/src/routes/api/templates/upload/+server.ts',
  'frontend/src/routes/api/templates/file/[name]/+server.ts',
  'frontend/src/routes/publish/+page.svelte',
  'frontend/src/routes/publish/+page.server.ts',
  // Plans config (synced)
  'config/plans.json',
  'frontend/static/config/plans.json'
].map(p => path.join(ROOT, p));

function exists(p: string) { try { fs.accessSync(p, fs.constants.R_OK); return true; } catch { return false; } }
function hashFile(p: string) { return crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex'); }

async function list(dir: string, filterExt?: string) {
  try {
    const out: { name: string; path: string; size: number; mtime: string }[] = [];
    for (const e of await fsp.readdir(dir, { withFileTypes: true })) {
      if (!e.isFile()) continue;
      if (filterExt && !e.name.toLowerCase().endsWith(filterExt)) continue;
      const abs = path.join(dir, e.name);
      const st = await fsp.stat(abs);
      out.push({ name: e.name, path: abs, size: st.size, mtime: st.mtime.toISOString() });
    }
    return out.sort((a,b)=>a.name.localeCompare(b.name));
  } catch { return []; }
}

function stripeKeyPresent(): boolean {
  // Prefer env var; fallback parse .env for demo/dev
  if (process.env.STRIPE_SECRET_KEY && String(process.env.STRIPE_SECRET_KEY).trim().length > 0) return true;
  try {
    const text = fs.readFileSync(ENV_FILE, 'utf8');
    return /STRIPE_SECRET_KEY\s*=\s*\S+/.test(text);
  } catch { return false; }
}

function readVersion(pkgPath: string): string | undefined {
  try {
    const raw = fs.readFileSync(pkgPath, 'utf8');
    const pkg = JSON.parse(raw);
    return pkg?.version;
  } catch { return undefined; }
}

export async function getHealth() {
  const timestamp = new Date().toISOString();

  // Files & sync
  const missing = REQUIRED_FILES.filter(p => !exists(p));
  const plansSynced = exists(PLANS_SRC) && exists(PLANS_DST)
    ? hashFile(PLANS_SRC) === hashFile(PLANS_DST)
    : false;

  // Artifacts
  const glbs = await list(TEMPLATES_DIR, '.glb');
  const ktx2 = await list(TEX_DIR, '.ktx2');

  // Guides
  const snapGuides = await list(SNAP_GUIDES, '.md');
  const tiktokGuides = await list(TIKTOK_GUIDES, '.md');

  // Versions
  const frontendPkgVer = readVersion(path.join(FRONT, 'package.json'));
  const projectPkgVer  = readVersion(path.join(ROOT, 'package.json'));

  // Capabilities (route presence by file existence)
  const hologramReady = REQUIRED_FILES.some(p => p.endsWith('routes\\hologram\\+page.svelte') || p.endsWith('routes/hologram/+page.svelte'))
    && REQUIRED_FILES.some(p => p.endsWith('HologramRecorder.svelte'));
  const pricingReady = REQUIRED_FILES.some(p => p.includes('routes\\pricing\\+page.svelte') || p.includes('routes/pricing/+page.svelte'));
  const billingReady = REQUIRED_FILES.some(p => p.includes('billing\\checkout\\+server.ts')) &&
                       REQUIRED_FILES.some(p => p.includes('billing\\portal\\+server.ts'));

  const exportersReady = REQUIRED_FILES.some(p => p.includes('glb-from-conceptmesh.ts')) &&
                         REQUIRED_FILES.some(p => p.includes('encode-ktx2.ps1'));

  const catalogReady = REQUIRED_FILES.some(p => p.includes('routes\\templates\\+page.svelte')) &&
                       REQUIRED_FILES.some(p => p.includes('api\\templates\\export\\+server.ts'));

  const uploadReady = REQUIRED_FILES.some(p => p.includes('templates\\upload\\+page.svelte')) &&
                      REQUIRED_FILES.some(p => p.includes('api\\templates\\upload\\+server.ts')) &&
                      REQUIRED_FILES.some(p => p.includes('api\\templates\\file\\[name]\\+server.ts'));

  const publishReadySnap   = glbs.length > 0 && ktx2.length > 0 && snapGuides.length > 0;
  const publishReadyTikTok = glbs.length > 0 && ktx2.length > 0 && tiktokGuides.length > 0;

  // Monetization gate
  const monetizationReady = pricingReady && billingReady && stripeKeyPresent() && plansSynced;

  // Overall demo-ready: hologram UI + recorder + templates catalog + monetization configured
  const ok = missing.length === 0 && hologramReady && catalogReady && exportersReady && monetizationReady;

  // Suggestions
  const advice: string[] = [];
  if (!hologramReady) advice.push('Add /hologram route and HologramRecorder.svelte wiring.');
  if (!catalogReady) advice.push('Ensure /templates route and /api/templates/export are present.');
  if (!exportersReady) advice.push('Install exporters: tools/exporters/glb-from-conceptmesh.ts and encode-ktx2.ps1.');
  if (!stripeKeyPresent()) advice.push('Set STRIPE_SECRET_KEY in frontend\\.env or environment.');
  if (!plansSynced) advice.push('Sync config\\plans.json → frontend\\static\\config\\plans.json (Sync-Plans.ps1).');
  if (!uploadReady) advice.push('Add /templates/upload UI and /api/templates/upload + /api/templates/file/* endpoints.');
  if (!publishReadySnap) advice.push('Create GLB + KTX2 + Snap guides under integrations\\snap\\guides.');
  if (!publishReadyTikTok) advice.push('Create GLB + KTX2 + TikTok guides under integrations\\tiktok\\guides.');

  return {
    ok,
    timestamp,
    env: {
      node: process.version,
      platform: process.platform,
      arch: process.arch,
      cwd: process.cwd(),
      frontendPackageVersion: frontendPkgVer,
      projectPackageVersion: projectPkgVer
    },
    files: {
      requiredTotal: REQUIRED_FILES.length,
      presentCount: REQUIRED_FILES.length - missing.length,
      missing
    },
    plans: { src: PLANS_SRC, dst: PLANS_DST, synced: plansSynced },
    monetization: {
      stripeKeyPresent: stripeKeyPresent(),
      pricingRouteExists: pricingReady,
      billingEndpointsExist: billingReady
    },
    hologram: { routeExists: hologramReady, recorderExists: REQUIRED_FILES.some(p => p.includes('HologramRecorder.svelte')) },
    exporters: { glbExporterExists: exists(path.join(ROOT,'tools','exporters','glb-from-conceptmesh.ts')), ktx2ScriptExists: exists(path.join(ROOT,'tools','exporters','encode-ktx2.ps1')) },
    templates: {
      routes: { catalog: catalogReady, upload: uploadReady },
      counts: { glb: glbs.length, ktx2: ktx2.length }
    },
    guides: { snap: snapGuides.length, tiktok: tiktokGuides.length },
    readiness: {
      demo: hologramReady && catalogReady && stripeKeyPresent() && plansSynced,
      publishSnap: publishReadySnap,
      publishTikTok: publishReadyTikTok
    },
    artifacts: {
      templatesDir: TEMPLATES_DIR,
      texturesDir: TEX_DIR,
      glb: glbs,
      ktx2: ktx2
    },
    advice
  };
}