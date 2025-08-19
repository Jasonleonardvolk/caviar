/* frontend/scripts/build_floater.mjs */
import esbuild from 'esbuild';
import path from 'node:path';
import { promises as fs } from 'node:fs';

const ROOT = process.cwd();
const SRC_DIR = path.join(ROOT, 'frontend', 'floater', 'src');
const DIST = path.join(ROOT, 'frontend', 'floater', 'dist');

await fs.mkdir(DIST, { recursive: true });

// ESM host
await esbuild.build({
  entryPoints: [path.join(SRC_DIR, 'host.ts')],
  outfile: path.join(DIST, 'gaea_floater.esm.js'),
  bundle: true,
  format: 'esm',
  target: ['es2022'],
  sourcemap: true,
  splitting: false,
  loader: { '.wgsl': 'text' },
  external: [
    // externalize model blobs; the app fetches at runtime
    '/public/models/waveop_fno_v1.onnx'
  ],
});

// UMD host (optional)
await esbuild.build({
  entryPoints: [path.join(SRC_DIR, 'host.ts')],
  outfile: path.join(DIST, 'gaea_floater.umd.js'),
  bundle: true,
  format: 'iife',
  globalName: 'GaeaFloater',
  target: ['es2019'],
  sourcemap: true,
  loader: { '.wgsl': 'text' },
});

// Worker
await esbuild.build({
  entryPoints: [path.join(SRC_DIR, 'renderer.worker.ts')],
  outfile: path.join(DIST, 'gaea_floater.worker.js'),
  bundle: true,
  format: 'esm',
  target: ['es2022'],
  sourcemap: true,
  loader: { '.wgsl': 'text' },
});

// Types (d.ts) — minimal hand-rolled based on host API
const DTS = `
// frontend/floater/dist/gaea_floater.d.ts
export type Policy = {
  donorMode: 'lan'|'off';
  keyframeInterval: number|'auto';
  compressedReturn: boolean;
  roiThreshold: number|'auto';
  maxMsPerFrame: number;
  maxMemMB: number;
};

export type FloaterEvent =
  | { type: 'metrics:frame'; ms: number; fps: number; backend: string; memMB: number }
  | { type: 'net:stats'; kbps: number; rttMs: number }
  | { type: 'donor:drop'; ts: number }
  | { type: 'onnx:ready'; backend: string }
  | { type: 'fft:backend'; kind: 'subgroup'|'baseline' }
  | { type: 'error'; message: string };

export type FloaterConfig = {
  mount: HTMLCanvasElement | HTMLVideoElement;
  dims: [number, number];
  kernel: 'onnx_waveop'|'splitstep_fft'|'biharmonic';
  params?: Record<string, any>;
  policy?: Partial<Policy>;
  onEvent?: (e: FloaterEvent) => void;
};

export type Floater = {
  start(): void;
  stop(): void;
  setDims(dims: [number, number]): void;
  setParams(params: Record<string, any>): void;
  setPolicy(policy: Partial<Policy>): void;
  destroy(): void;
};

export function createFloater(cfg: FloaterConfig): Promise<Floater>;
`;
await fs.writeFile(path.join(DIST, 'gaea_floater.d.ts'), DTS, 'utf8');

// CSS (HUD optional)
const CSS = `
/* frontend/floater/dist/gaea_floater.css */
.gaea-hud { position: absolute; top: 8px; right: 8px; padding: 6px 8px; background: rgba(0,0,0,.5); color: #fff; font: 12px/1.3 ui-sans-serif, system-ui; border-radius: 6px; }
.gaea-hud .led { display:inline-block; width:8px; height:8px; border-radius:50%; background:#1abc9c; margin-right:6px; vertical-align:middle; }
`;
await fs.writeFile(path.join(DIST, 'gaea_floater.css'), CSS, 'utf8');

console.log('Floater build complete → dist/gaea_floater.*');
