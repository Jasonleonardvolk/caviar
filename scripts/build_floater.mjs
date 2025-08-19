import esbuild from 'esbuild';
import path from 'node:path';
import { promises as fs } from 'node:fs';

const ROOT = process.cwd();
const SRC = path.join(ROOT, 'frontend', 'floater', 'src');
const DIST = path.join(ROOT, 'frontend', 'floater', 'dist');

// Create directories
await fs.mkdir(SRC, { recursive: true });
await fs.mkdir(DIST, { recursive: true });

// Create minimal host.ts if doesn't exist
const hostPath = path.join(SRC, 'host.ts');
try {
  await fs.access(hostPath);
} catch {
  await fs.writeFile(hostPath, `// Floater Host API
export interface FloaterConfig {
  mount: HTMLCanvasElement | HTMLVideoElement;
  dims: [number, number];
  kernel: 'onnx_waveop' | 'splitstep_fft' | 'biharmonic';
}

export async function createFloater(config: FloaterConfig) {
  console.log('Floater created with config:', config);
  return {
    start: () => console.log('Floater started'),
    stop: () => console.log('Floater stopped'),
    destroy: () => console.log('Floater destroyed')
  };
}
`, 'utf8');
}

// Create minimal worker if doesn't exist
const workerPath = path.join(SRC, 'renderer.worker.ts');
try {
  await fs.access(workerPath);
} catch {
  await fs.writeFile(workerPath, `// Floater Worker
self.addEventListener('message', (e) => {
  console.log('Worker received:', e.data);
  self.postMessage({ status: 'ok' });
});
`, 'utf8');
}

// Build ESM host
await esbuild.build({
  entryPoints: [hostPath],
  outfile: path.join(DIST, 'gaea_floater.esm.js'),
  bundle: true,
  format: 'esm',
  target: ['es2022'],
  sourcemap: true,
  loader: { '.wgsl': 'text' },
  external: ['onnxruntime-web']
});

// Build Worker
await esbuild.build({
  entryPoints: [workerPath],
  outfile: path.join(DIST, 'gaea_floater.worker.js'),
  bundle: true,
  format: 'esm',
  target: ['es2022'],
  sourcemap: true,
  loader: { '.wgsl': 'text' },
  external: ['onnxruntime-web']
});

console.log('Floater build complete → frontend/floater/dist/');
