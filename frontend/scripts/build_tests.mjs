/* frontend/scripts/build_tests.mjs */
import esbuild from 'esbuild';
import path from 'node:path';

const ROOT = process.cwd();

const entries = [
  path.join(ROOT, 'frontend', 'public', 'tests', 'waveop_dashboard.ts'),
  path.join(ROOT, 'frontend', 'public', 'tests', 'schrodinger_bench.ts'),
];

await esbuild.build({
  entryPoints: entries,
  outdir: path.join(ROOT, 'frontend', 'public', 'tests'),
  format: 'esm',
  bundle: true,
  sourcemap: true,
  loader: {
    '.wgsl': 'text',
  },
  define: { 'process.env.NODE_ENV': '"production"' },
});
console.log('Built test bundles.');
