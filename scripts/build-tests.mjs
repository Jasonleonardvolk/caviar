#!/usr/bin/env node

import * as esbuild from 'esbuild';
import { promises as fs } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, '..');

async function bundle(input, output) {
  const inputPath = path.resolve(rootDir, input);
  const outputPath = path.resolve(rootDir, output);
  
  console.log(`Building ${input} -> ${output}`);
  
  try {
    await esbuild.build({
      entryPoints: [inputPath],
      bundle: true,
      format: 'esm',
      platform: 'browser',
      target: 'es2020',
      outfile: outputPath,
      sourcemap: true,
      loader: {
        '.ts': 'ts',
        '.tsx': 'tsx',
        '.wgsl': 'text'
      },
      define: {
        'process.env.NODE_ENV': '"development"'
      }
    });
    console.log(`✓ Built ${output}`);
  } catch (err) {
    console.error(`✗ Failed to build ${input}:`, err);
    process.exit(1);
  }
}

async function main() {
  // Build test files
  await bundle('tests/quilt_display.ts', 'frontend/public/tests/quilt_display.bundle.js');
  
  // Add more test builds here as needed
  // await bundle('tests/another_test.ts', 'frontend/public/tests/another_test.bundle.js');
  
  console.log('All tests built successfully!');
}

main().catch(err => {
  console.error('Build failed:', err);
  process.exit(1);
});