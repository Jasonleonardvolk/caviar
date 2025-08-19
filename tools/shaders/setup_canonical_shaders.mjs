#!/usr/bin/env node
/**
 * setup_canonical_shaders.mjs
 * Sets up the canonical shader structure and copies to build output
 * Run this after deduplication to ensure build uses correct shaders
 */
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const REPO = path.resolve(process.cwd());
const CANONICAL = path.join(REPO, 'frontend', 'lib', 'webgpu', 'shaders');
const BUILD_OUTPUT = path.join(REPO, 'frontend', 'public', 'hybrid', 'wgsl');

const colors = {
  reset: '\x1b[0m',
  green: '\x1b[32m',
  blue: '\x1b[34m',
  yellow: '\x1b[33m',
  cyan: '\x1b[36m'
};

function log(color, ...args) {
  console.log(colors[color] + args.join(' ') + colors.reset);
}

async function main() {
  log('cyan', '\n========================================');
  log('cyan', '   Setting Up Canonical Shader Structure');
  log('cyan', '========================================\n');
  
  // Ensure canonical directory exists
  if (!fs.existsSync(CANONICAL)) {
    log('yellow', `Creating canonical directory: ${CANONICAL}`);
    fs.mkdirSync(CANONICAL, { recursive: true });
  }
  
  // List shaders in canonical
  const shaders = fs.readdirSync(CANONICAL)
    .filter(f => f.endsWith('.wgsl'));
  
  log('blue', `Found ${shaders.length} shaders in canonical location:`);
  shaders.forEach(s => log('green', `  ✓ ${s}`));
  
  // Sync to build output
  log('blue', `\nSyncing to build output: ${BUILD_OUTPUT}`);
  
  // Clear and recreate build output
  if (fs.existsSync(BUILD_OUTPUT)) {
    fs.rmSync(BUILD_OUTPUT, { recursive: true, force: true });
  }
  fs.mkdirSync(BUILD_OUTPUT, { recursive: true });
  
  // Copy all shaders to build output
  let copied = 0;
  for (const shader of shaders) {
    const src = path.join(CANONICAL, shader);
    const dst = path.join(BUILD_OUTPUT, shader);
    fs.copyFileSync(src, dst);
    copied++;
  }
  
  log('green', `✓ Copied ${copied} shaders to build output`);
  
  // Create import map for TypeScript
  const importMap = {
    generated: new Date().toISOString(),
    canonical: 'frontend/lib/webgpu/shaders',
    runtime: 'frontend/public/hybrid/wgsl',
    shaders: {}
  };
  
  for (const shader of shaders) {
    const name = shader.replace('.wgsl', '');
    importMap.shaders[name] = {
      canonical: `@/lib/webgpu/shaders/${shader}`,
      runtime: `/hybrid/wgsl/${shader}`
    };
  }
  
  const mapPath = path.join(CANONICAL, 'shader_map.json');
  fs.writeFileSync(mapPath, JSON.stringify(importMap, null, 2));
  log('green', `✓ Generated shader import map: ${path.relative(REPO, mapPath)}`);
  
  log('cyan', '\n✅ Canonical shader structure ready!');
  log('blue', '\nNext steps:');
  log('blue', '1. Update imports to use @/lib/webgpu/shaders/');
  log('blue', '2. Add prebuild script to package.json:');
  log('yellow', '   "prebuild": "node tools/shaders/setup_canonical_shaders.mjs"');
  log('blue', '3. Run shader validation:');
  log('yellow', '   node tools/shaders/shader_quality_gate_v2.mjs --dir=frontend/lib/webgpu/shaders');
}

main().catch(err => {
  console.error('Error:', err);
  process.exit(1);
});
