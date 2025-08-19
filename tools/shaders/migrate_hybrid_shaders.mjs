#!/usr/bin/env node
/**
 * migrate_hybrid_shaders.mjs
 * Moves hybrid-specific shaders to canonical location before sync
 */
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const REPO = path.resolve(process.cwd());
const CANONICAL = path.join(REPO, 'frontend', 'lib', 'webgpu', 'shaders');
const HYBRID_SOURCE = path.join(REPO, 'frontend', 'hybrid');
const HYBRID_NESTED = path.join(REPO, 'frontend', 'hybrid', 'wgsl');
const PUBLIC_OUTPUT = path.join(REPO, 'frontend', 'public', 'hybrid', 'wgsl');

const colors = {
  reset: '\x1b[0m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m'
};

function log(color, ...args) {
  console.log(colors[color] + args.join(' ') + colors.reset);
}

async function main() {
  log('cyan', '\n========================================');
  log('cyan', '   Migrating Hybrid Shaders to Canonical');
  log('cyan', '========================================\n');
  
  // Ensure canonical exists
  if (!fs.existsSync(CANONICAL)) {
    fs.mkdirSync(CANONICAL, { recursive: true });
  }
  
  // Find hybrid shaders
  const hybridShaders = [];
  
  // Check frontend/hybrid/*.wgsl
  if (fs.existsSync(HYBRID_SOURCE)) {
    const files = fs.readdirSync(HYBRID_SOURCE)
      .filter(f => f.endsWith('.wgsl'));
    files.forEach(f => {
      hybridShaders.push({
        name: f,
        source: path.join(HYBRID_SOURCE, f)
      });
    });
  }
  
  // Check frontend/hybrid/wgsl/*.wgsl
  if (fs.existsSync(HYBRID_NESTED)) {
    const files = fs.readdirSync(HYBRID_NESTED)
      .filter(f => f.endsWith('.wgsl'));
    files.forEach(f => {
      // Only add if not already found
      if (!hybridShaders.find(s => s.name === f)) {
        hybridShaders.push({
          name: f,
          source: path.join(HYBRID_NESTED, f)
        });
      }
    });
  }
  
  log('blue', `Found ${hybridShaders.length} hybrid shaders to migrate:\n`);
  
  // Copy to canonical
  let migrated = 0;
  for (const shader of hybridShaders) {
    const canonicalPath = path.join(CANONICAL, shader.name);
    
    if (fs.existsSync(canonicalPath)) {
      log('yellow', `  ⚠️  ${shader.name} already exists in canonical - skipping`);
    } else {
      fs.copyFileSync(shader.source, canonicalPath);
      log('green', `  ✓ Migrated ${shader.name} to canonical`);
      migrated++;
    }
  }
  
  log('blue', `\nMigrated ${migrated} shaders to canonical location`);
  
  // Now clear and sync public
  log('blue', '\nClearing public output directory...');
  if (fs.existsSync(PUBLIC_OUTPUT)) {
    fs.rmSync(PUBLIC_OUTPUT, { recursive: true, force: true });
  }
  fs.mkdirSync(PUBLIC_OUTPUT, { recursive: true });
  
  // Copy ALL shaders from canonical to public
  const allCanonical = fs.readdirSync(CANONICAL)
    .filter(f => f.endsWith('.wgsl'));
  
  log('blue', `\nSyncing ${allCanonical.length} shaders to public output:\n`);
  
  for (const shader of allCanonical) {
    const src = path.join(CANONICAL, shader);
    const dst = path.join(PUBLIC_OUTPUT, shader);
    fs.copyFileSync(src, dst);
    log('green', `  ✓ ${shader}`);
  }
  
  log('cyan', '\n✅ Migration complete!');
  log('blue', '\nNext steps:');
  log('blue', '1. Run a new scan to verify conflicts are resolved');
  log('blue', '2. Review remaining conflicts between frontend/shaders and canonical');
  log('blue', '3. Remove original hybrid shader files after confirming migration');
}

main().catch(err => {
  console.error('Error:', err);
  process.exit(1);
});
