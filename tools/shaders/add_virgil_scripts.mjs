#!/usr/bin/env node
/**
 * Add Virgil scripts to package.json
 * Run once to add the Dante-themed shader scripts
 */

import fs from 'node:fs';
import path from 'node:path';

const packagePath = path.join(process.cwd(), 'package.json');

console.log('📦 Adding Virgil scripts to package.json...');

try {
  const content = fs.readFileSync(packagePath, 'utf8');
  const pkg = JSON.parse(content);
  
  if (!pkg.scripts) {
    pkg.scripts = {};
  }
  
  // Add Virgil scripts
  const virgilScripts = {
    'shaders:sync': 'node tools/shaders/copy_canonical_to_public.mjs',
    'shaders:gate': 'node tools/shaders/validate_and_report.mjs --dir=frontend --limits=tools/shaders/device_limits/iphone15.json --targets=msl,hlsl,spirv --strict',
    'virgil': 'node tools/shaders/virgil_summon.mjs --strict',
    'shaders:check': 'node tools/shaders/guards/check_uniform_arrays.mjs --scan',
    'shaders:lethe': 'node tools/shaders/lethe_reindex_reports.mjs',
    'paradiso': 'powershell -ExecutionPolicy Bypass -File tools/release/crown_paradiso.ps1'
  };
  
  for (const [key, value] of Object.entries(virgilScripts)) {
    if (!pkg.scripts[key]) {
      pkg.scripts[key] = value;
      console.log(`  ✅ Added: ${key}`);
    } else {
      console.log(`  ⏭️  Exists: ${key}`);
    }
  }
  
  fs.writeFileSync(packagePath, JSON.stringify(pkg, null, 2) + '\n');
  
  console.log('\n✅ package.json updated!');
  console.log('\nRun: npm run virgil');
  
} catch (err) {
  console.error('❌ Failed:', err);
  process.exit(1);
}
