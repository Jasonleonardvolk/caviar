#!/usr/bin/env node
/**
 * Add device profile validation scripts to package.json
 */

import fs from 'node:fs';
import path from 'node:path';

const packagePath = path.join(process.cwd(), 'package.json');

console.log('Adding device profile validation scripts to package.json...');

try {
  const content = fs.readFileSync(packagePath, 'utf8');
  const pkg = JSON.parse(content);
  
  if (!pkg.scripts) {
    pkg.scripts = {};
  }
  
  // Add device profile scripts
  const profileScripts = {
    'shaders:gate:low': 'node tools/shaders/validate_and_report.mjs --dir=frontend/lib/webgpu/shaders --limits=tools/shaders/device_limits/desktop_low.json --targets=naga --strict',
    'shaders:gate:desktop': 'node tools/shaders/validate_and_report.mjs --dir=frontend/lib/webgpu/shaders --limits=tools/shaders/device_limits/desktop.json --targets=naga --strict',
    'shaders:gate:iphone': 'node tools/shaders/validate_and_report.mjs --dir=frontend/lib/webgpu/shaders --limits=tools/shaders/device_limits/iphone15.json --targets=naga --strict',
    'shaders:validate:all': 'npm run shaders:gate:low && npm run shaders:gate:desktop && npm run shaders:gate:iphone'
  };
  
  for (const [key, value] of Object.entries(profileScripts)) {
    if (!pkg.scripts[key]) {
      pkg.scripts[key] = value;
      console.log(`  Added: ${key}`);
    } else {
      console.log(`  Exists: ${key}`);
    }
  }
  
  fs.writeFileSync(packagePath, JSON.stringify(pkg, null, 2) + '\n');
  
  console.log('\npackage.json updated!');
  console.log('\nAvailable commands:');
  console.log('  npm run shaders:gate:low      - Validate against low-end desktop');
  console.log('  npm run shaders:gate:desktop  - Validate against regular desktop');
  console.log('  npm run shaders:gate:iphone   - Validate against iPhone 15');
  console.log('  npm run shaders:validate:all  - Run all profiles');
  
} catch (err) {
  console.error('Failed:', err);
  process.exit(1);
}