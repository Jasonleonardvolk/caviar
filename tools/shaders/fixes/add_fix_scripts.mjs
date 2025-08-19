#!/usr/bin/env node
/**
 * Adds shader fix scripts to package.json
 */

import fs from 'fs';
import path from 'path';

const packagePath = path.join(process.cwd(), 'package.json');

console.log('📦 Adding shader fix scripts to package.json...');

try {
  const content = fs.readFileSync(packagePath, 'utf8');
  const pkg = JSON.parse(content);
  
  if (!pkg.scripts) {
    pkg.scripts = {};
  }
  
  // Add fix scripts
  const fixScripts = {
    'shaders:fix:vec3': 'node tools/shaders/fixes/fix_vec3_alignment.mjs',
    'shaders:fix:bounds': 'node tools/shaders/fixes/fix_bounds_checking.mjs',
    'shaders:fix:const': 'node tools/shaders/fixes/fix_const_let.mjs',
    'shaders:fix:all': 'node tools/shaders/fixes/fix_all.mjs',
    'shaders:fix': 'npm run shaders:fix:all && npm run shaders:sync && npm run shaders:gate:iphone'
  };
  
  let added = 0;
  for (const [key, value] of Object.entries(fixScripts)) {
    if (!pkg.scripts[key]) {
      pkg.scripts[key] = value;
      console.log(`  ✅ Added: ${key}`);
      added++;
    } else {
      console.log(`  ⏭️  Exists: ${key}`);
    }
  }
  
  if (added > 0) {
    fs.writeFileSync(packagePath, JSON.stringify(pkg, null, 2) + '\n');
    console.log(`\n✅ Added ${added} new scripts to package.json`);
  } else {
    console.log('\n⏭️  All scripts already exist');
  }
  
  console.log('\n📋 Available fix commands:');
  console.log('  npm run shaders:fix:vec3   - Fix vec3 storage alignment');
  console.log('  npm run shaders:fix:bounds - Add bounds checking');
  console.log('  npm run shaders:fix:const  - Fix const/let warnings');
  console.log('  npm run shaders:fix:all    - Run all fixes');
  console.log('  npm run shaders:fix        - Fix, sync, and validate');
  
} catch (err) {
  console.error('❌ Failed:', err);
  process.exit(1);
}