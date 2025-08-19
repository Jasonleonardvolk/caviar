#!/usr/bin/env node
/**
 * check_shader_dates.mjs
 * Quick check to see which shader versions are newer
 */
import fs from 'fs';
import path from 'path';

const conflicts = [
  'avatarShader.wgsl',
  'lenticularInterlace.wgsl',
  'multiViewSynthesis.wgsl',
  'propagation.wgsl',
  'velocityField.wgsl'
];

const locations = [
  'frontend/shaders',
  'frontend/lib/webgpu/shaders'
];

console.log('\nShader Version Comparison (by last modified date):');
console.log('==================================================\n');

for (const shader of conflicts) {
  console.log(`📄 ${shader}:`);
  
  for (const loc of locations) {
    const filePath = path.join(loc, shader);
    if (fs.existsSync(filePath)) {
      const stats = fs.statSync(filePath);
      const size = stats.size;
      const modified = stats.mtime.toISOString().split('T')[0];
      const isCanonical = loc.includes('lib/webgpu');
      const marker = isCanonical ? '✅ [CANONICAL]' : '⚠️  [LEGACY]';
      
      console.log(`  ${marker} ${loc}/`);
      console.log(`    Modified: ${modified}, Size: ${size} bytes`);
    }
  }
  console.log('');
}

console.log('Recommendation:');
console.log('If CANONICAL is newer → Safe to run dedupe_shaders_v2.mjs');
console.log('If LEGACY is newer → Manually copy changes to canonical first!');
