#!/usr/bin/env node

/**
 * Quick validation check for shader fixes
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

console.log('🚀 Running shader validation with latest limits\n');
console.log('============================================\n');

// Check if validator exists
const validators = [
  'tools/shaders/shader_quality_gate_v2.mjs',
  'tools/shaders/validate-wgsl.js',
  'tools/shaders/validate_and_report.mjs'
];

let validator = null;
for (const v of validators) {
  if (fs.existsSync(v)) {
    validator = v;
    break;
  }
}

if (!validator) {
  console.log('❌ No validator script found');
  process.exit(1);
}

console.log(`📄 Using validator: ${validator}\n`);

// Run with latest limits
try {
  const cmd = `node ${validator} --dir=frontend --limits=latest --strict`;
  console.log(`Running: ${cmd}\n`);
  
  const output = execSync(cmd, { encoding: 'utf8', stdio: 'pipe' });
  console.log(output);
  
  console.log('\n✅ Validation passed!');
} catch (error) {
  const output = error.stdout || error.stderr || '';
  
  // Parse for specific issues
  if (output.includes('applyPhaseLUT')) {
    console.log('❌ applyPhaseLUT still has issues');
    
    // Check what the issue is
    if (output.includes('duplicate')) {
      console.log('   Still seeing duplicate bindings');
    }
    if (output.includes('texture')) {
      console.log('   Texture/sampler issue');
    }
  }
  
  if (output.includes('propagation_v2')) {
    console.log('❌ propagation_v2 still has issues');
  }
  
  // Count errors vs warnings
  const errorMatch = output.match(/(\d+)\s+fail/);
  const warnMatch = output.match(/(\d+)\s+warn/);
  
  if (errorMatch) {
    console.log(`\n📊 Results: ${errorMatch[1]} failures`);
  }
  if (warnMatch) {
    console.log(`📊 Warnings: ${warnMatch[1]} (non-blocking)`);
  }
  
  console.log('\n' + output.substring(0, 1000)); // First 1000 chars
}

console.log('\n============================================');
console.log('💡 If issues persist:');
console.log('1. Clear any build caches');
console.log('2. Check the exact error message above');
console.log('3. Verify the pipeline code matches the shader bindings');
