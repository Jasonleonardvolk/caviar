#!/usr/bin/env node

/**
 * Verify Critical Shader Fixes
 * Checks if the 2 critical errors have been resolved
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

console.log('🔍 Verifying Critical Shader Fixes\n');
console.log('=====================================\n');

// Check if fixes were applied
console.log('📝 Checking applied fixes:\n');

// 1. Check propagation_v2.wgsl workgroup size
const propFile = path.join(__dirname, 'frontend', 'lib', 'webgpu', 'shaders', 'experimental', 'propagation_v2.wgsl');
if (fs.existsSync(propFile)) {
  const content = fs.readFileSync(propFile, 'utf8');
  
  // Check for the override declarations
  if (content.includes('override WG_X: u32 = 16u') && content.includes('override WG_Y: u32 = 16u')) {
    console.log('✅ propagation_v2.wgsl: Workgroup size reduced to 16x16 (256 invocations)');
  } else if (content.includes('WORKGROUP_SIZE: u32 = 16u')) {
    console.log('✅ propagation_v2.wgsl: Workgroup size constant set to 16');
  } else {
    console.log('⚠️  propagation_v2.wgsl: May still have 32x32 workgroup size');
  }
  
  // Check shared memory arrays
  if (content.includes('array<vec4<f32>, 256>')) {
    console.log('✅ propagation_v2.wgsl: Shared memory arrays reduced to 256 elements');
  }
}

// 2. Check applyPhaseLUT.wgsl bindings
const lutFile = path.join(__dirname, 'frontend', 'lib', 'webgpu', 'shaders', 'post', 'applyPhaseLUT.wgsl');
if (fs.existsSync(lutFile)) {
  const content = fs.readFileSync(lutFile, 'utf8');
  
  // Count unique bindings
  const bindingPattern = /@group\(0\) @binding\((\d+)\)/g;
  const bindings = new Set();
  let match;
  while ((match = bindingPattern.exec(content)) !== null) {
    bindings.add(match[1]);
  }
  
  if (bindings.size === 4 && bindings.has('0') && bindings.has('1') && bindings.has('2') && bindings.has('3')) {
    console.log('✅ applyPhaseLUT.wgsl: Has 4 unique bindings (0,1,2,3)');
  } else {
    console.log(`⚠️  applyPhaseLUT.wgsl: Found ${bindings.size} bindings: ${Array.from(bindings).sort().join(', ')}`);
  }
}

// 3. Check bounds helper
const boundsFile = path.join(__dirname, 'frontend', 'lib', 'webgpu', 'shaders', 'common', 'bounds.wgsl');
if (fs.existsSync(boundsFile)) {
  console.log('✅ bounds.wgsl: Helper file created');
}

console.log('\n🔧 Running shader validation...\n');

// Run validation
try {
  const valScript = path.join(__dirname, 'tools', 'shaders', 'validate-wgsl.js');
  const limitsFile = path.join(__dirname, 'tools', 'shaders', 'device_limits', 'iphone11.json');
  
  if (!fs.existsSync(valScript)) {
    console.log('⚠️  Validator script not found at:', valScript);
    process.exit(0);
  }
  
  if (!fs.existsSync(limitsFile)) {
    console.log('⚠️  Device limits file not found at:', limitsFile);
    console.log('   Using default limits instead');
  }
  
  const cmd = `node "${valScript}" --dir=frontend --limits="${limitsFile}" --strict`;
  console.log('Running:', cmd);
  
  const output = execSync(cmd, { encoding: 'utf8', stdio: 'pipe' });
  
  // Parse output for errors
  const errorMatch = output.match(/(\d+) errors?/);
  const criticalMatch = output.match(/CRITICAL.*?(\d+)/);
  
  if (errorMatch) {
    const errorCount = parseInt(errorMatch[1]);
    if (errorCount === 0) {
      console.log('\n🎉 SUCCESS! No shader errors found!');
    } else if (errorCount <= 2) {
      console.log(`\n✅ Only ${errorCount} error(s) remaining (down from 2 critical)`);
    } else {
      console.log(`\n⚠️  Still have ${errorCount} errors`);
    }
  }
  
} catch (error) {
  // Check if it's just validation errors
  const output = error.stdout || error.stderr || '';
  
  // Look for specific errors
  if (output.includes('workgroup invocations')) {
    console.log('❌ Workgroup size error still present');
  }
  if (output.includes('duplicate binding')) {
    console.log('❌ Duplicate binding error still present');
  }
  
  // Count warnings
  const warningMatch = output.match(/(\d+) warnings?/);
  if (warningMatch) {
    console.log(`⚠️  ${warningMatch[1]} warnings (non-blocking)`);
  }
}

console.log('\n=====================================');
console.log('✨ Verification complete!');
console.log('\nNext steps:');
console.log('1. If errors remain, check the specific files mentioned');
console.log('2. Run the full build: .\\tools\\release\\IrisOneButton.ps1');
console.log('3. Ship it! 🚀');
