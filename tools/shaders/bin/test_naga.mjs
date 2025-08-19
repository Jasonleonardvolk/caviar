#!/usr/bin/env node
/**
 * Test Naga installation and create compatibility wrapper
 */

import { execSync } from 'child_process';
import { existsSync, writeFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const nagaPath = join(__dirname, 'naga.exe');

console.log('🔍 Testing Naga installation...\n');

if (!existsSync(nagaPath)) {
  console.error('❌ naga.exe not found at:', nagaPath);
  process.exit(1);
}

try {
  // Test version
  const version = execSync(`"${nagaPath}" --version`, { encoding: 'utf8' });
  console.log('✅ Naga Version:', version.trim());
  
  // Test validation with simple shader
  const testShader = `
@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
`;
  
  const testFile = join(__dirname, 'test_shader.wgsl');
  writeFileSync(testFile, testShader);
  
  console.log('\n📝 Testing WGSL validation...');
  const result = execSync(`"${nagaPath}" "${testFile}"`, { encoding: 'utf8' });
  console.log('✅ Validation successful!');
  console.log(result || '   (No errors)');
  
  // Clean up
  require('fs').unlinkSync(testFile);
  
  console.log('\n' + '='.repeat(60));
  console.log('✅ NAGA IS WORKING!');
  console.log('='.repeat(60));
  
  console.log('\nNaga Commands:');
  console.log('  Validate:  naga.exe shader.wgsl');
  console.log('  Info:      naga.exe info shader.wgsl');
  console.log('  To SPIR-V: naga.exe shader.wgsl output.spv');
  console.log('  To HLSL:   naga.exe shader.wgsl output.hlsl --hlsl');
  console.log('  To MSL:    naga.exe shader.wgsl output.metal --msl');
  
  console.log('\n🚀 Next: Run Virgil to validate all shaders:');
  console.log('  cd ..\\..\\..  (back to repo root)');
  console.log('  npm run virgil');
  
} catch (err) {
  console.error('❌ Naga test failed:', err.message);
  process.exit(1);
}
