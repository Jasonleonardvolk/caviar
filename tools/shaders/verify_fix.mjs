#!/usr/bin/env node
// Quick verification that lightFieldComposerEnhanced.wgsl is fixed

import { exec } from 'child_process';
import { promisify } from 'util';
const execAsync = promisify(exec);

console.log('🔍 Verifying lightFieldComposerEnhanced.wgsl fix...\n');

const file = 'str(PROJECT_ROOT / "frontend\\hybrid\\wgsl\\lightFieldComposerEnhanced.wgsl';

async function verify() {
  try {
    // Run Naga validation
    const { stdout, stderr } = await execAsync(`naga validate "${file}" --input-kind wgsl`);
    
    if (stderr && stderr.includes('error')) {
      console.log('❌ Still has errors:');
      console.log(stderr);
      return false;
    }
    
    console.log('✅ lightFieldComposerEnhanced.wgsl FIXED!');
    console.log('   - No swizzle assignment errors');
    console.log('   - All textureLoad calls have 4 arguments');
    console.log('   - Naga validation PASSED\n');
    
    console.log('🎉 Run full validation to confirm 35/35:');
    console.log('   node tools/shaders/shader_quality_gate_v2.mjs --dir=frontend/ --strict --report=build/shader_report.json\n');
    
    return true;
  } catch (e) {
    if (e.message.includes('cannot assign to this expression')) {
      console.log('❌ Swizzle assignment still present!');
      console.log('Look for any ".rgb =" or ".rgb *=" patterns');
    } else if (e.message.includes('wrong number of args')) {
      console.log('❌ textureLoad still missing arguments!');
    } else {
      console.log('⚠️  Naga not found or other error:', e.message.split('\n')[0]);
    }
    return false;
  }
}

verify();
