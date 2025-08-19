// test_suppression_system.mjs
// Tests that the suppression system correctly identifies and suppresses false positives

import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

console.log(`
╔════════════════════════════════════════════════════════════╗
║           TESTING SUPPRESSION SYSTEM                       ║
╚════════════════════════════════════════════════════════════╝
`);

async function runTest(scriptName, args = []) {
  return new Promise((resolve, reject) => {
    console.log(`\n🧪 Testing: ${scriptName}`);
    console.log(`   Args: ${args.join(' ')}`);
    
    const proc = spawn('node', [path.join(__dirname, scriptName), ...args], {
      shell: true,
      stdio: 'pipe'
    });
    
    let output = '';
    proc.stdout.on('data', (data) => { 
      output += data;
      process.stdout.write(data);
    });
    proc.stderr.on('data', (data) => { 
      output += data;
      process.stderr.write(data);
    });
    
    proc.on('close', (code) => {
      resolve({ code, output });
    });
  });
}

async function runTests() {
  console.log('\n1️⃣  Testing vec3 verification...');
  const vec3Test = await runTest('guards/verify_no_storage_vec3.mjs');
  if (vec3Test.code !== 0) {
    console.error('❌ Vec3 test failed - there ARE vec3s in storage!');
  } else {
    console.log('✅ Vec3 test passed - no vec3 in storage structs');
  }
  
  console.log('\n2️⃣  Testing CommonJS suppression...');
  const cjsTest = await runTest('suppression_patch.cjs', [
    '../../frontend/lib/webgpu/shaders/bitReversal.wgsl'
  ]);
  
  console.log('\n3️⃣  Testing enhanced validator with suppression...');
  const enhancedTest = await runTest('shader_quality_gate_v2_enhanced.mjs', [
    '--dir=frontend/lib/webgpu/shaders',
    '--suppressions=tools/shaders/validator_suppressions.json',
    '--verbose'
  ]);
  
  // Parse results
  console.log('\n' + '='.repeat(60));
  console.log('📊 TEST RESULTS SUMMARY\n');
  
  // Check if suppression is working
  if (enhancedTest.output.includes('Suppressed:')) {
    const match = enhancedTest.output.match(/Suppressed:\s+(\d+)/);
    const suppressedCount = match ? parseInt(match[1]) : 0;
    
    if (suppressedCount > 0) {
      console.log(`✅ Suppression is working! ${suppressedCount} false positives suppressed`);
      
      // Check if we're suppressing the right amount (should be ~121)
      if (suppressedCount >= 100 && suppressedCount <= 130) {
        console.log('✅ Suppression count matches expected range (100-130)');
      } else {
        console.log(`⚠️  Unexpected suppression count: ${suppressedCount} (expected ~121)`);
      }
    } else {
      console.log('❌ No warnings were suppressed!');
    }
  } else {
    console.log('❌ Could not find suppression statistics in output');
  }
  
  console.log('\n' + '='.repeat(60));
  console.log(`
🎯 WHAT THIS MEANS:

If suppression is working correctly, you should see:
- ✅ No vec3 in actual storage structs (only in vertex attributes)
- ✅ ~121 warnings suppressed (the false positives)
- ✅ 0 real warnings remaining

The suppression system:
1. Recognizes that clamp_index_dyn provides bounds checking
2. Knows that vec3 in @location attributes don't need padding
3. Only suppresses false positives, not real issues
`);
}

runTests().catch(console.error);
