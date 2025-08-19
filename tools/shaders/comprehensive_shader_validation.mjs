// comprehensive_shader_validation.mjs
// Runs all shader validation checks in sequence

import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function runCommand(command, args = []) {
  return new Promise((resolve, reject) => {
    const proc = spawn(command, args, { 
      shell: true, 
      stdio: 'inherit',
      cwd: path.join(__dirname, '..', '..')
    });
    
    proc.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`Command failed with code ${code}`));
      } else {
        resolve();
      }
    });
  });
}

async function runValidation() {
  console.log('🚀 Running Comprehensive Shader Validation\n');
  console.log('=' . repeat(50));
  
  try {
    // 1. Check for vec3 padding issues
    console.log('\n1️⃣  Checking vec3 padding in storage structs...');
    await runCommand('node', ['tools/shaders/guards/check_storage_vec3_padding.mjs']);
    
    // 2. Remove BOM and sync files
    console.log('\n2️⃣  Removing BOM and syncing files...');
    await runCommand('powershell', ['-ExecutionPolicy', 'Bypass', '-File', 'tools/shaders/scrub_bom_and_recopy.ps1']);
    
    // 3. Run WGSL validation
    console.log('\n3️⃣  Running WGSL validation...');
    await runCommand('node', [
      'tools/shaders/validate_and_report.mjs',
      '--dir=frontend/lib/webgpu/shaders',
      '--limits=tools/shaders/device_limits/iphone15.json',
      '--targets=naga',
      '--strict'
    ]);
    
    // 4. Apply const optimizations
    console.log('\n4️⃣  Applying const optimizations...');
    await runCommand('node', ['tools/shaders/fixes/apply_const_optimizations.mjs']);
    
    // 5. Final sync to public
    console.log('\n5️⃣  Final sync to public directory...');
    await runCommand('node', ['tools/copy_shaders_to_public.mjs']);
    
    console.log('\n' + '=' . repeat(50));
    console.log('✅ All validation checks passed!');
    console.log('\n📊 Summary:');
    console.log('  • Vec3 padding: ✓');
    console.log('  • BOM removal: ✓');
    console.log('  • WGSL validation: ✓');
    console.log('  • Const optimizations: ✓');
    console.log('  • File sync: ✓');
    
  } catch (error) {
    console.error('\n❌ Validation failed:', error.message);
    process.exit(1);
  }
}

runValidation();
