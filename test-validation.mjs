import { execSync } from 'child_process';
import fs from 'fs';
import path from 'path';

console.log('🔍 Checking shader validation results...\n');

// First, look for any report files
const reportDirs = [
  'tools/shaders/reports',
  'reports',
  'tools/release/error_logs'
];

console.log('📄 Looking for report files...\n');
for (const dir of reportDirs) {
  if (fs.existsSync(dir)) {
    const files = fs.readdirSync(dir);
    const recent = files.filter(f => 
      f.includes('validation') || 
      f.includes('report') || 
      f.includes('latest')
    ).sort().reverse();
    
    if (recent.length > 0) {
      console.log(`Found in ${dir}:`);
      recent.slice(0, 3).forEach(f => {
        const stats = fs.statSync(path.join(dir, f));
        const age = Date.now() - stats.mtime.getTime();
        const mins = Math.floor(age / 60000);
        console.log(`  ${f} (${mins} mins ago)`);
      });
      
      // Try to read the most recent
      const latest = path.join(dir, recent[0]);
      try {
        const content = fs.readFileSync(latest, 'utf8');
        if (content.includes('{')) {
          const data = JSON.parse(content);
          console.log(`\n📊 Results from ${recent[0]}:`);
          console.log('  Total:', data.total || data.summary?.total || '?');
          console.log('  Passed:', data.passed || data.summary?.passed || '?');
          console.log('  Failed:', data.failed || data.summary?.failed || '?');
          console.log('  Warnings:', data.warnings || data.summary?.warnings || '?');
          
          if (data.failed === 0 || data.summary?.failed === 0) {
            console.log('\n✅ NO FAILURES - Ready to ship!');
          }
        }
      } catch (e) {
        // Try as text
        if (content.includes('pass') || content.includes('PASS')) {
          console.log('\n✅ Report indicates PASS');
        }
      }
    }
  }
}

// Try running with explicit output
console.log('\n🔧 Running with explicit output options...\n');
try {
  const result = execSync('node tools/shaders/validate_and_report.mjs --dir=frontend/lib/webgpu/shaders --limits=latest.ios --targets=naga --format=json', {
    encoding: 'utf8'
  });
  
  if (result) {
    console.log('JSON Output:', result);
  } else {
    console.log('✅ No output usually means SUCCESS (0 errors)!');
  }
} catch (e) {
  if (e.status === 0) {
    console.log('✅ Exit code 0 - SUCCESS!');
  } else {
    console.log('Exit code:', e.status);
  }
}

// Check the actual shader file
console.log('\n📝 Checking applyPhaseLUT.wgsl...');
const shaderPath = 'frontend/lib/webgpu/shaders/post/applyPhaseLUT.wgsl';
if (fs.existsSync(shaderPath)) {
  const content = fs.readFileSync(shaderPath, 'utf8');
  if (content.includes('textureSampleLevel')) {
    console.log('✅ Uses textureSampleLevel (correct for compute)');
  } else if (content.includes('textureSample')) {
    console.log('❌ Still uses textureSample (needs Level suffix)');
  }
  
  // Count bindings
  const bindings = content.match(/@binding\((\d+)\)/g);
  if (bindings) {
    console.log(`Found ${bindings.length} bindings:`, bindings.join(', '));
  }
}
