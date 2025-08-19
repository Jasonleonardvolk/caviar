// Quick test to see validation results
const { execSync } = require('child_process');

console.log('🔍 Testing shader validation...\n');

try {
  // Run with more output options
  const cmd = 'node tools/shaders/validate_and_report.mjs --dir=frontend/lib/webgpu/shaders --limits=latest.ios --targets=naga';
  console.log(`Running: ${cmd}\n`);
  
  const output = execSync(cmd, { 
    encoding: 'utf8',
    stdio: 'pipe',
    maxBuffer: 10 * 1024 * 1024 // 10MB buffer
  });
  
  if (output) {
    console.log('Output:', output);
  } else {
    console.log('No output received');
  }
  
  // Try to find any JSON output files
  const fs = require('fs');
  const path = require('path');
  
  const reportDirs = [
    'tools/shaders/reports',
    'reports',
    '.'
  ];
  
  console.log('\n📄 Looking for report files...');
  for (const dir of reportDirs) {
    if (fs.existsSync(dir)) {
      const files = fs.readdirSync(dir);
      const reports = files.filter(f => f.includes('validation') || f.includes('report'));
      if (reports.length > 0) {
        console.log(`Found in ${dir}:`, reports);
        
        // Read the most recent one
        const latest = reports[reports.length - 1];
        const content = fs.readFileSync(path.join(dir, latest), 'utf8');
        try {
          const data = JSON.parse(content);
          console.log(`\n📊 Results from ${latest}:`);
          console.log(`  Passed: ${data.passed || data.summary?.passed || '?'}`);
          console.log(`  Failed: ${data.failed || data.summary?.failed || '?'}`);
          console.log(`  Warnings: ${data.warnings || data.summary?.warnings || '?'}`);
        } catch (e) {
          console.log('  (Could not parse as JSON)');
        }
      }
    }
  }
  
} catch (error) {
  console.log('Exit code:', error.status);
  if (error.stdout) {
    console.log('STDOUT:', error.stdout.toString());
  }
  if (error.stderr) {
    console.log('STDERR:', error.stderr.toString());
  }
}
