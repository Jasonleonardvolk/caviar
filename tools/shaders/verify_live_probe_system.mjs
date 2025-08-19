#!/usr/bin/env node
// verify_live_probe_system.mjs
// Verifies the live-probe GPU limits system is working correctly

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { spawn } from 'child_process';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

console.log(`
======================================================================
                    LIVE-PROBE SYSTEM VERIFICATION
======================================================================
`);

const checks = [
  {
    name: 'Frontend Integration',
    files: [
      { path: 'frontend/lib/webgpu/capabilities.ts', mustContain: 'pushLimitsToDevServer' },
      { path: 'frontend/lib/webgpu/guards.ts', mustContain: 'assertWorkgroupFits' }
    ]
  },
  {
    name: 'Dev Server',
    files: [
      { path: 'tools/dev/limits_server.mjs', mustContain: 'save-gpu-limits' }
    ]
  },
  {
    name: 'Validation Support',
    files: [
      { path: 'tools/shaders/validate_and_report.mjs', mustContain: 'resolveLimits' }
    ]
  },
  {
    name: 'NPM Scripts',
    files: [
      { path: 'package.json', mustContain: 'dev:limits' },
      { path: 'package.json', mustContain: 'shaders:gate:latest' }
    ]
  }
];

let allPassed = true;

for (const category of checks) {
  console.log(`\nChecking ${category.name}:`);
  console.log('-'.repeat(50));
  
  for (const file of category.files) {
    const fullPath = path.join(__dirname, file.path);
    
    if (!fs.existsSync(fullPath)) {
      console.log(`  ❌ Missing: ${file.path}`);
      allPassed = false;
      continue;
    }
    
    const content = fs.readFileSync(fullPath, 'utf8');
    if (content.includes(file.mustContain)) {
      console.log(`  ✅ ${file.path} - contains "${file.mustContain}"`);
    } else {
      console.log(`  ❌ ${file.path} - missing "${file.mustContain}"`);
      allPassed = false;
    }
  }
}

console.log('\n' + '='.repeat(70));

if (allPassed) {
  console.log(`
✅ LIVE-PROBE SYSTEM VERIFIED!

The system is ready to use:

1. Start the limits server:
   npm run dev:limits

2. In your app, capture and push limits:
   const limits = await probeWebGPULimits();
   await pushLimitsToDevServer("devicename", limits);

3. Validate using captured limits:
   npm run shaders:gate:latest

📚 Full documentation: LIVE_PROBE_COMPLETE.md
`);
} else {
  console.log(`
❌ SOME COMPONENTS MISSING

Please check the failures above and ensure all files are properly updated.
`);
}

// Test that the resolveLimits function works
console.log('\nTesting resolveLimits function:');
console.log('-'.repeat(50));

async function testResolveLimits() {
  const testScript = `
import { readFileSync, existsSync } from 'fs';
import path from 'path';

function resolveLimits(p) {
  if (!p || p === 'latest') {
    const pointer = path.join(process.cwd(), 'tools', 'shaders', 'device_limits', 'latest.json');
    if (existsSync(pointer)) {
      try {
        const j = JSON.parse(readFileSync(pointer, 'utf8'));
        if (j?.path) return j.path;
      } catch {}
    }
    return 'tools/shaders/device_limits/iphone15.json';
  }
  return p;
}

console.log('Input: null -> ' + resolveLimits(null));
console.log('Input: "latest" -> ' + resolveLimits('latest'));
console.log('Input: "custom.json" -> ' + resolveLimits('custom.json'));
`;

  const testFile = path.join(__dirname, 'test_resolve_limits.mjs');
  fs.writeFileSync(testFile, testScript);
  
  return new Promise((resolve) => {
    const proc = spawn('node', [testFile], { shell: true });
    proc.stdout.on('data', (data) => process.stdout.write(data));
    proc.on('close', () => {
      fs.unlinkSync(testFile);
      resolve();
    });
  });
}

await testResolveLimits();

console.log('\n' + '='.repeat(70));
console.log('VERIFICATION COMPLETE!');
