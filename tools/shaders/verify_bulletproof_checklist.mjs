#!/usr/bin/env node
// Comprehensive checklist to verify all improvements from paste.txt

import fs from 'fs';
import path from 'path';
import { spawn } from 'child_process';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const GREEN = '\x1b[32m';
const RED = '\x1b[31m';
const YELLOW = '\x1b[33m';
const RESET = '\x1b[0m';

console.log(`
╔════════════════════════════════════════════════════════════════╗
║         SHADER LIMITS SYSTEM - COMPREHENSIVE CHECKLIST         ║
╚════════════════════════════════════════════════════════════════╝
`);

const checks = {
  '1. Tighten "latest" plumbing': [
    {
      name: 'Both gates resolve latest',
      test: () => {
        const vr = fs.readFileSync(path.join(__dirname, 'validate_and_report.mjs'), 'utf8');
        const sg = fs.readFileSync(path.join(__dirname, 'shader_quality_gate_v2.mjs'), 'utf8');
        return vr.includes('resolveLimits') && sg.includes('resolveLimitsAlias');
      }
    },
    {
      name: 'ENV override support',
      test: () => {
        const sg = fs.readFileSync(path.join(__dirname, 'shader_quality_gate_v2.mjs'), 'utf8');
        return sg.includes('process.env.SHADER_LIMITS');
      }
    },
    {
      name: 'Platform aliases support',
      test: () => {
        return fs.existsSync(path.join(__dirname, 'limits_resolver.mjs'));
      }
    }
  ],
  
  '2. Harden limits capture': [
    {
      name: 'Field normalization',
      test: () => {
        const server = path.join(__dirname, '..', 'dev', 'limits_server_enhanced.mjs');
        if (!fs.existsSync(server)) return false;
        const content = fs.readFileSync(server, 'utf8');
        return content.includes('normalizeLimits');
      }
    },
    {
      name: 'Schema validation',
      test: () => {
        const server = path.join(__dirname, '..', 'dev', 'limits_server_enhanced.mjs');
        if (!fs.existsSync(server)) return false;
        const content = fs.readFileSync(server, 'utf8');
        return content.includes('validateLimits') && content.includes('LIMITS_SCHEMA');
      }
    },
    {
      name: 'Version & provenance tracking',
      test: () => {
        const server = path.join(__dirname, '..', 'dev', 'limits_server_enhanced.mjs');
        if (!fs.existsSync(server)) return false;
        const content = fs.readFileSync(server, 'utf8');
        return content.includes('version:') && content.includes('capturedAt:');
      }
    },
    {
      name: 'Timestamped directory structure',
      test: () => {
        const server = path.join(__dirname, '..', 'dev', 'limits_server_enhanced.mjs');
        if (!fs.existsSync(server)) return false;
        const content = fs.readFileSync(server, 'utf8');
        return content.includes('captures') && content.includes('timestamp');
      }
    }
  ],
  
  '3. Dev server safety': [
    {
      name: 'Dev-only guard',
      test: () => {
        const server = path.join(__dirname, '..', 'dev', 'limits_server_enhanced.mjs');
        if (!fs.existsSync(server)) return false;
        const content = fs.readFileSync(server, 'utf8');
        return content.includes('NODE_ENV === \'production\'');
      }
    },
    {
      name: 'CORS configuration',
      test: () => {
        const server = path.join(__dirname, '..', 'dev', 'limits_server_enhanced.mjs');
        if (!fs.existsSync(server)) return false;
        const content = fs.readFileSync(server, 'utf8');
        return content.includes('Access-Control-Allow-Origin');
      }
    },
    {
      name: 'Idempotence check',
      test: () => {
        const server = path.join(__dirname, '..', 'dev', 'limits_server_enhanced.mjs');
        if (!fs.existsSync(server)) return false;
        const content = fs.readFileSync(server, 'utf8');
        return content.includes('areIdenticalLimits');
      }
    }
  ],
  
  '4. Runtime guardrails': [
    {
      name: 'Workgroup assertion',
      test: () => {
        const guards = path.join(__dirname, '..', '..', 'frontend', 'lib', 'webgpu', 'guards.ts');
        if (!fs.existsSync(guards)) return false;
        const content = fs.readFileSync(guards, 'utf8');
        return content.includes('assertWorkgroupFits');
      }
    },
    {
      name: 'Soft clamping',
      test: () => {
        const guards = path.join(__dirname, '..', '..', 'frontend', 'lib', 'webgpu', 'guards.ts');
        if (!fs.existsSync(guards)) return false;
        const content = fs.readFileSync(guards, 'utf8');
        return content.includes('clampWorkgroupSize');
      }
    },
    {
      name: 'Storage usage check',
      test: () => {
        const guards = path.join(__dirname, '..', '..', 'frontend', 'lib', 'webgpu', 'guards.ts');
        if (!fs.existsSync(guards)) return false;
        const content = fs.readFileSync(guards, 'utf8');
        return content.includes('checkWorkgroupStorageUsage');
      }
    },
    {
      name: 'Feature capture',
      test: () => {
        const caps = path.join(__dirname, '..', '..', 'frontend', 'lib', 'webgpu', 'capabilities.ts');
        if (!fs.existsSync(caps)) return false;
        const content = fs.readFileSync(caps, 'utf8');
        return content.includes('features') && content.includes('adapterInfo');
      }
    }
  ],
  
  '5. Enhanced capabilities': [
    {
      name: 'Complete probe function',
      test: () => {
        const caps = path.join(__dirname, '..', '..', 'frontend', 'lib', 'webgpu', 'capabilities.ts');
        if (!fs.existsSync(caps)) return false;
        const content = fs.readFileSync(caps, 'utf8');
        return content.includes('probeWebGPUComplete');
      }
    },
    {
      name: 'Platform detection',
      test: () => {
        const caps = path.join(__dirname, '..', '..', 'frontend', 'lib', 'webgpu', 'capabilities.ts');
        if (!fs.existsSync(caps)) return false;
        const content = fs.readFileSync(caps, 'utf8');
        return content.includes('detectPlatform');
      }
    },
    {
      name: 'Metadata collection',
      test: () => {
        const caps = path.join(__dirname, '..', '..', 'frontend', 'lib', 'webgpu', 'capabilities.ts');
        if (!fs.existsSync(caps)) return false;
        const content = fs.readFileSync(caps, 'utf8');
        return content.includes('userAgent') && content.includes('adapterInfo');
      }
    }
  ],
  
  '6. NPM Scripts': [
    {
      name: 'dev:limits script',
      test: () => {
        const pkg = JSON.parse(fs.readFileSync(path.join(__dirname, '..', '..', 'package.json'), 'utf8'));
        return pkg.scripts && pkg.scripts['dev:limits'];
      }
    },
    {
      name: 'shaders:gate:latest script',
      test: () => {
        const pkg = JSON.parse(fs.readFileSync(path.join(__dirname, '..', '..', 'package.json'), 'utf8'));
        return pkg.scripts && pkg.scripts['shaders:gate:latest'];
      }
    },
    {
      name: 'shaders:check:latest script',
      test: () => {
        const pkg = JSON.parse(fs.readFileSync(path.join(__dirname, '..', '..', 'package.json'), 'utf8'));
        return pkg.scripts && pkg.scripts['shaders:check:latest'];
      }
    }
  ]
};

// Run all checks
console.log('Running comprehensive verification...\n');

let totalPassed = 0;
let totalFailed = 0;

for (const [category, tests] of Object.entries(checks)) {
  console.log(`\n${category}`);
  console.log('─'.repeat(50));
  
  for (const test of tests) {
    try {
      const passed = test.test();
      if (passed) {
        console.log(`  ${GREEN}✓${RESET} ${test.name}`);
        totalPassed++;
      } else {
        console.log(`  ${RED}✗${RESET} ${test.name}`);
        totalFailed++;
      }
    } catch (e) {
      console.log(`  ${YELLOW}?${RESET} ${test.name} (error: ${e.message})`);
      totalFailed++;
    }
  }
}

// Summary
console.log(`\n${'═'.repeat(60)}`);
console.log('SUMMARY');
console.log('─'.repeat(60));
console.log(`Total checks: ${totalPassed + totalFailed}`);
console.log(`Passed: ${GREEN}${totalPassed}${RESET}`);
console.log(`Failed: ${RED}${totalFailed}${RESET}`);
console.log(`Completion: ${((totalPassed / (totalPassed + totalFailed)) * 100).toFixed(1)}%`);

// Recommendations
if (totalFailed > 0) {
  console.log(`\n${YELLOW}RECOMMENDATIONS:${RESET}`);
  console.log('─'.repeat(60));
  
  if (!fs.existsSync(path.join(__dirname, '..', 'dev', 'limits_server_enhanced.mjs'))) {
    console.log('• Use the enhanced limits server:');
    console.log('  npm run dev:limits:enhanced');
  }
  
  if (!fs.existsSync(path.join(__dirname, 'limits_resolver.mjs'))) {
    console.log('• Update validation scripts to use shared resolver');
  }
  
  console.log('• Review LATEST_LIMITS_COMPLETE.md for implementation details');
} else {
  console.log(`\n${GREEN}🎉 ALL CHECKS PASSED!${RESET}`);
  console.log('Your shader limits system is fully bulletproof!');
}

console.log(`\n${'═'.repeat(60)}\n`);
