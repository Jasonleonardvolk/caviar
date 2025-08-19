#!/usr/bin/env node
// Quick smoke test for the unified validation system

import { execSync } from 'child_process';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const GREEN = '\x1b[32m';
const RED = '\x1b[31m';
const YELLOW = '\x1b[33m';
const RESET = '\x1b[0m';

console.log(`
╔════════════════════════════════════════════════════════════════╗
║           UNIFIED VALIDATION SYSTEM - SMOKE TEST               ║
╚════════════════════════════════════════════════════════════════╝
`);

const tests = [
  {
    name: 'Shared validation library exists',
    test: () => fs.existsSync(path.join(__dirname, 'lib', 'limits_schema.js'))
  },
  {
    name: 'CLI validation tool exists',
    test: () => fs.existsSync(path.join(__dirname, 'lib', 'validate_limits_cli.mjs'))
  },
  {
    name: 'Enhanced server v2 exists',
    test: () => fs.existsSync(path.join(__dirname, '..', 'dev', 'limits_server_enhanced_v2.mjs'))
  },
  {
    name: 'Unified resolver v2 exists',
    test: () => fs.existsSync(path.join(__dirname, 'limits_resolver_v2.mjs'))
  },
  {
    name: 'Updated validators exist',
    test: () => {
      const v1 = fs.existsSync(path.join(__dirname, 'validate_and_report_v2.mjs'));
      const v2 = fs.existsSync(path.join(__dirname, 'shader_quality_gate_v2_unified.mjs'));
      return v1 && v2;
    }
  },
  {
    name: 'JSON schema exists',
    test: () => fs.existsSync(path.join(__dirname, 'schemas', 'limits.schema.json'))
  },
  {
    name: 'Test limits validation',
    test: () => {
      try {
        const testLimits = {
          maxComputeInvocationsPerWorkgroup: 256,
          maxComputeWorkgroupSizeX: 256,
          maxComputeWorkgroupSizeY: 256,
          maxComputeWorkgroupSizeZ: 64
        };
        
        const { validateLimitsSchema } = require('./lib/limits_schema.js');
        const result = validateLimitsSchema(testLimits, { platform: 'ios' });
        return result.ok;
      } catch {
        return false;
      }
    }
  },
  {
    name: 'Test CLI validator',
    test: () => {
      try {
        const limitsFile = path.join(__dirname, 'device_limits', 'iphone15.json');
        if (!fs.existsSync(limitsFile)) return false;
        
        execSync(`node ${path.join(__dirname, 'lib', 'validate_limits_cli.mjs')} --in=${limitsFile} --platform=ios`, {
          stdio: 'pipe'
        });
        return true;
      } catch {
        return false;
      }
    }
  }
];

let passed = 0;
let failed = 0;

for (const { name, test } of tests) {
  try {
    if (test()) {
      console.log(`${GREEN}✓${RESET} ${name}`);
      passed++;
    } else {
      console.log(`${RED}✗${RESET} ${name}`);
      failed++;
    }
  } catch (e) {
    console.log(`${RED}✗${RESET} ${name} (error: ${e.message})`);
    failed++;
  }
}

console.log(`
═══════════════════════════════════════════════════════════════
RESULTS: ${GREEN}${passed} passed${RESET}, ${failed > 0 ? RED : ''}${failed} failed${RESET}
═══════════════════════════════════════════════════════════════
`);

if (failed === 0) {
  console.log(`${GREEN}🎉 All smoke tests passed! The unified validation system is ready.${RESET}\n`);
  
  console.log('Quick commands to try:');
  console.log('─'.repeat(60));
  console.log('# Start enhanced server v2:');
  console.log('  npm run dev:limits:v2\n');
  console.log('# Validate a limits file:');
  console.log('  npm run limits:check -- --in=tools/shaders/device_limits/iphone15.json --platform=ios\n');
  console.log('# Run validation with platform alias:');
  console.log('  npm run shaders:gate:v2 -- --limits=latest.ios\n');
  console.log('# Run with auto-detection:');
  console.log('  npm run shaders:gate:v2 -- --limits=auto\n');
} else {
  console.log(`${YELLOW}⚠ Some tests failed. Check the implementation.${RESET}\n`);
}

process.exit(failed > 0 ? 1 : 0);
