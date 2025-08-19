#!/usr/bin/env node
/**
 * Alternative WGSL validators if Tint is not available
 * This script provides fallback options for shader validation
 */

import { execSync } from 'child_process';
import { existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

console.log('🔍 Checking for WGSL validators...\n');

const validators = [
  {
    name: 'Tint',
    exe: 'tint.exe',
    test: '--version',
    download: 'https://dawn.googlesource.com/dawn/',
    priority: 1
  },
  {
    name: 'Naga',
    exe: 'naga.exe',
    test: '--version',
    download: 'https://github.com/gfx-rs/wgpu/releases',
    install: 'cargo install naga-cli',
    priority: 2
  },
  {
    name: 'wgpu_validate',
    exe: 'wgpu_validate.exe',
    test: '--help',
    download: 'https://github.com/gfx-rs/wgpu/releases',
    priority: 3
  }
];

const available = [];
const missing = [];

for (const validator of validators) {
  const path = join(__dirname, validator.exe);
  if (existsSync(path)) {
    try {
      execSync(`"${path}" ${validator.test}`, { stdio: 'ignore' });
      available.push(validator);
      console.log(`✅ ${validator.name} - FOUND at ${path}`);
    } catch (err) {
      console.log(`⚠️  ${validator.name} - EXISTS but won't run`);
      missing.push(validator);
    }
  } else {
    console.log(`❌ ${validator.name} - NOT FOUND`);
    missing.push(validator);
  }
}

console.log('\n' + '='.repeat(60));

if (available.length > 0) {
  console.log('\n✅ You can use:', available[0].name);
  console.log(`   Path: ${join(__dirname, available[0].exe)}`);
} else {
  console.log('\n❌ No WGSL validators found!');
  console.log('\nPlease install one of these:\n');
  
  for (const validator of validators) {
    console.log(`${validator.priority}. ${validator.name}:`);
    console.log(`   Download: ${validator.download}`);
    if (validator.install) {
      console.log(`   Or install: ${validator.install}`);
    }
    console.log();
  }
  
  console.log('Recommended: Download Tint from Dawn/Chromium builds');
  console.log('Alternative: Install Naga via Rust: cargo install naga-cli');
}

console.log('\n' + '='.repeat(60));
console.log('\nOnce installed, test with:');
console.log('  .\\test_tint.cmd');
console.log('  OR');
console.log('  node check_validators.mjs');
