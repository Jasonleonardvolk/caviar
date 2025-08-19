#!/usr/bin/env node
// Debug script to show the actual JSON schema

import fs from 'fs';
import path from 'path';

const reportPath = path.join(process.cwd(), 'build', 'shader_report.json');

if (!fs.existsSync(reportPath)) {
  console.error('No report found at:', reportPath);
  process.exit(1);
}

const data = JSON.parse(fs.readFileSync(reportPath, 'utf8'));

console.log('=== JSON SCHEMA DEBUG ===\n');
console.log('Top-level keys:', Object.keys(data));
console.log('\nStructure:');

// Check for different schema patterns
if (data.files) {
  console.log('  - Has data.files (array):', Array.isArray(data.files));
  if (data.files[0]) {
    console.log('  - First file structure:', Object.keys(data.files[0]));
  }
}

if (data.passed) {
  console.log('  - Has data.passed (array):', Array.isArray(data.passed));
  console.log('  - Passed count:', data.passed.length);
}

if (data.failed) {
  console.log('  - Has data.failed (array):', Array.isArray(data.failed));
  console.log('  - Failed count:', data.failed.length);
}

if (data.warnings) {
  console.log('  - Has data.warnings (array):', Array.isArray(data.warnings));
  console.log('  - Warnings count:', data.warnings.length);
}

if (data.results) {
  console.log('  - Has data.results:', typeof data.results);
  if (data.results.files) {
    console.log('    - Has data.results.files:', Array.isArray(data.results.files));
  }
}

console.log('\n=== SAMPLE DATA ===\n');
console.log(JSON.stringify(data, null, 2).substring(0, 800));
