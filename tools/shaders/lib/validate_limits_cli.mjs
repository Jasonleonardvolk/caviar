#!/usr/bin/env node
// tools/shaders/lib/validate_limits_cli.mjs
// CLI tool to validate limits files by hand

const path = require('node:path');
const fs = require('node:fs');
const { validateLimitsSchema } = require('./limits_schema.js');

const args = Object.fromEntries(process.argv.slice(2).map(s => {
  const [k,v] = s.split('=');
  return [k.replace(/^--/, ''), v ?? true];
}));

if (!args.in) {
  console.error('Usage: node tools/shaders/lib/validate_limits_cli.mjs --in=tools/shaders/device_limits/iphone15.json [--platform=ios]');
  process.exit(2);
}

const abs = path.resolve(process.cwd(), args.in);

if (!fs.existsSync(abs)) {
  console.error(`File not found: ${abs}`);
  process.exit(2);
}

const json = JSON.parse(fs.readFileSync(abs, 'utf8'));
const { ok, errors, warnings, normalized } = validateLimitsSchema(json, { platform: args.platform || 'default' });

console.log('=' .repeat(60));
console.log('LIMITS VALIDATION REPORT');
console.log('=' .repeat(60));
console.log('File:', abs);
console.log('Platform:', args.platform || 'default');
console.log('Validation:', ok ? '✅ PASS' : '❌ FAIL');
console.log('-' .repeat(60));

if (errors.length) {
  console.log('\n❌ ERRORS:');
  errors.forEach(e => console.log('  -', e));
}

if (warnings.length) {
  console.log('\n⚠️  WARNINGS:');
  warnings.forEach(w => console.log('  -', w));
}

console.log('\n📊 NORMALIZED VALUES:');
console.log('  Label:', normalized.label || '(none)');
console.log('  maxComputeInvocationsPerWorkgroup:', normalized.maxComputeInvocationsPerWorkgroup);
console.log('  maxComputeWorkgroupSizeX:', normalized.maxComputeWorkgroupSizeX);
console.log('  maxComputeWorkgroupSizeY:', normalized.maxComputeWorkgroupSizeY);
console.log('  maxComputeWorkgroupSizeZ:', normalized.maxComputeWorkgroupSizeZ);
console.log('  maxComputeWorkgroupStorageSize:', normalized.maxComputeWorkgroupStorageSize, 
  normalized.maxComputeWorkgroupStorageSize ? `(${(normalized.maxComputeWorkgroupStorageSize/1024).toFixed(1)} KiB)` : '');
console.log('  maxSampledTexturesPerShaderStage:', normalized.maxSampledTexturesPerShaderStage);
console.log('  maxSamplersPerShaderStage:', normalized.maxSamplersPerShaderStage);

console.log('\n' + '=' .repeat(60));
process.exit(ok ? 0 : 3);
