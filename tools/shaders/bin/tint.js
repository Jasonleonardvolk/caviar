#!/usr/bin/env node
/**
 * tint.js - Wrapper that makes Naga behave like Tint for compatibility
 * This allows scripts expecting Tint to work with Naga
 */

const { execSync } = require('child_process');
const path = require('path');
const fs = require('fs');

const nagaPath = path.join(__dirname, 'naga.exe');

if (!fs.existsSync(nagaPath)) {
  console.error('Error: naga.exe not found');
  process.exit(1);
}

const args = process.argv.slice(2);

// Parse arguments to determine what's being requested
let inputFile = null;
let outputFile = null;
let format = null;

for (let i = 0; i < args.length; i++) {
  const arg = args[i];
  
  if (arg.endsWith('.wgsl')) {
    inputFile = arg;
  } else if (arg === '-o' || arg === '--output') {
    outputFile = args[++i];
  } else if (arg === '--format') {
    format = args[++i];
  } else if (arg.includes('hlsl')) {
    format = 'hlsl';
  } else if (arg.includes('msl') || arg.includes('metal')) {
    format = 'msl';
  } else if (arg.includes('spirv') || arg.includes('spv')) {
    format = 'spirv';
  }
}

if (!inputFile) {
  // If no input file, just pass through to naga
  try {
    execSync(`"${nagaPath}" ${args.join(' ')}`, { stdio: 'inherit' });
  } catch (err) {
    process.exit(err.status || 1);
  }
  process.exit(0);
}

// Determine what to do based on format
try {
  if (!format || format === 'validate') {
    // Just validate
    execSync(`"${nagaPath}" "${inputFile}"`, { stdio: 'inherit' });
  } else if (format === 'spirv') {
    // Convert to SPIR-V
    const output = outputFile || inputFile.replace('.wgsl', '.spv');
    execSync(`"${nagaPath}" "${inputFile}" "${output}"`, { stdio: 'inherit' });
  } else if (format === 'hlsl') {
    // Naga doesn't directly output HLSL via CLI in older versions
    // Just validate for now
    console.log(`Note: Naga validation only (HLSL conversion not available in CLI)`);
    execSync(`"${nagaPath}" "${inputFile}"`, { stdio: 'inherit' });
  } else if (format === 'msl' || format === 'metal') {
    // Naga doesn't directly output MSL via CLI in older versions
    // Just validate for now
    console.log(`Note: Naga validation only (MSL conversion not available in CLI)`);
    execSync(`"${nagaPath}" "${inputFile}"`, { stdio: 'inherit' });
  } else {
    // Unknown format, just validate
    execSync(`"${nagaPath}" "${inputFile}"`, { stdio: 'inherit' });
  }
  process.exit(0);
} catch (err) {
  process.exit(err.status || 1);
}
