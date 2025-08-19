#!/usr/bin/env node
/**
 * Wrapper to make Naga work with our validation pipeline
 * Translates between our expected interface and Naga's CLI
 */

import { execSync, spawn } from 'child_process';
import { existsSync, readdirSync, statSync } from 'fs';
import { join, dirname, extname, relative } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const nagaPath = join(__dirname, '..', 'bin', 'naga.exe');

if (!existsSync(nagaPath)) {
  console.error('❌ naga.exe not found at:', nagaPath);
  console.error('Run: powershell -ExecutionPolicy Bypass -File Install-Naga.ps1');
  process.exit(1);
}

// Parse arguments
const args = process.argv.slice(2);
const options = {
  dir: 'frontend',
  targets: ['spirv', 'hlsl', 'msl'],
  strict: false,
  report: null
};

args.forEach(arg => {
  if (arg.startsWith('--dir=')) {
    options.dir = arg.split('=')[1];
  } else if (arg.startsWith('--targets=')) {
    options.targets = arg.split('=')[1].split(',');
  } else if (arg === '--strict') {
    options.strict = true;
  } else if (arg.startsWith('--report=')) {
    options.report = arg.split('=')[1];
  }
});

console.log('🔍 Validating shaders with Naga...');
console.log(`   Directory: ${options.dir}`);
console.log(`   Targets: ${options.targets.join(', ')}`);

// Find all WGSL files
function* walkDir(dir) {
  const files = readdirSync(dir);
  for (const file of files) {
    const path = join(dir, file);
    const stat = statSync(path);
    if (stat.isDirectory() && !file.includes('node_modules') && !file.includes('.git')) {
      yield* walkDir(path);
    } else if (file.endsWith('.wgsl')) {
      yield path;
    }
  }
}

const results = {
  totalFiles: 0,
  passed: [],
  failed: [],
  warnings: []
};

const rootDir = join(dirname(__dirname), '..', '..', options.dir);
console.log(`   Root: ${rootDir}\n`);

for (const file of walkDir(rootDir)) {
  const relPath = relative(rootDir, file);
  results.totalFiles++;
  
  try {
    // Validate with Naga
    execSync(`"${nagaPath}" "${file}"`, { 
      encoding: 'utf8',
      stdio: 'pipe'
    });
    
    // If we get here, validation passed
    results.passed.push(relPath);
    console.log(`✅ ${relPath}`);
    
  } catch (err) {
    // Parse error output
    const errorOutput = err.stderr || err.stdout || err.toString();
    const errors = [];
    
    // Naga error format parsing
    const lines = errorOutput.split('\n');
    for (const line of lines) {
      if (line.includes('error:') || line.includes('Error')) {
        errors.push({
          message: line.trim(),
          line: 0 // Naga doesn't always give line numbers in the same format
        });
      }
    }
    
    if (errors.length > 0) {
      results.failed.push({
        file: relPath,
        errors: errors
      });
      console.log(`❌ ${relPath}`);
      errors.forEach(e => console.log(`   ${e.message}`));
    } else {
      results.warnings.push({
        file: relPath,
        warnings: [{ message: errorOutput.trim() }]
      });
      console.log(`⚠️  ${relPath}`);
    }
  }
}

// Summary
console.log('\n' + '='.repeat(60));
console.log('VALIDATION SUMMARY:');
console.log(`Total Files: ${results.totalFiles}`);
console.log(`Passed: ${results.passed.length}`);
console.log(`Failed: ${results.failed.length}`);
console.log(`Warnings: ${results.warnings.length}`);

// Write report if requested
if (options.report) {
  const fs = require('fs');
  fs.writeFileSync(options.report, JSON.stringify({
    timestamp: new Date().toISOString(),
    validator: 'naga',
    ...results
  }, null, 2));
  console.log(`\n📄 Report written to: ${options.report}`);
}

// Exit code
if (results.failed.length > 0) {
  process.exit(2);
} else if (results.warnings.length > 0) {
  process.exit(1);
} else {
  process.exit(0);
}
