#!/usr/bin/env node
/**
 * Analyze TypeScript errors and create detailed report
 * Specifically focuses on shader-related errors
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { execSync } from 'child_process';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const REPO_ROOT = path.resolve(__dirname, '../..');
const ERROR_LOGS_DIR = path.join(REPO_ROOT, 'tools/release/error_logs');

// Create timestamp
const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
const errorDir = path.join(ERROR_LOGS_DIR, timestamp);

// Ensure error directory exists
if (!fs.existsSync(errorDir)) {
  fs.mkdirSync(errorDir, { recursive: true });
}

console.log('========================================');
console.log(' SHADER BUNDLE ERROR ANALYSIS');
console.log('========================================');
console.log('');
console.log(`Logging to: ${errorDir}`);
console.log('');

// Step 1: Check current state of shaderSources.ts
console.log('Step 1: Analyzing current shaderSources.ts...');
const shaderSourcesPath = path.join(REPO_ROOT, 'frontend/lib/webgpu/generated/shaderSources.ts');

if (fs.existsSync(shaderSourcesPath)) {
  const content = fs.readFileSync(shaderSourcesPath, 'utf8');
  const lines = content.split('\n');
  
  // Check for common issues
  const issues = [];
  
  // Check if exports are properly quoted
  const unquotedExports = lines.filter(line => 
    line.startsWith('export const') && 
    line.includes('_wgsl = ') &&
    !line.includes('_wgsl = "') &&
    !line.includes('_wgsl = `') &&
    !line.includes('_wgsl = \'')
  );
  
  if (unquotedExports.length > 0) {
    issues.push({
      type: 'UNQUOTED_EXPORTS',
      count: unquotedExports.length,
      examples: unquotedExports.slice(0, 3),
      fix: 'Shader content must be wrapped in quotes or backticks'
    });
  }
  
  // Check file metadata
  const metadata = {
    totalLines: lines.length,
    fileSize: Math.round(content.length / 1024) + ' KB',
    generatedDate: lines.find(l => l.includes('Generated:'))?.match(/Generated:\s*(.+)/)?.[1] || 'Unknown',
    exports: lines.filter(l => l.startsWith('export const')).length,
    issues: issues
  };
  
  // Save analysis
  fs.writeFileSync(
    path.join(errorDir, 'shader_sources_analysis.json'),
    JSON.stringify(metadata, null, 2)
  );
  
  console.log(`  File size: ${metadata.fileSize}`);
  console.log(`  Generated: ${metadata.generatedDate}`);
  console.log(`  Total exports: ${metadata.exports}`);
  
  if (issues.length > 0) {
    console.log(`  ⚠️  Issues found: ${issues.length}`);
    issues.forEach(issue => {
      console.log(`    - ${issue.type}: ${issue.count} occurrences`);
      console.log(`      Fix: ${issue.fix}`);
    });
  }
} else {
  console.log('  ❌ shaderSources.ts not found!');
}

console.log('');

// Step 2: Run TypeScript compiler and capture errors
console.log('Step 2: Running TypeScript compiler...');
const tsErrorsPath = path.join(errorDir, 'typescript_raw.txt');

try {
  execSync(`npx tsc -p ${REPO_ROOT}/frontend/tsconfig.json --noEmit`, {
    cwd: REPO_ROOT,
    stdio: 'pipe'
  });
  console.log('  ✅ No TypeScript errors!');
} catch (error) {
  // TypeScript failed, capture and analyze errors
  fs.writeFileSync(tsErrorsPath, error.stdout?.toString() || error.stderr?.toString() || 'Unknown error');
  
  const errorOutput = error.stdout?.toString() || error.stderr?.toString() || '';
  const errorLines = errorOutput.split('\n');
  
  // Parse errors
  const errors = [];
  const shaderErrors = [];
  
  errorLines.forEach(line => {
    const match = line.match(/(.+?)\((\d+),(\d+)\):\s+error\s+(TS\d+):\s+(.+)/);
    if (match) {
      const [, file, line, col, code, message] = match;
      const error = { file, line: parseInt(line), col: parseInt(col), code, message };
      errors.push(error);
      
      // Check if it's shader-related
      if (file.includes('shaderSources') || file.includes('shader')) {
        shaderErrors.push(error);
      }
    }
  });
  
  // Create detailed report
  const report = {
    timestamp,
    totalErrors: errors.length,
    shaderRelatedErrors: shaderErrors.length,
    errorsByFile: {},
    errorsBySeverity: {},
    suggestedFixes: []
  };
  
  // Group errors by file
  errors.forEach(error => {
    if (!report.errorsByFile[error.file]) {
      report.errorsByFile[error.file] = [];
    }
    report.errorsByFile[error.file].push(error);
  });
  
  // Analyze shader-specific errors
  if (shaderErrors.length > 0) {
    console.log(`  ❌ Found ${shaderErrors.length} shader-related errors`);
    
    // Check for specific patterns
    const syntaxErrors = shaderErrors.filter(e => e.code === 'TS1005' || e.code === 'TS1002');
    if (syntaxErrors.length > 0) {
      report.suggestedFixes.push({
        issue: 'Syntax errors in shader exports',
        fix: 'Run: node scripts/bundleShaders.mjs',
        command: 'node scripts/bundleShaders.mjs'
      });
    }
  }
  
  // Save detailed report
  fs.writeFileSync(
    path.join(errorDir, 'error_report.json'),
    JSON.stringify(report, null, 2)
  );
  
  // Create human-readable summary
  const summary = `
TypeScript Error Summary
========================
Timestamp: ${timestamp}
Total Errors: ${errors.length}
Shader-Related Errors: ${shaderErrors.length}

Files with Most Errors:
${Object.entries(report.errorsByFile)
  .sort((a, b) => b[1].length - a[1].length)
  .slice(0, 5)
  .map(([file, errs]) => `  - ${path.relative(REPO_ROOT, file)}: ${errs.length} errors`)
  .join('\n')}

${report.suggestedFixes.length > 0 ? `
Suggested Fixes:
${report.suggestedFixes.map(fix => `  1. ${fix.issue}\n     Fix: ${fix.fix}`).join('\n')}
` : ''}

Full error log: ${tsErrorsPath}
Detailed report: ${path.join(errorDir, 'error_report.json')}
`;

  fs.writeFileSync(path.join(errorDir, 'SUMMARY.txt'), summary);
  
  console.log(summary);
}

// Step 3: If errors exist, attempt automatic fix
if (fs.existsSync(tsErrorsPath)) {
  console.log('');
  console.log('Step 3: Attempting automatic fix...');
  console.log('  Running: node scripts/bundleShaders.mjs');
  
  try {
    const output = execSync('node scripts/bundleShaders.mjs', {
      cwd: REPO_ROOT,
      encoding: 'utf8'
    });
    
    fs.writeFileSync(path.join(errorDir, 'bundler_output.log'), output);
    console.log('  ✅ Shader bundle regenerated');
    
    // Test again
    console.log('  Testing TypeScript again...');
    try {
      execSync(`npx tsc -p ${REPO_ROOT}/frontend/tsconfig.json --noEmit`, {
        cwd: REPO_ROOT,
        stdio: 'pipe'
      });
      console.log('  ✅ TypeScript errors FIXED!');
      
      fs.writeFileSync(path.join(errorDir, 'FIX_SUCCESS.txt'), `
Automatic Fix Successful!
========================
The shader bundle was regenerated and TypeScript errors are resolved.

You can now run the full release gate:
  ./tools/release/IrisOneButton.ps1
`);
      
    } catch (e) {
      console.log('  ⚠️  Some TypeScript errors remain after fix');
      const remaining = e.stdout?.toString() || e.stderr?.toString() || '';
      fs.writeFileSync(path.join(errorDir, 'remaining_errors.txt'), remaining);
    }
    
  } catch (error) {
    console.log('  ❌ Failed to regenerate shader bundle');
    fs.writeFileSync(path.join(errorDir, 'bundler_error.log'), error.toString());
  }
}

console.log('');
console.log(`All logs saved to: ${errorDir}`);
