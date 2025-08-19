#!/usr/bin/env node
// Show detailed error/warning report

import fs from 'fs';
import path from 'path';

const reportPath = path.join(process.cwd(), 'build', 'shader_report.json');
const data = JSON.parse(fs.readFileSync(reportPath, 'utf8'));

console.log('=== SHADER VALIDATION DETAILED REPORT ===\n');
console.log(`Total: ${data.shaders.length} files`);
console.log(`Summary claims: ${data.summary.passed} passed, ${data.summary.failed} failed, ${data.summary.warnings} warnings\n`);

// Categorize shaders
const failed = [];
const warned = [];
const passed = [];

for (const shader of data.shaders) {
  const fileName = shader.file.split('\\').pop();
  
  if (shader.errors && shader.errors.length > 0) {
    failed.push({ name: fileName, errors: shader.errors });
  } else if (shader.warnings && shader.warnings.length > 0) {
    warned.push({ name: fileName, warnings: shader.warnings });
  } else {
    passed.push(fileName);
  }
}

// Show failures
if (failed.length > 0) {
  console.log(`\n❌ FAILED (${failed.length} files with errors):`);
  for (const f of failed) {
    console.log(`\n  ${f.name}:`);
    for (const err of f.errors) {
      if (typeof err === 'object') {
        console.log(`    Line ${err.line || '?'}: ${err.message || err.rule || JSON.stringify(err)}`);
      } else {
        console.log(`    ${err}`);
      }
    }
  }
} else {
  console.log('\n❌ FAILED: None (0 files have actual errors in JSON)');
}

// Show warnings (summarized)
if (warned.length > 0) {
  console.log(`\n⚠️  WARNINGS (${warned.length} files):`);
  
  // Group by warning type
  const warningTypes = new Map();
  
  for (const w of warned) {
    console.log(`\n  ${w.name}: ${w.warnings.length} warnings`);
    for (const warning of w.warnings.slice(0, 3)) { // Show first 3
      const rule = warning.rule || 'unknown';
      console.log(`    Line ${warning.line}: ${warning.message}`);
    }
    if (w.warnings.length > 3) {
      console.log(`    ... and ${w.warnings.length - 3} more`);
    }
    
    // Count warning types
    for (const warning of w.warnings) {
      const rule = warning.rule || 'unknown';
      warningTypes.set(rule, (warningTypes.get(rule) || 0) + 1);
    }
  }
  
  console.log('\n📊 Warning Types:');
  for (const [type, count] of [...warningTypes.entries()].sort((a,b) => b[1] - a[1])) {
    console.log(`  ${type}: ${count} occurrences`);
  }
}

// Show passed (just names)
if (passed.length > 0) {
  console.log(`\n✅ PASSED (${passed.length} files):`);
  for (const name of passed) {
    console.log(`  ${name}`);
  }
}

// Check for mismatch
console.log('\n=== ANALYSIS ===');
if (failed.length === 0 && data.summary.failed > 0) {
  console.log(`⚠️  The validator counted ${data.summary.failed} as "failed" but they have no errors in JSON.`);
  console.log('   These might be files that failed to compile/validate but the errors weren\'t captured.');
  console.log('   Or the validator might be counting warnings as failures in strict mode.');
}
