#!/usr/bin/env node
// Check for any actual errors in the JSON

import fs from 'fs';
import path from 'path';

const reportPath = path.join(process.cwd(), 'build', 'shader_report.json');
const data = JSON.parse(fs.readFileSync(reportPath, 'utf8'));

console.log('=== CHECKING FOR ERRORS ===\n');
console.log('Summary says:');
console.log('  Total:', data.summary.total);
console.log('  Passed:', data.summary.passed);
console.log('  Failed:', data.summary.failed);
console.log('  Warnings:', data.summary.warnings);

let actualPassed = 0;
let actualFailed = 0;
let actualWarnings = 0;

console.log('\n=== ACTUAL COUNTS ===\n');

for (const shader of data.shaders) {
  if (shader.errors && shader.errors.length > 0) {
    actualFailed++;
    console.log(`ERROR in ${shader.file}:`, shader.errors.length, 'errors');
  } else if (shader.warnings && shader.warnings.length > 0) {
    actualWarnings++;
  } else {
    actualPassed++;
  }
}

console.log('\nActual counts from shaders array:');
console.log('  Passed (no errors/warnings):', actualPassed);
console.log('  Failed (has errors):', actualFailed);
console.log('  Has warnings (no errors):', actualWarnings);
console.log('  Total shaders in array:', data.shaders.length);

// Check if there's a mismatch
if (actualFailed === 0 && data.summary.failed > 0) {
  console.log('\n⚠️  MISMATCH: Summary says', data.summary.failed, 'failed but no shaders have errors!');
  console.log('This might mean the validator is counting compilation failures that aren\'t in the JSON.');
}
