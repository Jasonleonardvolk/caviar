/**
 * Touch WGSL file and validate.
 * Used for testing validation pipeline.
 */
import fs from 'fs';
import { execSync } from 'child_process';
import path from 'path';

const file = process.argv[2];
if (!file) {
  console.error("Usage: node touch_wgsl_and_validate.js <path/to/shader.wgsl>");
  process.exit(2);
}

const macro = "/*CHAOS_TOGGLE*/";

try {
  // Read current content
  let content = fs.readFileSync(file, 'utf-8');
  
  // Toggle macro
  if (content.includes(macro)) {
    content = content.replace(macro, '');
    console.log(`Removed macro from ${path.basename(file)}`);
  } else {
    content = content + '\n' + macro + '\n';
    console.log(`Added macro to ${path.basename(file)}`);
  }
  
  // Write back
  fs.writeFileSync(file, content);
  
  // Try to validate
  try {
    execSync(`naga "${file}"`, { stdio: 'pipe' });
    console.log(`✅ Validation PASSED for ${path.basename(file)}`);
  } catch (error) {
    console.error(`❌ Validation FAILED for ${path.basename(file)}`);
    if (error.stdout) console.error(error.stdout.toString());
    if (error.stderr) console.error(error.stderr.toString());
    process.exit(1);
  }
  
} catch (error) {
  console.error(`Error processing file: ${error.message}`);
  process.exit(1);
}
