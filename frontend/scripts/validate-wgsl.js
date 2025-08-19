/**
 * WGSL Syntax Validator
 * Runs naga on all WGSL files in the project.
 */
import { execSync } from 'child_process';
import { globSync } from 'glob';

const shaderFiles = globSync('hybrid/wgsl/**/*.wgsl', { absolute: true });
let hasError = false;

for (const file of shaderFiles) {
  try {
    execSync(`naga "${file}"`, { stdio: 'pipe' });
    console.log(`✅ WGSL OK: ${file}`);
  } catch (err) {
    console.error(`❌ WGSL ERROR in ${file}`);
    console.error(err.stdout?.toString() || err.message);
    hasError = true;
  }
}

if (hasError) {
  process.exit(1);
}