// update_package_scripts.mjs
// Adds all shader validation scripts to package.json

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const packagePath = path.join(__dirname, '../../package.json');

// Read current package.json
const packageJson = JSON.parse(fs.readFileSync(packagePath, 'utf8'));

// Ensure scripts object exists
if (!packageJson.scripts) {
  packageJson.scripts = {};
}

// Define all shader-related scripts
const shaderScripts = {
  // Core validation
  "shaders:validate": "node tools/shaders/validate_and_report.mjs --dir=frontend/lib/webgpu/shaders",
  "shaders:gate": "node tools/shaders/shader_quality_gate_v2.mjs --strict",
  "shaders:gate:iphone": "node tools/shaders/shader_quality_gate_v2.mjs --dir=frontend/lib/webgpu/shaders --limits=tools/shaders/device_limits/iphone15.json --targets=naga --strict",
  
  // Enhanced validation with suppression
  "shaders:validate:smart": "node tools/shaders/shader_quality_gate_v2_enhanced.mjs --dir=frontend/lib/webgpu/shaders --suppressions=tools/shaders/validator_suppressions.json",
  "shaders:gate:smart": "node tools/shaders/shader_quality_gate_v2_enhanced.mjs --dir=frontend/lib/webgpu/shaders --suppressions=tools/shaders/validator_suppressions.json --strict",
  
  // Sync and maintenance
  "shaders:sync": "node tools/shaders/copy_canonical_to_public.mjs",
  "shaders:canonical": "node tools/shaders/promote_to_canonical.mjs",
  "shaders:dedupe": "node tools/shaders/dedupe_shaders_v2.mjs",
  
  // Guards and checks
  "shaders:guard": "node tools/shaders/guards/check_uniform_arrays.mjs",
  "shaders:guard:vec3": "node tools/shaders/guards/verify_no_storage_vec3.mjs",
  "shaders:guard:all": "npm run shaders:guard && npm run shaders:guard:vec3",
  
  // Testing and verification
  "shaders:test": "node tools/shaders/test_suppression_system.mjs",
  "shaders:report": "node tools/shaders/show_report.mjs",
  
  // Meta-tools
  "shaders:summon": "node tools/shaders/virgil_summon.mjs",
  
  // Comprehensive workflows
  "shaders:check": "npm run shaders:guard:all && npm run shaders:validate:smart",
  "shaders:ci": "npm run shaders:sync && npm run shaders:guard:all && npm run shaders:gate:smart",
  "shaders:fix": "npm run shaders:sync && npm run shaders:test"
};

// Add or update scripts
let updated = 0;
let added = 0;

for (const [name, command] of Object.entries(shaderScripts)) {
  if (packageJson.scripts[name]) {
    if (packageJson.scripts[name] !== command) {
      console.log(`📝 Updating: ${name}`);
      packageJson.scripts[name] = command;
      updated++;
    }
  } else {
    console.log(`➕ Adding: ${name}`);
    packageJson.scripts[name] = command;
    added++;
  }
}

// Sort scripts alphabetically (optional)
const sortedScripts = {};
Object.keys(packageJson.scripts).sort().forEach(key => {
  sortedScripts[key] = packageJson.scripts[key];
});
packageJson.scripts = sortedScripts;

// Write back to package.json
fs.writeFileSync(packagePath, JSON.stringify(packageJson, null, 2) + '\n');

console.log(`
╔════════════════════════════════════════════════════════════╗
║              PACKAGE.JSON UPDATED                          ║
╚════════════════════════════════════════════════════════════╝

📊 Summary:
  • Scripts added: ${added}
  • Scripts updated: ${updated}
  • Total shader scripts: ${Object.keys(shaderScripts).length}

🚀 Key Commands:

  Quick Check:
    npm run shaders:check      # Guards + smart validation
    
  Full CI Pipeline:
    npm run shaders:ci         # Sync + guards + strict gate
    
  Test Suppression:
    npm run shaders:test       # Verify suppression works
    
  Smart Validation (with suppression):
    npm run shaders:validate:smart
    npm run shaders:gate:smart
    
  Legacy Validation (all warnings):
    npm run shaders:validate
    npm run shaders:gate
`);
