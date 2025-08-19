#!/usr/bin/env node
// Quick fix to regenerate shader bundle with proper TypeScript syntax

const fs = require('fs');
const path = require('path');

const SHADER_DIR = path.join(__dirname, '../frontend/lib/webgpu/shaders');
const OUTPUT_FILE = path.join(__dirname, '../frontend/lib/webgpu/generated/shaderSources.ts');

console.log('Regenerating shader bundle...');

// Ensure output directory exists
const outputDir = path.dirname(OUTPUT_FILE);
if (!fs.existsSync(outputDir)) {
  fs.mkdirSync(outputDir, { recursive: true });
}

// Read all .wgsl files
let shaderFiles = [];
try {
  shaderFiles = fs.readdirSync(SHADER_DIR)
    .filter(file => file.endsWith('.wgsl'))
    .sort();
  console.log(`Found ${shaderFiles.length} shader files`);
} catch (err) {
  console.error(`Error reading shader directory: ${err.message}`);
  process.exit(1);
}

// Generate the TypeScript file
const timestamp = new Date().toISOString();
let output = `// Auto-generated shader bundle
// Generated: ${timestamp}
// Total shaders: ${shaderFiles.length}

`;

const validShaders = [];

for (const file of shaderFiles) {
  const varName = file.replace('.wgsl', '_wgsl').replace(/[^a-zA-Z0-9_]/g, '_');
  const filepath = path.join(SHADER_DIR, file);
  
  try {
    const content = fs.readFileSync(filepath, 'utf8');
    // Properly escape the content as a JavaScript string
    output += `export const ${varName} = ${JSON.stringify(content)};\n\n`;
    validShaders.push(varName);
    console.log(`  ✅ ${file}`);
  } catch (err) {
    console.error(`  ❌ ${file}: ${err.message}`);
    output += `export const ${varName} = "";\n\n`;
  }
}

// Add the exports object
output += `export const shaderSources = {
${validShaders.map(name => `  ${name}: ${name}`).join(',\n')}
};

export default shaderSources;
`;

// Write the file
fs.writeFileSync(OUTPUT_FILE, output);

console.log(`\n✅ Generated ${OUTPUT_FILE}`);
console.log(`   Size: ${Math.round(output.length / 1024)} KB`);
console.log(`   Valid shaders: ${validShaders.length}/${shaderFiles.length}`);
