#!/usr/bin/env node
// Find missing shader files that aren't in the report

import fs from 'fs';
import path from 'path';

const reportPath = path.join(process.cwd(), 'build', 'shader_report.json');
const shaderDir = path.join(process.cwd(), 'frontend', 'lib', 'webgpu', 'shaders');

// Get all WGSL files in directory
function getAllWgslFiles(dir) {
  const files = [];
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...getAllWgslFiles(fullPath));
    } else if (entry.name.endsWith('.wgsl')) {
      files.push(fullPath);
    }
  }
  return files;
}

const allShaderFiles = getAllWgslFiles(shaderDir);
console.log(`Found ${allShaderFiles.length} .wgsl files in ${shaderDir}\n`);

// Get files from report
const data = JSON.parse(fs.readFileSync(reportPath, 'utf8'));
const reportedFiles = new Set(data.shaders.map(s => {
  // Normalize path for comparison
  return path.resolve(process.cwd(), s.file);
}));

console.log(`Report contains ${reportedFiles.size} files\n`);

// Find missing files
const missingFiles = [];
for (const file of allShaderFiles) {
  if (!reportedFiles.has(file)) {
    missingFiles.push(file);
  }
}

if (missingFiles.length > 0) {
  console.log(`❌ MISSING FROM REPORT (${missingFiles.length} files that likely have syntax errors):\n`);
  for (const file of missingFiles) {
    const relativePath = path.relative(process.cwd(), file);
    console.log(`  ${relativePath}`);
    
    // Try to show first few lines to spot obvious errors
    try {
      const content = fs.readFileSync(file, 'utf8');
      const lines = content.split('\n').slice(0, 5);
      console.log('    First 5 lines:');
      lines.forEach((line, i) => {
        console.log(`      ${i+1}: ${line.slice(0, 80)}${line.length > 80 ? '...' : ''}`);
      });
    } catch (e) {
      console.log('    (Could not read file)');
    }
    console.log();
  }
  
  console.log('⚠️  These files likely have syntax errors preventing validation!');
  console.log('   Check for: missing semicolons, unclosed brackets, invalid syntax, BOM characters, etc.');
} else {
  console.log('✅ All shader files are accounted for in the report.');
  console.log('\nThe "failed" count might be from strict mode treating warnings as failures.');
}