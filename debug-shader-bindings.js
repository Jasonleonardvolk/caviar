#!/usr/bin/env node

/**
 * Debug applyPhaseLUT.wgsl binding issue
 * Helps identify why duplicate binding error might be reported
 */

const fs = require('fs');
const path = require('path');

console.log('🔍 Debugging applyPhaseLUT.wgsl Binding Issue\n');
console.log('=============================================\n');

const shaderDir = path.join(__dirname, 'frontend', 'lib', 'webgpu', 'shaders');

// Find all WGSL files with @group(0) bindings
console.log('📂 Scanning all shaders for @group(0) bindings...\n');

function scanFile(filePath, relativePath) {
  const content = fs.readFileSync(filePath, 'utf8');
  const bindingPattern = /@group\((\d+)\)\s*@binding\((\d+)\)/g;
  const bindings = [];
  let match;
  
  while ((match = bindingPattern.exec(content)) !== null) {
    bindings.push({
      group: parseInt(match[1]),
      binding: parseInt(match[2]),
      line: content.substring(0, match.index).split('\n').length
    });
  }
  
  if (bindings.length > 0) {
    console.log(`📄 ${relativePath}:`);
    const group0Bindings = bindings.filter(b => b.group === 0);
    
    if (group0Bindings.length > 0) {
      console.log(`   Group 0 bindings: ${group0Bindings.map(b => b.binding).join(', ')}`);
      
      // Check for duplicates within the file
      const bindingNumbers = group0Bindings.map(b => b.binding);
      const duplicates = bindingNumbers.filter((b, i) => bindingNumbers.indexOf(b) !== i);
      
      if (duplicates.length > 0) {
        console.log(`   ⚠️  DUPLICATE BINDINGS FOUND: ${duplicates.join(', ')}`);
      }
    }
    
    // Show all groups
    const otherGroups = [...new Set(bindings.map(b => b.group))].filter(g => g !== 0);
    if (otherGroups.length > 0) {
      console.log(`   Other groups: ${otherGroups.join(', ')}`);
    }
    console.log('');
  }
}

function scanDirectory(dir, baseDir = dir) {
  const files = fs.readdirSync(dir, { withFileTypes: true });
  
  for (const file of files) {
    const fullPath = path.join(dir, file.name);
    
    if (file.isDirectory()) {
      scanDirectory(fullPath, baseDir);
    } else if (file.name.endsWith('.wgsl')) {
      const relativePath = path.relative(baseDir, fullPath);
      scanFile(fullPath, relativePath);
    }
  }
}

scanDirectory(shaderDir);

// Specifically check applyPhaseLUT.wgsl
console.log('🔎 Detailed check of applyPhaseLUT.wgsl:\n');

const lutFile = path.join(shaderDir, 'post', 'applyPhaseLUT.wgsl');
if (fs.existsSync(lutFile)) {
  const content = fs.readFileSync(lutFile, 'utf8');
  const lines = content.split('\n');
  
  console.log('Line-by-line binding declarations:');
  lines.forEach((line, i) => {
    if (line.includes('@group') && line.includes('@binding')) {
      console.log(`   Line ${i + 1}: ${line.trim()}`);
    }
  });
  
  // Check for any texture or sampler declarations that might be missing bindings
  console.log('\nTexture/Sampler usage:');
  let hasTextures = false;
  lines.forEach((line, i) => {
    if (line.includes('texture_') || line.includes('sampler')) {
      console.log(`   Line ${i + 1}: ${line.trim()}`);
      hasTextures = true;
    }
  });
  
  if (!hasTextures) {
    console.log('   No texture or sampler declarations found');
  }
  
  // Check if there are any included files
  console.log('\nIncludes/Imports:');
  let hasIncludes = false;
  lines.forEach((line, i) => {
    if (line.includes('#include') || line.includes('import')) {
      console.log(`   Line ${i + 1}: ${line.trim()}`);
      hasIncludes = true;
    }
  });
  
  if (!hasIncludes) {
    console.log('   No includes or imports found');
  }
}

console.log('\n=============================================');
console.log('💡 Possible issues to check:');
console.log('1. Multiple shaders being compiled together with conflicting bindings');
console.log('2. Validator checking against a different version of the file');
console.log('3. Hidden characters or encoding issues in the file');
console.log('4. False positive from the validator\n');
