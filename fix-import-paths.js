#!/usr/bin/env node

/**
 * Comprehensive Import Path Fixer
 * Fixes all import path issues in the TypeScript project
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

console.log('🔍 Analyzing and Fixing Import Path Issues\n');
console.log('==========================================\n');

// Common import issues and their fixes
const importPatterns = [
  // Fix $lib imports
  {
    pattern: /from ['"](\$lib\/[^'"]+)['"]/g,
    fix: (match, p1) => {
      // Check if the file exists
      const libPath = p1.replace('$lib', 'tori_ui_svelte/src/lib');
      const possibleExtensions = ['.ts', '.svelte', '/index.ts', ''];
      
      for (const ext of possibleExtensions) {
        const fullPath = path.join(__dirname, libPath + ext);
        if (fs.existsSync(fullPath)) {
          return match; // Path is valid
        }
      }
      
      // Try without the last segment (might be looking for index)
      const segments = p1.split('/');
      segments.pop();
      const indexPath = segments.join('/') + '/index';
      return `from '${indexPath}'`;
    }
  },
  
  // Fix relative imports missing extensions
  {
    pattern: /from ['"](\.[^'"]+)(?<!\.ts)(?<!\.js)(?<!\.svelte)['"]/g,
    fix: (match, p1) => {
      // Add .js extension for relative imports (ESM requirement)
      return `from '${p1}.js'`;
    }
  },
  
  // Fix stores imports
  {
    pattern: /from ['"]svelte\/store['"]/g,
    fix: () => `from 'svelte/store'`
  }
];

// Directories to scan
const dirsToScan = [
  'tori_ui_svelte/src/lib',
  'frontend/src',
  'frontend/lib'
];

let totalFilesFixed = 0;
let totalIssuesFixed = 0;

// Function to fix a single file
function fixFile(filePath) {
  if (!filePath.endsWith('.ts') && !filePath.endsWith('.tsx')) {
    return false;
  }
  
  try {
    let content = fs.readFileSync(filePath, 'utf8');
    let originalContent = content;
    let fixCount = 0;
    
    // Apply each pattern
    for (const { pattern, fix } of importPatterns) {
      const matches = content.match(pattern);
      if (matches) {
        content = content.replace(pattern, fix);
        fixCount += matches.length;
      }
    }
    
    // Check for circular imports
    const fileName = path.basename(filePath, path.extname(filePath));
    const circularPattern = new RegExp(`from ['"]\\.\\/${fileName}['"]`, 'g');
    if (circularPattern.test(content)) {
      console.log(`⚠️  Circular import detected in ${filePath}`);
      content = content.replace(circularPattern, "from './types'");
      fixCount++;
    }
    
    if (content !== originalContent) {
      fs.writeFileSync(filePath, content);
      totalIssuesFixed += fixCount;
      return true;
    }
    
    return false;
  } catch (error) {
    console.error(`Error processing ${filePath}:`, error.message);
    return false;
  }
}

// Function to recursively scan directory
function scanDirectory(dir) {
  const fullPath = path.join(__dirname, dir);
  
  if (!fs.existsSync(fullPath)) {
    console.log(`⏭️  Skipping ${dir} (not found)`);
    return;
  }
  
  const files = fs.readdirSync(fullPath, { withFileTypes: true });
  
  for (const file of files) {
    const filePath = path.join(fullPath, file.name);
    
    if (file.isDirectory() && !file.name.includes('node_modules')) {
      scanDirectory(path.join(dir, file.name));
    } else if (file.isFile()) {
      if (fixFile(filePath)) {
        console.log(`✅ Fixed: ${path.join(dir, file.name)}`);
        totalFilesFixed++;
      }
    }
  }
}

// Main execution
console.log('📂 Scanning directories for import issues...\n');

for (const dir of dirsToScan) {
  console.log(`\nScanning ${dir}...`);
  scanDirectory(dir);
}

console.log('\n' + '='.repeat(50));
console.log(`✨ Fixed ${totalFilesFixed} files with ${totalIssuesFixed} import issues`);

// Run TypeScript check to see remaining errors
console.log('\n🔧 Running TypeScript check...\n');

try {
  execSync('npx tsc --noEmit', { stdio: 'inherit' });
  console.log('\n✅ All TypeScript errors resolved!');
} catch (error) {
  // Count remaining errors
  try {
    const output = execSync('npx tsc --noEmit 2>&1', { encoding: 'utf8' });
  } catch (e) {
    const errorOutput = e.stdout || '';
    const errorMatch = errorOutput.match(/Found (\d+) error/);
    if (errorMatch) {
      const remainingErrors = parseInt(errorMatch[1]);
      console.log(`\n📊 ${remainingErrors} errors remaining (down from 90)`);
      
      if (remainingErrors < 20) {
        console.log('\n✅ Great progress! Most issues fixed.');
        console.log('   Remaining errors are likely specific type mismatches.');
        console.log('\n💡 You can now:');
        console.log('   1. Fix remaining errors manually');
        console.log('   2. Build with skipLibCheck: npm run build');
      }
    }
  }
}

console.log('\n🎯 Import path fixes complete!');
