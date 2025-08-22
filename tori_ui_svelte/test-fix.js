#!/usr/bin/env node

/**
 * Test PostCSS Fix
 * Quick test to see if PostCSS error is resolved
 */

import fs from 'fs';
import path from 'path';

console.log('🔍 Testing PostCSS Fix...');
console.log('='.repeat(40));

// Check if we're in the right directory
const currentDir = process.cwd();
console.log(`Current directory: ${currentDir}`);

const packageJsonPath = path.join(currentDir, 'package.json');
const postcssConfigPath = path.join(currentDir, 'postcss.config.js');

if (!fs.existsSync(packageJsonPath)) {
  console.log('❌ package.json not found');
  console.log('Make sure you are in the tori_ui_svelte directory');
  process.exit(1);
}

// Check if PostCSS dependencies are installed
try {
  const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
  const allDeps = { ...packageJson.dependencies, ...packageJson.devDependencies };
  
  const required = ['tailwindcss', 'autoprefixer', 'postcss'];
  const missing = required.filter(dep => !allDeps[dep]);
  
  if (missing.length === 0) {
    console.log('✅ All PostCSS dependencies found!');
    required.forEach(dep => {
      console.log(`  ✓ ${dep}: ${allDeps[dep]}`);
    });
  } else {
    console.log('❌ Missing dependencies:', missing.join(', '));
  }
  
} catch (error) {
  console.log('❌ Error reading package.json:', error.message);
}

// Check PostCSS config
if (fs.existsSync(postcssConfigPath)) {
  console.log('✅ PostCSS config exists');
} else {
  console.log('⚠️ Creating PostCSS config...');
  const config = `export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {}
  }
};`;
  fs.writeFileSync(postcssConfigPath, config);
  console.log('✅ PostCSS config created');
}

console.log('\n🚀 Ready to test! Run: npm run dev');
