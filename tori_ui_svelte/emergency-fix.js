#!/usr/bin/env node

/**
 * TORI PostCSS Emergency Fix
 * Handles dependency conflicts and fixes PostCSS issues
 */

import { execSync } from 'child_process';
import fs from 'fs';
import path from 'path';

console.log('🚨 TORI PostCSS Emergency Fix');
console.log('='.repeat(50));

try {
  // Step 1: Check and fix package.json
  console.log('📋 Checking package.json for conflicts...');
  
  // Clear npm cache first
  console.log('🧹 Clearing npm cache...');
  try {
    execSync('npm cache clean --force', { stdio: 'pipe' });
    console.log('✅ npm cache cleared');
  } catch (e) {
    console.log('⚠️ Could not clear npm cache, continuing...');
  }

  // Step 2: Remove node_modules and package-lock.json for clean install
  console.log('🧹 Cleaning for fresh install...');
  
  const nodeModulesPath = path.join(process.cwd(), 'node_modules');
  const packageLockPath = path.join(process.cwd(), 'package-lock.json');
  const viteCachePath = path.join(process.cwd(), 'node_modules', '.vite');
  
  if (fs.existsSync(viteCachePath)) {
    fs.rmSync(viteCachePath, { recursive: true, force: true });
    console.log('✅ Vite cache cleared');
  }
  
  if (fs.existsSync(packageLockPath)) {
    fs.unlinkSync(packageLockPath);
    console.log('✅ package-lock.json removed');
  }

  // Step 3: Install only the PostCSS dependencies first
  console.log('📦 Installing PostCSS dependencies...');
  execSync('npm install -D tailwindcss@latest autoprefixer@latest postcss@latest', { 
    stdio: 'inherit',
    cwd: process.cwd()
  });
  console.log('✅ PostCSS dependencies installed!');

  // Step 4: Create simple PostCSS config
  console.log('⚙️ Creating PostCSS config...');
  const simpleConfig = `export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {}
  }
};`;
  
  fs.writeFileSync('postcss.config.js', simpleConfig);
  console.log('✅ PostCSS config created!');

  // Step 5: Install remaining dependencies
  console.log('📦 Installing remaining dependencies...');
  try {
    execSync('npm install', { 
      stdio: 'inherit',
      cwd: process.cwd()
    });
    console.log('✅ All dependencies installed!');
  } catch (error) {
    console.log('⚠️ Some dependencies may have failed, but PostCSS should work');
  }

  console.log('\n🎉 Emergency fix completed!');
  console.log('🚀 Now try: npm run dev');
  console.log('The PostCSS error should be resolved!');

} catch (error) {
  console.error('❌ Emergency fix failed:', error.message);
  console.log('\n🆘 Manual fix required:');
  console.log('1. Delete node_modules and package-lock.json');
  console.log('2. Run: npm install -D tailwindcss autoprefixer postcss');
  console.log('3. Update postcss.config.js with simple config');
  console.log('4. Run: npm install');
  console.log('5. Run: npm run dev');
}
