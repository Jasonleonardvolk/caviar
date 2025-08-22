#!/usr/bin/env node

/**
 * Quick PostCSS Fix - Minimal approach
 * This script just does the essential fixes without complex error handling
 */

import { execSync } from 'child_process';
import fs from 'fs';
import path from 'path';

console.log('🔧 Quick PostCSS Fix Starting...');
console.log('='.repeat(50));

try {
  // Step 1: Install the missing dependencies
  console.log('📦 Installing dependencies...');
  execSync('npm install -D tailwindcss autoprefixer postcss', { 
    stdio: 'inherit',
    cwd: process.cwd()
  });
  console.log('✅ Dependencies installed!');

  // Step 2: Clear Vite cache
  console.log('🧹 Clearing cache...');
  const viteCachePath = path.join(process.cwd(), 'node_modules', '.vite');
  if (fs.existsSync(viteCachePath)) {
    fs.rmSync(viteCachePath, { recursive: true, force: true });
    console.log('✅ Cache cleared!');
  }

  // Step 3: Simple PostCSS config
  console.log('⚙️ Creating simple PostCSS config...');
  const simpleConfig = `export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {}
  }
};`;
  
  fs.writeFileSync('postcss.config.js', simpleConfig);
  console.log('✅ PostCSS config updated!');

  console.log('\n🎉 Fix completed!');
  console.log('Now run: npm run dev');

} catch (error) {
  console.error('❌ Error:', error.message);
  console.log('Try running the commands manually:');
  console.log('  npm install -D tailwindcss autoprefixer postcss');
  console.log('  npm run dev');
}
