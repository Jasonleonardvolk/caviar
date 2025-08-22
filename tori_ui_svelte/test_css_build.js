#!/usr/bin/env node
// Quick test script to build CSS and check for errors

import { exec } from 'child_process';
import { promises as fs } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

console.log('🔨 Testing TailwindCSS build...\n');

// Run the CSS build
exec('npx tailwindcss -i ./src/app.css -o ./test-build.css', { cwd: __dirname }, async (error, stdout, stderr) => {
  if (error) {
    console.error('❌ Build failed with error:', error.message);
    if (stderr) {
      console.error('\nError details:', stderr);
    }
    process.exit(1);
  }
  
  console.log('✅ CSS build completed successfully!');
  
  try {
    // Check if the output contains our button class
    const output = await fs.readFile(join(__dirname, 'test-build.css'), 'utf-8');
    
    if (output.includes('.tori-button')) {
      console.log('✅ .tori-button class found in output');
    } else {
      console.log('⚠️  .tori-button class NOT found in output - checking why...');
    }
    
    // Clean up
    await fs.unlink(join(__dirname, 'test-build.css'));
    
    console.log('\n✨ Test complete! Now you can run: npm run dev');
  } catch (err) {
    console.error('Error checking output:', err);
  }
});
