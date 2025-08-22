#!/usr/bin/env node
/**
 * Verify TailwindCSS build
 */

const fs = require('fs');
const path = require('path');
const { exec } = require('child_process');

console.log('🔍 Verifying TailwindCSS configuration...\n');

// Check if tailwind.config.js exists and is valid
try {
  const configPath = path.join(__dirname, 'tailwind.config.js');
  require(configPath);
  console.log('✅ tailwind.config.js is valid');
} catch (error) {
  console.error('❌ Error in tailwind.config.js:', error.message);
  process.exit(1);
}

// Check app.css for any remaining issues
const appCssPath = path.join(__dirname, 'src', 'app.css');
const appCss = fs.readFileSync(appCssPath, 'utf-8');

// Check for circular @apply references
const applyPattern = /@apply\s+tori-button(?:\s|;)/g;
const matches = appCss.match(applyPattern);

if (matches) {
  console.error('❌ Found circular @apply references:');
  matches.forEach(match => console.error(`   ${match}`));
} else {
  console.log('✅ No circular @apply references found');
}

// Try to build CSS
console.log('\n🔨 Testing CSS build...');

exec('npx tailwindcss -i ./src/app.css -o ./test-output.css', (error, stdout, stderr) => {
  if (error) {
    console.error('❌ CSS build failed:', error.message);
    if (stderr) console.error('Error details:', stderr);
  } else {
    console.log('✅ CSS build successful!');
    
    // Check if tori-button is in the output
    const output = fs.readFileSync('./test-output.css', 'utf-8');
    if (output.includes('.tori-button')) {
      console.log('✅ .tori-button class found in output');
    } else {
      console.log('⚠️  .tori-button class not found in output');
    }
    
    // Clean up test file
    fs.unlinkSync('./test-output.css');
  }
  
  console.log('\n✨ Verification complete!');
});
