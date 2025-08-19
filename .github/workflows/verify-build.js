#!/usr/bin/env node

/**
 * Verification script to ensure the built TORI Chat UI is not the placeholder/redirect version
 * This can be run as part of CI to prevent accidental deployment of the wrong UI version
 */

const fs = require('fs');
const path = require('path');

// Path to the built index.html
const indexPath = path.join(__dirname, '../../tori_chat_frontend/dist/index.html');

try {
  // Check if the file exists
  if (!fs.existsSync(indexPath)) {
    console.error('\x1b[31mERROR: dist/index.html not found!\x1b[0m');
    console.error('Did you run the build process?');
    process.exit(1);
  }

  // Read the file content
  const content = fs.readFileSync(indexPath, 'utf8');

  // Check for redirect indicators
  if (content.includes('redirected to the') || content.includes('Go to demo now')) {
    console.error('\x1b[31mERROR: Built UI is the redirect placeholder, not the actual TORI Chat interface!\x1b[0m');
    console.error('Check vite.config.js and .env.production to ensure proper build configuration.');
    process.exit(1);
  }

  // Check for React indicators (either React directly or the compiled assets)
  if (!content.includes('React') && !content.includes('/assets/')) {
    console.error('\x1b[31mWARNING: Built UI does not seem to contain React references or asset imports!\x1b[0m');
    console.error('The build may be incomplete or incorrect.');
    process.exit(1);
  }

  console.log('\x1b[32mSUCCESS: Built UI appears to be the correct TORI Chat interface.\x1b[0m');
  process.exit(0);
} catch (error) {
  console.error('\x1b[31mERROR during verification:\x1b[0m', error.message);
  process.exit(1);
}
