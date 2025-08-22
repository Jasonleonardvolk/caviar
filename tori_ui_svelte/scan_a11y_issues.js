#!/usr/bin/env node
/**
 * Scan for common accessibility issues in Svelte components
 */

import { promises as fs } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { glob } from 'glob';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

async function scanForA11yIssues() {
  console.log('🔍 Scanning for Accessibility Issues in Svelte Components...\n');

  // Find all Svelte files
  const files = await glob('src/**/*.svelte', { cwd: __dirname });
  
  const issues = {
    clickableDivs: [],
    unlabeledInputs: [],
    missingAltText: [],
    videosWithoutCaptions: [],
    missingKeyboardHandlers: []
  };

  for (const file of files) {
    const content = await fs.readFile(join(__dirname, file), 'utf-8');
    const lines = content.split('\n');
    
    lines.forEach((line, index) => {
      const lineNum = index + 1;
      
      // Check for clickable divs without role
      if (line.includes('<div') && line.includes('on:click') && !line.includes('role=')) {
        issues.clickableDivs.push({ file, line: lineNum, code: line.trim() });
      }
      
      // Check for inputs without labels
      if (line.includes('<input') && !line.includes('aria-label')) {
        // Check if there's a label nearby (simple heuristic)
        const prevLine = lines[index - 1] || '';
        const nextLine = lines[index + 1] || '';
        if (!prevLine.includes('<label') && !nextLine.includes('</label>')) {
          issues.unlabeledInputs.push({ file, line: lineNum, code: line.trim() });
        }
      }
      
      // Check for images without alt
      if (line.includes('<img') && !line.includes('alt=')) {
        issues.missingAltText.push({ file, line: lineNum, code: line.trim() });
      }
      
      // Check for videos without tracks
      if (line.includes('<video')) {
        // Look ahead for track element
        let hasTrack = false;
        for (let i = index; i < Math.min(index + 10, lines.length); i++) {
          if (lines[i].includes('</video>')) break;
          if (lines[i].includes('<track')) {
            hasTrack = true;
            break;
          }
        }
        if (!hasTrack) {
          issues.videosWithoutCaptions.push({ file, line: lineNum, code: line.trim() });
        }
      }
      
      // Check for click without keyboard handler
      if (line.includes('on:click') && !line.includes('on:key')) {
        // Check nearby lines for keyboard handlers
        const contextStart = Math.max(0, index - 3);
        const contextEnd = Math.min(lines.length, index + 3);
        const context = lines.slice(contextStart, contextEnd).join('\n');
        if (!context.includes('on:key')) {
          issues.missingKeyboardHandlers.push({ file, line: lineNum, code: line.trim() });
        }
      }
    });
  }

  // Report findings
  console.log('📊 Accessibility Scan Results:\n');
  
  if (issues.clickableDivs.length > 0) {
    console.log(`❌ Clickable <div> elements without proper role (${issues.clickableDivs.length} found):`);
    issues.clickableDivs.slice(0, 5).forEach(issue => {
      console.log(`   ${issue.file}:${issue.line}`);
    });
    if (issues.clickableDivs.length > 5) {
      console.log(`   ... and ${issues.clickableDivs.length - 5} more\n`);
    }
  }
  
  if (issues.unlabeledInputs.length > 0) {
    console.log(`\n❌ Input elements without labels (${issues.unlabeledInputs.length} found):`);
    issues.unlabeledInputs.slice(0, 5).forEach(issue => {
      console.log(`   ${issue.file}:${issue.line}`);
    });
  }
  
  if (issues.missingAltText.length > 0) {
    console.log(`\n❌ Images without alt text (${issues.missingAltText.length} found):`);
    issues.missingAltText.slice(0, 5).forEach(issue => {
      console.log(`   ${issue.file}:${issue.line}`);
    });
  }
  
  if (issues.videosWithoutCaptions.length > 0) {
    console.log(`\n❌ Videos without caption tracks (${issues.videosWithoutCaptions.length} found):`);
    issues.videosWithoutCaptions.forEach(issue => {
      console.log(`   ${issue.file}:${issue.line}`);
    });
  }
  
  if (issues.missingKeyboardHandlers.length > 0) {
    console.log(`\n⚠️  Click handlers without keyboard support (${issues.missingKeyboardHandlers.length} found):`);
    issues.missingKeyboardHandlers.slice(0, 5).forEach(issue => {
      console.log(`   ${issue.file}:${issue.line}`);
    });
  }
  
  const totalIssues = Object.values(issues).reduce((sum, arr) => sum + arr.length, 0);
  
  if (totalIssues === 0) {
    console.log('✅ No accessibility issues found!');
  } else {
    console.log(`\n📋 Total issues found: ${totalIssues}`);
    console.log('\nRefer to A11Y_FIXES_GUIDE.md for how to fix these issues.');
  }
}

// Check if glob is installed
import('glob').then(() => {
  scanForA11yIssues().catch(console.error);
}).catch(() => {
  console.log('Installing required dependency...');
  import('child_process').then(({ execSync }) => {
    execSync('npm install glob', { stdio: 'inherit' });
    console.log('Dependency installed. Please run the script again.');
  });
});
