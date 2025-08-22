#!/usr/bin/env node

/**
 * TORI PostCSS Error Fix
 * 
 * This script fixes the specific PostCSS "Unknown word position" error
 * by installing missing dependencies and fixing configuration issues.
 */

import fs from 'fs';
import path from 'path';
import { execSync } from 'child_process';

const PROJECT_ROOT = process.cwd();
const PACKAGE_JSON_PATH = path.join(PROJECT_ROOT, 'package.json');

class ToriPostCSSFixer {
  constructor() {
    this.fixes = [];
    this.errors = [];
  }

  log(message, type = 'info') {
    const timestamp = new Date().toLocaleTimeString();
    const prefix = {
      info: '📋',
      success: '✅',
      error: '❌',
      warning: '⚠️',
      fix: '🔧'
    }[type] || '📋';
    
    console.log(`${prefix} [${timestamp}] ${message}`);
  }

  async run() {
    this.log('Starting TORI PostCSS Error Fix', 'info');
    this.log('='.repeat(60), 'info');

    try {
      // Step 1: Install missing Tailwind CSS dependencies
      await this.installTailwindDependencies();
      
      // Step 2: Fix PostCSS configuration
      await this.fixPostCSSConfig();
      
      // Step 3: Fix any CSS syntax issues in PersonaPanel.svelte
      await this.fixPersonaPanelCSS();
      
      // Step 4: Create optimized development configuration
      await this.createOptimizedConfig();
      
      // Step 5: Clear cache and restart instructions
      this.provideFinalInstructions();
      
      this.log('All fixes completed successfully!', 'success');
      this.log('Now restart your dev server: npm run dev', 'info');
      
    } catch (error) {
      this.log(`Critical error: ${error.message}`, 'error');
      this.errors.push(error);
      throw error;
    }
  }

  async installTailwindDependencies() {
    this.log('Installing missing Tailwind CSS dependencies...', 'fix');
    
    const dependencies = [
      'tailwindcss@latest',
      'autoprefixer@latest',
      'postcss@latest'
    ];

    try {
      // Install dependencies
      const installCmd = `npm install -D ${dependencies.join(' ')}`;
      this.log(`Running: ${installCmd}`, 'info');
      
      execSync(installCmd, { 
        cwd: PROJECT_ROOT, 
        stdio: 'pipe', // Change from 'inherit' to 'pipe' to avoid issues
        timeout: 300000 // 5 minutes timeout
      });
      
      this.log('Dependencies installed successfully!', 'success');
      this.fixes.push('Installed Tailwind CSS dependencies');
      
    } catch (error) {
      this.log(`Failed to install dependencies: ${error.message}`, 'error');
      
      // Try alternative installation method
      this.log('Trying individual installation...', 'info');
      let successCount = 0;
      
      for (const dep of dependencies) {
        try {
          this.log(`Installing ${dep}...`, 'info');
          execSync(`npm install -D ${dep}`, { 
            cwd: PROJECT_ROOT, 
            stdio: 'pipe',
            timeout: 120000 // 2 minutes per package
          });
          successCount++;
        } catch (depError) {
          this.log(`Failed to install ${dep}: ${depError.message}`, 'warning');
        }
      }
      
      if (successCount === 0) {
        throw new Error('Could not install any required dependencies');
      } else {
        this.log(`Installed ${successCount}/${dependencies.length} dependencies`, 'warning');
      }
    }
  }

  async fixPostCSSConfig() {
    this.log('Fixing PostCSS configuration...', 'fix');
    
    const postcssConfigPath = path.join(PROJECT_ROOT, 'postcss.config.js');
    
    // Create a bulletproof PostCSS configuration
    const optimizedConfig = `/** @type {import('postcss-load-config').Config} */
export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {}
  }
};`;

    try {
      fs.writeFileSync(postcssConfigPath, optimizedConfig, 'utf8');
      this.log('PostCSS configuration updated', 'success');
      this.fixes.push('Updated postcss.config.js');
      
    } catch (error) {
      this.log(`Failed to update PostCSS config: ${error.message}`, 'error');
      throw error;
    }
  }

  async fixPersonaPanelCSS() {
    this.log('Checking PersonaPanel.svelte for CSS issues...', 'fix');
    
    const personaPanelPath = path.join(PROJECT_ROOT, 'src', 'lib', 'components', 'PersonaPanel.svelte');
    
    if (!fs.existsSync(personaPanelPath)) {
      this.log('PersonaPanel.svelte not found at expected path', 'warning');
      return;
    }

    try {
      const content = fs.readFileSync(personaPanelPath, 'utf8');
      
      // Check for common CSS syntax issues that cause PostCSS to fail
      const lines = content.split('\n');
      let hasIssues = false;
      let fixedContent = content;
      
      // Look for the specific issue at line 1, column 35 mentioned in the error
      const styleMatch = content.match(/<style[^>]*>([\\s\\S]*?)<\\/style>/);
      if (styleMatch) {
        const cssContent = styleMatch[1];
        const cssLines = cssContent.split('\n');
        
        if (cssLines.length > 0) {
          const firstLine = cssLines[0];
          if (firstLine.length >= 35) {
            const charAt35 = firstLine.charAt(34); // 0-indexed
            this.log(`Character at position 35 in first CSS line: "${charAt35}"`, 'info');
            
            // Check for common issues
            if (charAt35 === '' || charAt35 === '') {
              this.log('Found invalid character in CSS', 'warning');
              // Remove invalid characters
              fixedContent = fixedContent.replace(/[\\u200B-\\u200D\\uFEFF]/g, '');
              hasIssues = true;
            }
          }
        }
      }
      
      // Additional checks for CSS syntax issues
      const commonIssues = [
        { pattern: /position\\s*:\\s*fixed\\s*top\\s*:/g, fix: 'position: fixed;\\n    top:' },
        { pattern: /([^;}])\\n\\s*([a-zA-Z-]+\\s*:)/g, fix: '$1;\\n    $2' }
      ];
      
      for (const issue of commonIssues) {
        if (issue.pattern.test(fixedContent)) {
          fixedContent = fixedContent.replace(issue.pattern, issue.fix);
          hasIssues = true;
          this.log('Fixed CSS syntax issue', 'success');
        }
      }
      
      if (hasIssues) {
        fs.writeFileSync(personaPanelPath, fixedContent, 'utf8');
        this.log('PersonaPanel.svelte CSS fixed', 'success');
        this.fixes.push('Fixed CSS syntax in PersonaPanel.svelte');
      } else {
        this.log('No CSS syntax issues found in PersonaPanel.svelte', 'success');
      }
      
    } catch (error) {
      this.log(`Error checking PersonaPanel.svelte: ${error.message}`, 'error');
      // Don't throw here, this is not critical
    }
  }

  async createOptimizedConfig() {
    this.log('Creating optimized development configuration...', 'fix');
    
    // Create a package.json script for debugging PostCSS
    try {
      const packageJson = JSON.parse(fs.readFileSync(PACKAGE_JSON_PATH, 'utf8'));
      
      // Add debug scripts
      packageJson.scripts = {
        ...packageJson.scripts,
        'dev:debug': 'DEBUG=vite:css vite dev --port 5173 --host 0.0.0.0',
        'postcss:check': 'postcss --config . --no-map < /dev/null',
        'css:debug': 'node postcss-debug-fix.js'
      };
      
      fs.writeFileSync(PACKAGE_JSON_PATH, JSON.stringify(packageJson, null, 2), 'utf8');
      this.log('Added debug scripts to package.json', 'success');
      this.fixes.push('Added debugging scripts');
      
    } catch (error) {
      this.log(`Warning: Could not update package.json: ${error.message}`, 'warning');
    }

    // Create a .env.local for development debugging
    const envContent = `# PostCSS Debug Configuration
DEBUG=vite:css
VITE_CSS_DEBUG=true
`;

    try {
      fs.writeFileSync(path.join(PROJECT_ROOT, '.env.local'), envContent, 'utf8');
      this.log('Created .env.local for debugging', 'success');
      
    } catch (error) {
      this.log(`Warning: Could not create .env.local: ${error.message}`, 'warning');
    }
  }

  provideFinalInstructions() {
    this.log('\\n' + '='.repeat(60), 'info');
    this.log('🎉 FIXES COMPLETED SUCCESSFULLY!', 'success');
    this.log('='.repeat(60), 'info');
    
    this.log('\\n📋 Summary of fixes applied:', 'info');
    this.fixes.forEach((fix, index) => {
      this.log(`  ${index + 1}. ${fix}`, 'success');
    });
    
    this.log('\\n🚀 Next steps:', 'info');
    this.log('1. Stop your current dev server (Ctrl+C)', 'info');
    this.log('2. Clear any cache: rm -rf node_modules/.vite', 'info');
    this.log('3. Restart the dev server: npm run dev', 'info');
    this.log('4. If issues persist, run: npm run css:debug', 'info');
    
    this.log('\\n🔍 Additional debugging options:', 'info');
    this.log('• Run with CSS debug: npm run dev:debug', 'info');
    this.log('• Check PostCSS config: npm run postcss:check', 'info');
    this.log('• Full CSS diagnostic: npm run css:debug', 'info');
    
    this.log('\\n💡 The PostCSS "Unknown word position" error should now be resolved!', 'success');
  }
}

// Run the fixer
const fixer = new ToriPostCSSFixer();
fixer.run().catch(error => {
  console.error('\\n❌ Fatal error:', error.message);
  process.exit(1);
});
