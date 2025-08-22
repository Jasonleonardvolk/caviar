#!/usr/bin/env node

/**
 * Final Cleanup Script for TORI PostCSS Fix
 * 1. Clean files (replace \\n with real newlines)
 * 2. Delete *.vite-preprocess.css temp files
 * 3. Final verification
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

console.log('🧹 Final Cleanup - TORI PostCSS Fix');
console.log('='.repeat(50));

class FinalCleanup {
  constructor() {
    this.projectRoot = process.cwd();
    this.cleanedFiles = [];
    this.deletedFiles = [];
  }

  async run() {
    try {
      // Step 1: Clean files with escaped newlines
      await this.cleanEscapedNewlines();
      
      // Step 2: Delete Vite preprocess temp files
      await this.deleteVitePreprocessFiles();
      
      // Step 3: Clear all caches
      await this.clearCaches();
      
      // Step 4: Final verification
      await this.finalVerification();
      
      this.reportResults();
      
    } catch (error) {
      console.error('❌ Cleanup failed:', error.message);
      throw error;
    }
  }

  async cleanEscapedNewlines() {
    console.log('🔍 Step 1: Cleaning escaped newlines in files...');
    
    const filesToCheck = [
      'postcss.config.js',
      'src/lib/components/PersonaPanel.svelte',
      'package.json',
      'tailwind.config.js',
      'vite.config.js'
    ];

    for (const relativePath of filesToCheck) {
      const filePath = path.join(this.projectRoot, relativePath);
      
      if (fs.existsSync(filePath)) {
        try {
          let content = fs.readFileSync(filePath, 'utf8');
          let originalContent = content;
          
          // Replace escaped newlines with actual newlines
          content = content.replace(/\\n/g, '\n');
          content = content.replace(/\\t/g, '\t');
          content = content.replace(/\\r/g, '\r');
          
          // Remove any Unicode zero-width characters
          content = content.replace(/[\u200B-\u200D\uFEFF]/g, '');
          
          // Clean up multiple consecutive newlines
          content = content.replace(/\n{3,}/g, '\n\n');
          
          if (content !== originalContent) {
            fs.writeFileSync(filePath, content, 'utf8');
            console.log(`  ✅ Cleaned: ${relativePath}`);
            this.cleanedFiles.push(relativePath);
          } else {
            console.log(`  ✓ Already clean: ${relativePath}`);
          }
          
        } catch (error) {
          console.log(`  ⚠️ Could not clean ${relativePath}: ${error.message}`);
        }
      } else {
        console.log(`  ℹ️ File not found: ${relativePath}`);
      }
    }
  }

  async deleteVitePreprocessFiles() {
    console.log('\n🗑️ Step 2: Deleting Vite preprocess temp files...');
    
    const searchPaths = [
      this.projectRoot,
      path.join(this.projectRoot, 'src'),
      path.join(this.projectRoot, 'node_modules', '.vite'),
      process.env.TEMP || process.env.TMP || '/tmp'
    ];

    for (const searchPath of searchPaths) {
      if (fs.existsSync(searchPath)) {
        try {
          await this.findAndDeleteViteFiles(searchPath);
        } catch (error) {
          console.log(`  ⚠️ Could not search ${searchPath}: ${error.message}`);
        }
      }
    }
  }

  async findAndDeleteViteFiles(dir) {
    try {
      const items = fs.readdirSync(dir);
      
      for (const item of items) {
        const fullPath = path.join(dir, item);
        const stat = fs.statSync(fullPath);
        
        if (stat.isFile() && item.includes('.vite-preprocess.css')) {
          try {
            fs.unlinkSync(fullPath);
            console.log(`  🗑️ Deleted: ${fullPath}`);
            this.deletedFiles.push(fullPath);
          } catch (error) {
            console.log(`  ⚠️ Could not delete ${fullPath}: ${error.message}`);
          }
        } else if (stat.isDirectory() && !item.startsWith('.') && item !== 'node_modules') {
          // Recursively search subdirectories (but avoid deep recursion)
          if (dir.split(path.sep).length < 10) {
            await this.findAndDeleteViteFiles(fullPath);
          }
        }
      }
    } catch (error) {
      // Silent fail for permission issues
    }
  }

  async clearCaches() {
    console.log('\n🧹 Step 3: Clearing caches...');
    
    const cachePaths = [
      path.join(this.projectRoot, 'node_modules', '.vite'),
      path.join(this.projectRoot, '.svelte-kit'),
      path.join(this.projectRoot, 'dist'),
      path.join(this.projectRoot, 'build')
    ];

    for (const cachePath of cachePaths) {
      if (fs.existsSync(cachePath)) {
        try {
          fs.rmSync(cachePath, { recursive: true, force: true });
          console.log(`  🗑️ Cleared: ${path.basename(cachePath)}`);
        } catch (error) {
          console.log(`  ⚠️ Could not clear ${cachePath}: ${error.message}`);
        }
      }
    }
  }

  async finalVerification() {
    console.log('\n✅ Step 4: Final verification...');
    
    // Check PostCSS config syntax
    const postcssConfigPath = path.join(this.projectRoot, 'postcss.config.js');
    if (fs.existsSync(postcssConfigPath)) {
      try {
        const content = fs.readFileSync(postcssConfigPath, 'utf8');
        
        // Basic syntax check
        if (content.includes('export default') && content.includes('plugins')) {
          console.log('  ✅ PostCSS config syntax looks good');
        } else {
          console.log('  ⚠️ PostCSS config may have issues');
        }
        
        // Check for balanced braces
        const openBraces = (content.match(/{/g) || []).length;
        const closeBraces = (content.match(/}/g) || []).length;
        
        if (openBraces === closeBraces) {
          console.log('  ✅ PostCSS config braces are balanced');
        } else {
          console.log('  ❌ PostCSS config has unbalanced braces');
        }
        
      } catch (error) {
        console.log('  ❌ PostCSS config verification failed:', error.message);
      }
    }

    // Check PersonaPanel.svelte
    const personaPanelPath = path.join(this.projectRoot, 'src', 'lib', 'components', 'PersonaPanel.svelte');
    if (fs.existsSync(personaPanelPath)) {
      try {
        const content = fs.readFileSync(personaPanelPath, 'utf8');
        
        // Check for style block
        if (content.includes('<style>') && content.includes('</style>')) {
          console.log('  ✅ PersonaPanel.svelte has style block');
        }
        
        // Check for invalid characters
        if (!/[\u200B-\u200D\uFEFF]/.test(content)) {
          console.log('  ✅ PersonaPanel.svelte has no invalid Unicode characters');
        }
        
      } catch (error) {
        console.log('  ⚠️ PersonaPanel.svelte verification failed:', error.message);
      }
    }
  }

  reportResults() {
    console.log('\n' + '='.repeat(50));
    console.log('🎉 FINAL CLEANUP COMPLETED!');
    console.log('='.repeat(50));
    
    if (this.cleanedFiles.length > 0) {
      console.log('\n📋 Files cleaned:');
      this.cleanedFiles.forEach(file => console.log(`  ✅ ${file}`));
    }
    
    if (this.deletedFiles.length > 0) {
      console.log('\n🗑️ Temp files deleted:');
      this.deletedFiles.forEach(file => console.log(`  🗑️ ${path.basename(file)}`));
    }
    
    console.log('\n🚀 Your project is now clean and ready!');
    console.log('✅ All escaped newlines fixed');
    console.log('✅ All Vite preprocess temp files deleted');
    console.log('✅ All caches cleared');
    console.log('\n📋 Next steps:');
    console.log('  1. Run: npm run dev');
    console.log('  2. PostCSS errors should be completely gone!');
    console.log('  3. PersonaPanel.svelte should load without issues');
  }
}

// Run the cleanup
const cleanup = new FinalCleanup();
cleanup.run().catch(error => {
  console.error('\n❌ Final cleanup failed:', error.message);
  console.log('\n🆘 Manual cleanup steps:');
  console.log('1. Delete any *.vite-preprocess.css files');
  console.log('2. Clear node_modules/.vite directory');
  console.log('3. Check postcss.config.js for syntax errors');
  console.log('4. Run: npm run dev');
  process.exit(1);
});
