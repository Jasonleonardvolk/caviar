/**
 * PostCSS Debug and Fix Utility
 * 
 * This script diagnoses and fixes PostCSS configuration issues in Svelte projects.
 * Common issues include missing dependencies, malformed CSS, and plugin conflicts.
 */

import fs from 'fs';
import path from 'path';
import { execSync } from 'child_process';

class PostCSSDiagnostic {
  constructor(projectPath) {
    this.projectPath = projectPath;
    this.packageJsonPath = path.join(projectPath, 'package.json');
    this.postcssConfigPath = path.join(projectPath, 'postcss.config.js');
    this.tailwindConfigPath = path.join(projectPath, 'tailwind.config.js');
    this.issues = [];
    this.fixes = [];
  }

  async diagnose() {
    console.log('🔍 PostCSS Diagnostic Starting...');
    
    // Check package.json for missing dependencies
    await this.checkDependencies();
    
    // Check PostCSS configuration
    await this.checkPostCSSConfig();
    
    // Check for malformed CSS syntax
    await this.checkCSSFiles();
    
    // Check Tailwind configuration
    await this.checkTailwindConfig();
    
    this.reportFindings();
    return this.issues;
  }

  async checkDependencies() {
    try {
      const packageJson = JSON.parse(fs.readFileSync(this.packageJsonPath, 'utf8'));
      const allDeps = {
        ...packageJson.dependencies,
        ...packageJson.devDependencies
      };

      const requiredDeps = [
        'tailwindcss',
        'autoprefixer',
        'postcss'
      ];

      for (const dep of requiredDeps) {
        if (!allDeps[dep]) {
          this.issues.push({
            type: 'missing_dependency',
            message: `Missing dependency: ${dep}`,
            severity: 'critical',
            fix: `npm install -D ${dep}`
          });
        }
      }

      // Check for version conflicts
      if (allDeps['postcss'] && allDeps['vite']) {
        console.log('✅ Found PostCSS and Vite dependencies');
      }

    } catch (error) {
      this.issues.push({
        type: 'config_error',
        message: 'Cannot read package.json',
        severity: 'critical',
        error: error.message
      });
    }
  }

  async checkPostCSSConfig() {
    try {
      if (!fs.existsSync(this.postcssConfigPath)) {
        this.issues.push({
          type: 'missing_config',
          message: 'postcss.config.js not found',
          severity: 'warning'
        });
        return;
      }

      const config = fs.readFileSync(this.postcssConfigPath, 'utf8');
      
      // Check for common syntax issues
      if (config.includes('tailwindcss') && !config.includes('autoprefixer')) {
        this.issues.push({
          type: 'config_incomplete',
          message: 'PostCSS config missing autoprefixer',
          severity: 'warning'
        });
      }

      // Check for ES module vs CommonJS issues
      if (config.includes('export default') && !config.includes('module.exports')) {
        console.log('✅ Using ES modules in PostCSS config');
      }

    } catch (error) {
      this.issues.push({
        type: 'config_error',
        message: 'Error reading PostCSS config',
        severity: 'high',
        error: error.message
      });
    }
  }

  async checkCSSFiles() {
    const srcPath = path.join(this.projectPath, 'src');
    if (!fs.existsSync(srcPath)) return;

    const findSvelteFiles = (dir) => {
      const files = [];
      const items = fs.readdirSync(dir);
      
      for (const item of items) {
        const fullPath = path.join(dir, item);
        const stat = fs.statSync(fullPath);
        
        if (stat.isDirectory()) {
          files.push(...findSvelteFiles(fullPath));
        } else if (item.endsWith('.svelte')) {
          files.push(fullPath);
        }
      }
      return files;
    };

    const svelteFiles = findSvelteFiles(srcPath);
    
    for (const file of svelteFiles) {
      try {
        const content = fs.readFileSync(file, 'utf8');
        this.checkSvelteCSS(content, file);
      } catch (error) {
        this.issues.push({
          type: 'file_error',
          message: `Cannot read file: ${file}`,
          severity: 'medium',
          error: error.message
        });
      }
    }
  }

  checkSvelteCSS(content, filePath) {
    // Extract CSS from <style> blocks
    const styleRegex = /<style[^>]*>([\s\S]*?)<\/style>/g;
    let match;
    
    while ((match = styleRegex.exec(content)) !== null) {
      const cssContent = match[1];
      const lineOffset = content.substring(0, match.index).split('\n').length;
      
      // Check for common CSS syntax errors
      this.validateCSSSyntax(cssContent, filePath, lineOffset);
    }
  }

  validateCSSSyntax(cssContent, filePath, lineOffset) {
    const lines = cssContent.split('\n');
    
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      const lineNumber = i + lineOffset;
      
      // Check for missing semicolons
      if (line.includes(':') && !line.endsWith(';') && !line.endsWith('{') && !line.endsWith('}') && line !== '') {
        // Skip CSS custom properties and certain valid cases
        if (!line.includes('--') && !line.includes('@') && !line.includes('/*')) {
          this.issues.push({
            type: 'css_syntax',
            message: `Possible missing semicolon in ${path.basename(filePath)} at line ${lineNumber}`,
            severity: 'medium',
            line: lineNumber,
            content: line,
            file: filePath
          });
        }
      }
      
      // Check for unclosed braces
      const openBraces = (line.match(/{/g) || []).length;
      const closeBraces = (line.match(/}/g) || []).length;
      
      // Check for invalid characters that might cause PostCSS to fail
      if (line.includes('') || line.includes('')) {
        this.issues.push({
          type: 'css_syntax',
          message: `Invalid character in CSS at line ${lineNumber}`,
          severity: 'high',
          line: lineNumber,
          content: line,
          file: filePath
        });
      }
    }
  }

  async checkTailwindConfig() {
    if (!fs.existsSync(this.tailwindConfigPath)) {
      this.issues.push({
        type: 'missing_config',
        message: 'tailwind.config.js found but tailwindcss not installed',
        severity: 'high'
      });
    }
  }

  reportFindings() {
    console.log('\n📊 Diagnostic Results:');
    console.log('='.repeat(50));
    
    if (this.issues.length === 0) {
      console.log('✅ No issues found!');
      return;
    }

    const criticalIssues = this.issues.filter(i => i.severity === 'critical');
    const highIssues = this.issues.filter(i => i.severity === 'high');
    const mediumIssues = this.issues.filter(i => i.severity === 'medium');
    const warningIssues = this.issues.filter(i => i.severity === 'warning');

    if (criticalIssues.length > 0) {
      console.log('\n🚨 CRITICAL ISSUES:');
      criticalIssues.forEach(issue => {
        console.log(`  ❌ ${issue.message}`);
        if (issue.fix) console.log(`     Fix: ${issue.fix}`);
      });
    }

    if (highIssues.length > 0) {
      console.log('\n⚠️  HIGH PRIORITY:');
      highIssues.forEach(issue => {
        console.log(`  🔴 ${issue.message}`);
        if (issue.file) console.log(`     File: ${issue.file}:${issue.line}`);
      });
    }

    if (mediumIssues.length > 0) {
      console.log('\n📋 MEDIUM PRIORITY:');
      mediumIssues.forEach(issue => {
        console.log(`  🟡 ${issue.message}`);
        if (issue.content) console.log(`     Content: "${issue.content}"`);
      });
    }

    if (warningIssues.length > 0) {
      console.log('\n💡 WARNINGS:');
      warningIssues.forEach(issue => {
        console.log(`  🟠 ${issue.message}`);
      });
    }
  }

  async autoFix() {
    console.log('\n🔧 Attempting automatic fixes...');
    
    for (const issue of this.issues) {
      if (issue.type === 'missing_dependency' && issue.fix) {
        console.log(`Installing ${issue.fix}...`);
        try {
          execSync(issue.fix, { cwd: this.projectPath, stdio: 'inherit' });
          console.log(`✅ Fixed: ${issue.message}`);
        } catch (error) {
          console.log(`❌ Failed to fix: ${issue.message}`);
          console.log(`Error: ${error.message}`);
        }
      }
    }
  }
}

export default PostCSSDiagnostic;
