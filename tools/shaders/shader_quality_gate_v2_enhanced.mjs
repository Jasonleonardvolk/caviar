// shader_quality_gate_v2_enhanced.mjs
// Enhanced shader validator with smart suppression for false positives

import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ============================================================================
// SUPPRESSION SYSTEM
// ============================================================================

class SuppressionSystem {
  constructor(rulesPath) {
    this.rules = this.loadRules(rulesPath);
    this.fileCache = new Map();
    this.stats = {
      totalSuppressed: 0,
      byRule: new Map()
    };
  }

  loadRules(explicitPath) {
    const defaultPath = path.join(__dirname, 'validator_suppressions.json');
    const rulesPath = explicitPath || defaultPath;
    
    if (!fs.existsSync(rulesPath)) {
      console.log('⚠️  No suppression rules found, all warnings will be shown');
      return null;
    }
    
    try {
      const data = JSON.parse(fs.readFileSync(rulesPath, 'utf8'));
      console.log(`📋 Loaded ${data.rules?.length || 0} suppression rules from ${path.basename(rulesPath)}`);
      return data;
    } catch (err) {
      console.error('❌ Failed to load suppression rules:', err.message);
      return null;
    }
  }

  getFileText(filePath) {
    if (this.fileCache.has(filePath)) {
      return this.fileCache.get(filePath);
    }
    
    try {
      const text = fs.readFileSync(filePath, 'utf8');
      this.fileCache.set(filePath, text);
      return text;
    } catch {
      this.fileCache.set(filePath, '');
      return '';
    }
  }

  shouldSuppress(filePath, message, verbose = false) {
    if (!this.rules?.rules) return false;
    
    const fileText = this.getFileText(filePath);
    
    for (const rule of this.rules.rules) {
      // Check message pattern
      if (rule.messageContains && !message.includes(rule.messageContains)) {
        continue;
      }
      
      // Check file must contain
      if (rule.fileMustContain && !fileText.includes(rule.fileMustContain)) {
        if (verbose) {
          console.log(`  ↳ Not suppressing: file missing "${rule.fileMustContain}"`);
        }
        continue;
      }
      
      // Check file must NOT contain
      if (rule.fileMustContainNot && fileText.includes(rule.fileMustContainNot)) {
        if (verbose) {
          console.log(`  ↳ Not suppressing: file contains "${rule.fileMustContainNot}"`);
        }
        continue;
      }
      
      // This rule matches - suppress the warning
      this.stats.totalSuppressed++;
      this.stats.byRule.set(rule.name || rule.reason, 
        (this.stats.byRule.get(rule.name || rule.reason) || 0) + 1);
      
      if (verbose) {
        console.log(`  ✓ Suppressed by rule: ${rule.reason}`);
      }
      
      return true;
    }
    
    return false;
  }

  printStats() {
    if (this.stats.totalSuppressed === 0) return;
    
    console.log('\n📊 Suppression Statistics:');
    console.log(`  Total warnings suppressed: ${this.stats.totalSuppressed}`);
    
    for (const [ruleName, count] of this.stats.byRule) {
      console.log(`    - ${ruleName}: ${count}`);
    }
  }
}

// ============================================================================
// SHADER VALIDATOR
// ============================================================================

class ShaderValidator {
  constructor(options = {}) {
    this.dir = options.dir || 'frontend/lib/webgpu/shaders';
    this.limits = options.limits;
    this.targets = options.targets?.split(',') || ['naga'];
    this.strict = options.strict === 'true' || options.strict === true;
    this.report = options.report || path.join(process.cwd(), 'build', 'shader_report.json');
    this.junit = options.junit || path.join(process.cwd(), 'build', 'shader_report.junit.xml');
    this.suppressions = options.suppressions;
    this.verbose = options.verbose === 'true' || options.verbose === true;
    
    this.suppressor = new SuppressionSystem(this.suppressions);
    this.results = {
      timestamp: new Date().toISOString(),
      files: [],
      summary: {
        total: 0,
        passed: 0,
        failed: 0,
        warnings: 0,
        suppressed: 0
      }
    };
  }

  async checkToolAvailability() {
    const tools = {};
    
    for (const target of this.targets) {
      tools[target] = await this.isToolAvailable(target);
    }
    
    return tools;
  }

  async isToolAvailable(tool) {
    return new Promise((resolve) => {
      const proc = spawn(tool, ['--version'], { shell: true });
      proc.on('close', (code) => resolve(code === 0));
      proc.on('error', () => resolve(false));
    });
  }

  async validateFile(filePath, target) {
    return new Promise((resolve) => {
      const args = target === 'naga' ? [filePath] : ['--validate', filePath];
      const proc = spawn(target, args, { shell: true });
      
      let stdout = '';
      let stderr = '';
      
      proc.stdout.on('data', (data) => { stdout += data; });
      proc.stderr.on('data', (data) => { stderr += data; });
      
      proc.on('close', (code) => {
        const errors = [];
        const warnings = [];
        
        // Parse output for errors and warnings
        const output = stdout + stderr;
        const lines = output.split('\n');
        
        for (const line of lines) {
          if (line.includes('error:') || line.includes('Error:')) {
            errors.push(line.trim());
          } else if (line.includes('warning:') || line.includes('Warning:')) {
            warnings.push(line.trim());
          } else if (line.includes('Dynamic array access')) {
            warnings.push(line.trim());
          } else if (line.includes('vec3') && line.includes('storage')) {
            warnings.push(line.trim());
          }
        }
        
        resolve({
          target,
          success: code === 0,
          errors,
          warnings
        });
      });
      
      proc.on('error', () => {
        resolve({
          target,
          success: false,
          errors: [`${target} not available`],
          warnings: []
        });
      });
    });
  }

  async processFile(filePath) {
    const fileName = path.basename(filePath);
    const fileResult = {
      file: filePath,
      errors: [],
      warnings: [],
      suppressed: [],
      targets: {}
    };
    
    console.log(`\n📄 ${fileName}`);
    
    for (const target of this.targets) {
      const result = await this.validateFile(filePath, target);
      fileResult.targets[target] = result;
      
      if (result.success) {
        console.log(`  ✅ ${target}: Valid`);
      } else if (result.errors.includes(`${target} not available`)) {
        console.log(`  ⏭️  ${target}: Skipped (not available)`);
      } else {
        console.log(`  ❌ ${target}: Failed`);
      }
      
      // Collect and filter warnings
      for (const warning of result.warnings) {
        if (this.suppressor.shouldSuppress(filePath, warning, this.verbose)) {
          fileResult.suppressed.push({
            message: warning,
            target
          });
        } else {
          fileResult.warnings.push({
            message: warning,
            target
          });
        }
      }
      
      // Never suppress errors
      fileResult.errors.push(...result.errors.map(e => ({ message: e, target })));
    }
    
    return fileResult;
  }

  async validate() {
    console.log('🚀 Enhanced Shader Quality Gate v2.1');
    console.log('================================\n');
    
    // Check tool availability
    const tools = await this.checkToolAvailability();
    for (const [tool, available] of Object.entries(tools)) {
      if (available) {
        console.log(`✅ ${tool} found`);
      } else {
        console.log(`⚠️  ${tool} not found`);
      }
    }
    
    // Load device limits if specified
    if (this.limits) {
      console.log(`📱 Using device limits: ${path.basename(this.limits)}`);
    }
    
    // Find all shader files
    const fullDir = path.resolve(process.cwd(), this.dir);
    const files = fs.readdirSync(fullDir)
      .filter(f => f.endsWith('.wgsl'))
      .map(f => path.join(fullDir, f));
    
    console.log(`\n🔍 Found ${files.length} WGSL files\n`);
    
    // Process each file
    for (const file of files) {
      const result = await this.processFile(file);
      this.results.files.push(result);
      
      // Update summary
      this.results.summary.total++;
      if (result.errors.length === 0 && result.warnings.length === 0) {
        this.results.summary.passed++;
      } else if (result.errors.length > 0) {
        this.results.summary.failed++;
      }
      this.results.summary.warnings += result.warnings.length;
      this.results.summary.suppressed += result.suppressed.length;
    }
    
    // Print summary
    this.printSummary();
    
    // Print suppression stats
    this.suppressor.printStats();
    
    // Save reports
    await this.saveReports();
    
    // Exit with error if strict and there are failures
    if (this.strict && this.results.summary.failed > 0) {
      process.exit(1);
    }
  }

  printSummary() {
    console.log('\n================================');
    console.log('📊 Summary');
    console.log(`  Total: ${this.results.summary.total}`);
    console.log(`  ✅ Passed: ${this.results.summary.passed}`);
    console.log(`  ❌ Failed: ${this.results.summary.failed}`);
    console.log(`  ⚠️  Warnings: ${this.results.summary.warnings}`);
    console.log(`  🔇 Suppressed: ${this.results.summary.suppressed}`);
  }

  async saveReports() {
    // Ensure build directory exists
    const buildDir = path.dirname(this.report);
    if (!fs.existsSync(buildDir)) {
      fs.mkdirSync(buildDir, { recursive: true });
    }
    
    // Save JSON report
    fs.writeFileSync(this.report, JSON.stringify(this.results, null, 2));
    console.log(`\n📄 Report saved to ${this.report}`);
    
    // Save JUnit report
    const junit = this.generateJUnit();
    fs.writeFileSync(this.junit, junit);
    console.log(`📄 JUnit report saved to ${this.junit}`);
  }

  generateJUnit() {
    const testcases = this.results.files.map(file => {
      const name = path.basename(file.file);
      const hasErrors = file.errors.length > 0;
      const hasWarnings = file.warnings.length > 0;
      const status = hasErrors ? 'failed' : hasWarnings ? 'warning' : 'passed';
      
      let testcase = `    <testcase name="${name}" classname="ShaderValidation" status="${status}">`;
      
      if (hasErrors) {
        testcase += `\n      <failure message="${file.errors.length} errors">`;
        for (const error of file.errors) {
          testcase += `\n        ${error.message}`;
        }
        testcase += `\n      </failure>`;
      }
      
      if (hasWarnings) {
        testcase += `\n      <system-out>`;
        testcase += `\n        Warnings: ${file.warnings.length}`;
        for (const warning of file.warnings) {
          testcase += `\n        - ${warning.message}`;
        }
        testcase += `\n      </system-out>`;
      }
      
      if (file.suppressed.length > 0) {
        testcase += `\n      <!-- Suppressed: ${file.suppressed.length} false positives -->`;
      }
      
      testcase += `\n    </testcase>`;
      return testcase;
    }).join('\n');
    
    return `<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="ShaderValidation" tests="${this.results.summary.total}" failures="${this.results.summary.failed}" warnings="${this.results.summary.warnings}">
${testcases}
</testsuite>`;
  }
}

// ============================================================================
// CLI INTERFACE
// ============================================================================

if (import.meta.url === `file://${process.argv[1]}`) {
  const args = {};
  
  // Parse command line arguments
  for (let i = 2; i < process.argv.length; i++) {
    const arg = process.argv[i];
    if (arg.startsWith('--')) {
      const [key, value] = arg.substring(2).split('=');
      args[key] = value || process.argv[++i];
    }
  }
  
  // Show help if requested
  if (args.help) {
    console.log(`
Enhanced Shader Quality Gate v2.1

Usage: node shader_quality_gate_v2_enhanced.mjs [options]

Options:
  --dir=PATH           Directory containing shaders (default: frontend/lib/webgpu/shaders)
  --limits=PATH        Device limits JSON file
  --targets=LIST       Comma-separated validation targets (default: naga)
  --strict             Exit with error on failures
  --report=PATH        JSON report output path
  --junit=PATH         JUnit XML report path
  --suppressions=PATH  Custom suppression rules file
  --verbose            Show detailed suppression info
  --help               Show this help message

Examples:
  # Basic validation
  node shader_quality_gate_v2_enhanced.mjs

  # iPhone validation with suppressions
  node shader_quality_gate_v2_enhanced.mjs \\
    --limits=tools/shaders/device_limits/iphone15.json \\
    --suppressions=tools/shaders/validator_suppressions.json \\
    --strict
`);
    process.exit(0);
  }
  
  // Run validation
  const validator = new ShaderValidator(args);
  validator.validate().catch(console.error);
}

export { ShaderValidator, SuppressionSystem };
