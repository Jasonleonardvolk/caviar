// validate_and_report.mjs
// Validates WGSL shaders against device limits and multiple compilation targets
// Generates detailed reports in multiple formats (JSON, JUnit, console)

import { spawn } from 'child_process';
import { promises as fs } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { loadAndValidateLimits } from './limits_resolver_v2.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

class ShaderValidator {
  constructor(options = {}) {
    this.dir = options.dir || 'frontend/lib/webgpu/shaders';
    this.limitsArg = options.limits || 'latest';
    this.targets = options.targets || ['naga', 'tint'];
    this.strict = options.strict || false;
    this.noSuppress = options.noSuppress || false;
    this.report = options.report || 'build/shader_report.json';
    this.junit = options.junit || 'build/shader_report.junit.xml';
    this.results = [];
    this.resolvedLimits = null;
    this.limitsPath = null;
  }

  async loadLimits() {
    const verbose = true; // Always show which limits are being used
    const { limits, path: resolvedPath, platform } = loadAndValidateLimits(this.limitsArg, { verbose });
    
    this.resolvedLimits = limits;
    this.limitsPath = resolvedPath;
    
    // Print limits summary
    console.log('\n📊 LIMITS SUMMARY:');
    console.log('─'.repeat(60));
    console.log(`Alias/Path: ${this.limitsArg}`);
    console.log(`Resolved: ${resolvedPath}`);
    console.log(`Platform: ${platform}`);
    console.log(`Values:`);
    console.log(`  - maxComputeInvocationsPerWorkgroup: ${limits.maxComputeInvocationsPerWorkgroup}`);
    console.log(`  - maxComputeWorkgroupSizeX: ${limits.maxComputeWorkgroupSizeX}`);
    console.log(`  - maxComputeWorkgroupSizeY: ${limits.maxComputeWorkgroupSizeY}`);
    console.log(`  - maxComputeWorkgroupSizeZ: ${limits.maxComputeWorkgroupSizeZ}`);
    if (limits.maxComputeWorkgroupStorageSize) {
      console.log(`  - maxComputeWorkgroupStorageSize: ${limits.maxComputeWorkgroupStorageSize} (${(limits.maxComputeWorkgroupStorageSize/1024).toFixed(1)} KiB)`);
    }
    console.log('─'.repeat(60) + '\n');
  }

  async validateFile(file, target) {
    return new Promise((resolve) => {
      const args = target === 'naga' ? [file] : ['--validate', file];
      const proc = spawn(target, args, { shell: true });
      
      let stdout = '';
      let stderr = '';
      
      proc.stdout.on('data', (data) => { stdout += data; });
      proc.stderr.on('data', (data) => { stderr += data; });
      
      proc.on('close', (code) => {
        resolve({
          file: path.basename(file),
          target,
          success: code === 0,
          stdout,
          stderr,
          warnings: this.extractWarnings(stdout + stderr)
        });
      });
      
      proc.on('error', () => {
        resolve({
          file: path.basename(file),
          target,
          success: false,
          error: `${target} not found`
        });
      });
    });
  }

  extractWarnings(output) {
    const warnings = [];
    const patterns = [
      /Dynamic array access.*without.*bounds checking/gi,
      /vec3.*storage buffer.*padding/gi,
      /Consider using const instead of let/gi
    ];
    
    for (const pattern of patterns) {
      const matches = output.match(pattern);
      if (matches) {
        warnings.push(...matches);
      }
    }
    
    return warnings;
  }

  async validate() {
    // Load and validate limits first
    await this.loadLimits();
    
    const fullDir = path.join(__dirname, '../..', this.dir);
    const files = await fs.readdir(fullDir);
    const wgslFiles = files.filter(f => f.endsWith('.wgsl'));
    
    console.log(`[validator] Found ${wgslFiles.length} WGSL files`);
    
    for (const file of wgslFiles) {
      const filePath = path.join(fullDir, file);
      console.log(`[validator] Checking ${file}...`);
      
      for (const target of this.targets) {
        const result = await this.validateFile(filePath, target);
        this.results.push(result);
        
        if (result.success) {
          console.log(`  ✅ ${target}: Valid`);
        } else if (result.error) {
          console.log(`  ⏭️ ${target}: ${result.error}`);
        } else {
          console.log(`  ❌ ${target}: Failed`);
          if (this.strict) {
            throw new Error(`Validation failed for ${file}`);
          }
        }
        
        if (result.warnings?.length && !this.noSuppress) {
          console.log(`  ⚠️  ${result.warnings.length} warnings (suppressed)`);
        } else if (result.warnings?.length) {
          console.log(`  ⚠️  ${result.warnings.length} warnings:`);
          result.warnings.forEach(w => console.log(`     - ${w}`));
        }
      }
    }
    
    // Generate reports
    await this.generateReports();
  }

  async generateReports() {
    // Ensure build directory exists
    const buildDir = path.dirname(this.report);
    await fs.mkdir(buildDir, { recursive: true });
    
    // JSON report with metadata
    const report = {
      timestamp: new Date().toISOString(),
      meta: {
        limits: {
          alias: this.limitsArg,
          resolved: this.limitsPath,
          values: this.resolvedLimits
        },
        targets: this.targets,
        strict: this.strict,
        noSuppress: this.noSuppress
      },
      results: this.results,
      summary: {
        total: this.results.length,
        passed: this.results.filter(r => r.success).length,
        failed: this.results.filter(r => !r.success && !r.error).length,
        skipped: this.results.filter(r => r.error).length,
        warnings: this.results.reduce((sum, r) => sum + (r.warnings?.length || 0), 0)
      }
    };
    
    await fs.writeFile(this.report, JSON.stringify(report, null, 2));
    console.log(`[validator] JSON report saved to ${this.report}`);
    
    // JUnit report
    const junit = this.generateJUnit();
    await fs.writeFile(this.junit, junit);
    console.log(`[validator] JUnit report saved to ${this.junit}`);
    
    // Console summary
    const failed = this.results.filter(r => !r.success && !r.error);
    const warnings = this.results.reduce((sum, r) => sum + (r.warnings?.length || 0), 0);
    
    console.log('\n=== VALIDATION SUMMARY ===');
    console.log(`Total: ${this.results.length}`);
    console.log(`Failed: ${failed.length}`);
    console.log(`Warnings: ${warnings}`);
    console.log(`Limits used: ${this.limitsPath}`);
  }

  generateJUnit() {
    const testcases = this.results.map(r => {
      const status = r.success ? 'passed' : r.error ? 'skipped' : 'failed';
      return `    <testcase name="${r.file}" classname="${r.target}" status="${status}">
      ${r.stderr ? `<failure>${r.stderr}</failure>` : ''}
      ${r.warnings?.length ? `<system-out>Warnings: ${r.warnings.join(', ')}</system-out>` : ''}
    </testcase>`;
    }).join('\n');
    
    return `<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="ShaderValidation" tests="${this.results.length}" limits="${this.limitsPath}">
${testcases}
</testsuite>`;
  }
}

// CLI interface
if (import.meta.url === `file://${process.argv[1]}`) {
  const args = process.argv.slice(2);
  const options = {};
  
  for (let i = 0; i < args.length; i++) {
    if (args[i].startsWith('--')) {
      const key = args[i].substring(2).split('=')[0];
      const value = args[i].includes('=') ? args[i].split('=')[1] : 
                    (args[i + 1] && !args[i + 1].startsWith('--')) ? args[i + 1] : true;
      options[key] = value;
      if (!args[i].includes('=') && value !== true) i++;
    }
  }
  
  // Print tool versions
  console.log('\n📦 TOOL VERSIONS:');
  console.log('─'.repeat(60));
  console.log(`Node: ${process.version}`);
  
  try {
    const { execSync } = await import('child_process');
    const nagaVersion = execSync('naga-cli --version', { encoding: 'utf8' }).trim();
    console.log(`Naga: ${nagaVersion}`);
  } catch {
    console.log('Naga: not found');
  }
  
  try {
    const { execSync } = await import('child_process');
    const tintVersion = execSync('tint --version', { encoding: 'utf8' }).trim();
    console.log(`Tint: ${tintVersion}`);
  } catch {
    console.log('Tint: not found');
  }
  
  console.log('─'.repeat(60));
  
  const validator = new ShaderValidator(options);
  validator.validate().catch(console.error);
}

export { ShaderValidator };
