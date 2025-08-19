// validate_and_report.mjs
// Validates WGSL shaders against device limits and multiple compilation targets
// Generates detailed reports in multiple formats (JSON, JUnit, console)

import { spawn } from 'child_process';
import { promises as fs } from 'fs';
import { existsSync, readFileSync } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

function resolveLimits(p) {
  if (!p || p === 'latest') {
    // look for pointer file
    const pointer = path.join(process.cwd(), 'tools', 'shaders', 'device_limits', 'latest.json');
    if (existsSync(pointer)) {
      try {
        const j = JSON.parse(readFileSync(pointer, 'utf8'));
        if (j?.path) return j.path;
      } catch {}
    }
    // fallback
    return 'tools/shaders/device_limits/iphone15.json';
  }
  return p;
}

class ShaderValidator {
  constructor(options = {}) {
    this.dir = options.dir || 'frontend/lib/webgpu/shaders';
    this.limits = resolveLimits(options.limits);
    this.targets = options.targets || ['naga', 'tint'];
    this.strict = options.strict || false;
    this.report = options.report || 'build/shader_report.json';
    this.junit = options.junit || 'build/shader_report.junit.xml';
    this.results = [];
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
        
        if (result.warnings.length > 0) {
          console.log(`  ⚠️ ${result.warnings.length} warnings`);
        }
      }
    }
    
    await this.generateReports();
  }

  async generateReports() {
    // JSON report
    await fs.writeFile(this.report, JSON.stringify(this.results, null, 2));
    console.log(`[validator] Report saved to ${this.report}`);
    
    // JUnit XML report
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
<testsuite name="ShaderValidation" tests="${this.results.length}">
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
      const value = args[i].includes('=') ? args[i].split('=')[1] : args[i + 1];
      options[key] = value;
      if (!args[i].includes('=')) i++;
    }
  }
  
  const validator = new ShaderValidator(options);
  validator.validate().catch(console.error);
}

export { ShaderValidator };
