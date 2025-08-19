// virgil_summon.mjs
// "Summon Virgil" - The guide through the circles of shader development
// Generates all scaffolding and helper scripts for shader validation
// Like Virgil guided Dante, this guides you through shader pipeline setup

import { promises as fs } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

console.log(`
╔════════════════════════════════════════════════════════════╗
║                     SUMMONING VIRGIL                       ║
║         "Through me the way to the suffering city"        ║
║                                                            ║
║   Creating your guide through the shader pipeline...      ║
╚════════════════════════════════════════════════════════════╝
`);

// File 1: copy_canonical_to_public.mjs
// Intent: Maintains a canonical source of truth for shaders and syncs to public
const copyCanonicalToPublic = `// copy_canonical_to_public.mjs
// Copies canonical shader files from lib/webgpu/shaders to public/hybrid/wgsl
// Maintains single source of truth for all shaders

import { promises as fs } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const CANONICAL_DIR = path.join(__dirname, '../../frontend/lib/webgpu/shaders');
const PUBLIC_DIR = path.join(__dirname, '../../frontend/public/hybrid/wgsl');

async function ensureDir(dir) {
  try {
    await fs.mkdir(dir, { recursive: true });
  } catch (err) {
    // Directory exists
  }
}

async function copyShaders() {
  console.log('[copy] Starting canonical → public sync...');
  
  await ensureDir(PUBLIC_DIR);
  
  const files = await fs.readdir(CANONICAL_DIR);
  const wgslFiles = files.filter(f => f.endsWith('.wgsl'));
  
  for (const file of wgslFiles) {
    const src = path.join(CANONICAL_DIR, file);
    const dest = path.join(PUBLIC_DIR, file);
    
    await fs.copyFile(src, dest);
    console.log(\`[copy] \${file}\`);
  }
  
  console.log(\`[copy] Done → \${PUBLIC_DIR}\`);
  return wgslFiles.length;
}

if (import.meta.url === \`file://\${process.argv[1]}\`) {
  copyShaders().catch(console.error);
}

export { copyShaders };
`;

// File 2: validate_and_report.mjs
// Intent: Comprehensive validation with device limits and multi-target support
const validateAndReport = `// validate_and_report.mjs
// Validates WGSL shaders against device limits and multiple compilation targets
// Generates detailed reports in multiple formats (JSON, JUnit, console)

import { spawn } from 'child_process';
import { promises as fs } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

class ShaderValidator {
  constructor(options = {}) {
    this.dir = options.dir || 'frontend/lib/webgpu/shaders';
    this.limits = options.limits || null;
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
          error: \`\${target} not found\`
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
    
    console.log(\`[validator] Found \${wgslFiles.length} WGSL files\`);
    
    for (const file of wgslFiles) {
      const filePath = path.join(fullDir, file);
      console.log(\`[validator] Checking \${file}...\`);
      
      for (const target of this.targets) {
        const result = await this.validateFile(filePath, target);
        this.results.push(result);
        
        if (result.success) {
          console.log(\`  ✅ \${target}: Valid\`);
        } else if (result.error) {
          console.log(\`  ⏭️ \${target}: \${result.error}\`);
        } else {
          console.log(\`  ❌ \${target}: Failed\`);
          if (this.strict) {
            throw new Error(\`Validation failed for \${file}\`);
          }
        }
        
        if (result.warnings.length > 0) {
          console.log(\`  ⚠️ \${result.warnings.length} warnings\`);
        }
      }
    }
    
    await this.generateReports();
  }

  async generateReports() {
    // JSON report
    await fs.writeFile(this.report, JSON.stringify(this.results, null, 2));
    console.log(\`[validator] Report saved to \${this.report}\`);
    
    // JUnit XML report
    const junit = this.generateJUnit();
    await fs.writeFile(this.junit, junit);
    console.log(\`[validator] JUnit report saved to \${this.junit}\`);
    
    // Console summary
    const failed = this.results.filter(r => !r.success && !r.error);
    const warnings = this.results.reduce((sum, r) => sum + (r.warnings?.length || 0), 0);
    
    console.log('\\n=== VALIDATION SUMMARY ===');
    console.log(\`Total: \${this.results.length}\`);
    console.log(\`Failed: \${failed.length}\`);
    console.log(\`Warnings: \${warnings}\`);
  }

  generateJUnit() {
    const testcases = this.results.map(r => {
      const status = r.success ? 'passed' : r.error ? 'skipped' : 'failed';
      return \`    <testcase name="\${r.file}" classname="\${r.target}" status="\${status}">
      \${r.stderr ? \`<failure>\${r.stderr}</failure>\` : ''}
      \${r.warnings?.length ? \`<system-out>Warnings: \${r.warnings.join(', ')}</system-out>\` : ''}
    </testcase>\`;
    }).join('\\n');
    
    return \`<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="ShaderValidation" tests="\${this.results.length}">
\${testcases}
</testsuite>\`;
  }
}

// CLI interface
if (import.meta.url === \`file://\${process.argv[1]}\`) {
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
`;

// File 3: guards/check_uniform_arrays.mjs
// Intent: Prevent mismatched array sizes between CPU and GPU
const checkUniformArrays = `// guards/check_uniform_arrays.mjs
// Guards against uniform array size mismatches between host and shader code
// Prevents the "256 vs 32" type bugs

import { promises as fs } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const SHADER_DIR = path.join(__dirname, '../../../frontend/lib/webgpu/shaders');
const HOST_DIR = path.join(__dirname, '../../../frontend/lib/webgpu');

class UniformArrayChecker {
  constructor() {
    this.constants = new Map();
    this.usages = [];
    this.mismatches = [];
  }

  async scanShaders() {
    const files = await fs.readdir(SHADER_DIR);
    const wgslFiles = files.filter(f => f.endsWith('.wgsl'));
    
    for (const file of wgslFiles) {
      const content = await fs.readFile(path.join(SHADER_DIR, file), 'utf8');
      this.extractConstants(content, file);
      this.extractUsages(content, file);
    }
  }

  extractConstants(content, file) {
    // Find const declarations like: const MAX_OSCILLATORS: u32 = 32u;
    const constPattern = /const\\s+(\\w+)\\s*:\\s*u32\\s*=\\s*(\\d+)u?/g;
    
    for (const match of content.matchAll(constPattern)) {
      const [, name, value] = match;
      this.constants.set(name, {
        value: parseInt(value),
        file,
        type: 'shader_const'
      });
    }
    
    // Find array declarations to infer sizes
    const arrayPattern = /array<[^,]+,\\s*(\\w+|\\d+)>/g;
    
    for (const match of content.matchAll(arrayPattern)) {
      const [, size] = match;
      if (!/\\d/.test(size)) {
        // It's a constant name, not a literal
        this.usages.push({
          file,
          constant: size,
          context: match[0]
        });
      }
    }
  }

  extractUsages(content, file) {
    // Find clamp_index_dyn or similar with hardcoded values
    const clampPattern = /clamp_index_dyn?\\([^,]+,\\s*(\\d+)u?\\)/g;
    
    for (const match of content.matchAll(clampPattern)) {
      const [full, size] = match;
      const value = parseInt(size);
      
      // Check if this value matches known constants
      for (const [name, info] of this.constants) {
        if (info.value !== value && value !== 0) {
          this.mismatches.push({
            file,
            issue: \`Hardcoded \${value} instead of \${name} (\${info.value})\`,
            line: full
          });
        }
      }
    }
  }

  async scanHostCode() {
    // Look for TypeScript/JavaScript files that create buffers
    const files = await this.findFiles(HOST_DIR, /\\.(ts|js|mjs)$/);
    
    for (const file of files) {
      const content = await fs.readFile(file, 'utf8');
      
      // Find buffer size declarations
      const bufferPattern = /new\\s+Float32Array\\((\\d+)\\)/g;
      const sizePattern = /\\.size\\s*=\\s*(\\d+)/g;
      
      for (const match of content.matchAll(bufferPattern)) {
        const [, size] = match;
        this.checkAgainstConstants(file, parseInt(size));
      }
    }
  }

  checkAgainstConstants(file, size) {
    // Common sizes that should match shader constants
    const knownSizes = {
      32: 'MAX_OSCILLATORS',
      256: 'WORKGROUP_SIZE * WORKGROUP_SIZE',
      1024: 'HOLOGRAM_SIZE'
    };
    
    if (knownSizes[size]) {
      console.log(\`[info] Host code uses size \${size} (\${knownSizes[size]}) in \${path.basename(file)}\`);
    }
  }

  async findFiles(dir, pattern) {
    const results = [];
    const entries = await fs.readdir(dir, { withFileTypes: true });
    
    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory() && !entry.name.includes('node_modules')) {
        results.push(...await this.findFiles(fullPath, pattern));
      } else if (pattern.test(entry.name)) {
        results.push(fullPath);
      }
    }
    
    return results;
  }

  report() {
    console.log('\\n🔍 Uniform Array Size Check Report\\n');
    
    if (this.mismatches.length === 0) {
      console.log('✅ No array size mismatches found!');
    } else {
      console.log('⚠️ Found potential mismatches:\\n');
      for (const mismatch of this.mismatches) {
        console.log(\`  \${mismatch.file}:\`);
        console.log(\`    Issue: \${mismatch.issue}\`);
        console.log(\`    Line: \${mismatch.line}\\n\`);
      }
    }
    
    console.log('\\n📊 Constants found:');
    for (const [name, info] of this.constants) {
      console.log(\`  \${name} = \${info.value} (\${info.file})\`);
    }
    
    return this.mismatches.length === 0;
  }
}

// Run the check
if (import.meta.url === \`file://\${process.argv[1]}\`) {
  const checker = new UniformArrayChecker();
  
  checker.scanShaders()
    .then(() => checker.scanHostCode())
    .then(() => {
      const success = checker.report();
      process.exit(success ? 0 : 1);
    })
    .catch(console.error);
}

export { UniformArrayChecker };
`;

// Create the files
async function summonVirgil() {
  const files = [
    { 
      path: path.join(__dirname, 'copy_canonical_to_public.mjs'),
      content: copyCanonicalToPublic,
      desc: 'Canonical → Public sync tool'
    },
    { 
      path: path.join(__dirname, 'validate_and_report.mjs'),
      content: validateAndReport,
      desc: 'Multi-target validation & reporting'
    },
    { 
      path: path.join(__dirname, 'guards', 'check_uniform_arrays.mjs'),
      content: checkUniformArrays,
      desc: 'Array size mismatch guard'
    }
  ];

  // Create guards directory
  await fs.mkdir(path.join(__dirname, 'guards'), { recursive: true });

  console.log('\n📜 Virgil speaks: "I shall be your guide..."\n');

  for (const file of files) {
    await fs.writeFile(file.path, file.content);
    console.log(`  ✨ Summoned: ${path.basename(file.path)}`);
    console.log(`     Purpose: ${file.desc}\n`);
  }

  // Create package.json scripts
  const packageJsonScripts = `
  "shaders:sync": "node tools/shaders/copy_canonical_to_public.mjs",
  "shaders:validate": "node tools/shaders/validate_and_report.mjs --dir=frontend/lib/webgpu/shaders",
  "shaders:gate": "node tools/shaders/validate_and_report.mjs --strict",
  "shaders:gate:iphone": "node tools/shaders/validate_and_report.mjs --dir=frontend/lib/webgpu/shaders --limits=tools/shaders/device_limits/iphone15.json --targets=naga --strict",
  "shaders:guard": "node tools/shaders/guards/check_uniform_arrays.mjs",
  "shaders:summon": "node tools/shaders/virgil_summon.mjs"`;

  console.log('📦 Add these to package.json scripts:');
  console.log(packageJsonScripts);

  console.log(`
╔════════════════════════════════════════════════════════════╗
║                    VIRGIL HAS BEEN SUMMONED                ║
║                                                            ║
║  "Through me the way to the shaders' blessed city,        ║
║   Through me the way to eternal validation,               ║
║   Through me the way among the lost pipelines."           ║
║                                                            ║
║  Your guide awaits. Use wisely:                          ║
║    npm run shaders:sync     → Sync canonical to public   ║
║    npm run shaders:validate → Validate all shaders       ║
║    npm run shaders:gate     → Strict validation gate     ║
║    npm run shaders:guard    → Check array sizes          ║
╚════════════════════════════════════════════════════════════╝
  `);
}

// Summon!
summonVirgil().catch(console.error);
