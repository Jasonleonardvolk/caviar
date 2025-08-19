// guards/check_uniform_arrays.mjs
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
    const constPattern = /const\s+(\w+)\s*:\s*u32\s*=\s*(\d+)u?/g;
    
    for (const match of content.matchAll(constPattern)) {
      const [, name, value] = match;
      this.constants.set(name, {
        value: parseInt(value),
        file,
        type: 'shader_const'
      });
    }
    
    // Find array declarations to infer sizes
    const arrayPattern = /array<[^,]+,\s*(\w+|\d+)>/g;
    
    for (const match of content.matchAll(arrayPattern)) {
      const [, size] = match;
      if (!/\d/.test(size)) {
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
    const clampPattern = /clamp_index_dyn?\([^,]+,\s*(\d+)u?\)/g;
    
    for (const match of content.matchAll(clampPattern)) {
      const [full, size] = match;
      const value = parseInt(size);
      
      // Check if this value matches known constants
      for (const [name, info] of this.constants) {
        if (info.value !== value && value !== 0) {
          this.mismatches.push({
            file,
            issue: `Hardcoded ${value} instead of ${name} (${info.value})`,
            line: full
          });
        }
      }
    }
  }

  async scanHostCode() {
    // Look for TypeScript/JavaScript files that create buffers
    const files = await this.findFiles(HOST_DIR, /\.(ts|js|mjs)$/);
    
    for (const file of files) {
      const content = await fs.readFile(file, 'utf8');
      
      // Find buffer size declarations
      const bufferPattern = /new\s+Float32Array\((\d+)\)/g;
      const sizePattern = /\.size\s*=\s*(\d+)/g;
      
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
      console.log(`[info] Host code uses size ${size} (${knownSizes[size]}) in ${path.basename(file)}`);
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
    console.log('\n🔍 Uniform Array Size Check Report\n');
    
    if (this.mismatches.length === 0) {
      console.log('✅ No array size mismatches found!');
    } else {
      console.log('⚠️ Found potential mismatches:\n');
      for (const mismatch of this.mismatches) {
        console.log(`  ${mismatch.file}:`);
        console.log(`    Issue: ${mismatch.issue}`);
        console.log(`    Line: ${mismatch.line}\n`);
      }
    }
    
    console.log('\n📊 Constants found:');
    for (const [name, info] of this.constants) {
      console.log(`  ${name} = ${info.value} (${info.file})`);
    }
    
    return this.mismatches.length === 0;
  }
}

// Run the check
if (import.meta.url === `file://${process.argv[1]}`) {
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
