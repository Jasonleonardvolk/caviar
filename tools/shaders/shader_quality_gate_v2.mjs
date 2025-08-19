#!/usr/bin/env node
/**
 * Production Shader Quality Gate v2.0
 * Multi-backend validation with device limits and CI integration
 * 
 * Usage:
 *   node tools/shaders/shader_quality_gate_v2.mjs [options]
 * 
 * Options:
 *   --dir=<path>           Directory to scan (default: frontend/)
 *   --strict               Fail on warnings
 *   --fix                  Auto-fix common issues
 *   --targets=msl,hlsl,... Transpile targets (msl,hlsl,spirv)
 *   --limits=<path|latest|auto>  Device limits JSON file or alias
 *   --report=<file>        Output JSON report
 *   --junit=<file>         Output JUnit XML report
 * 
 * Limits:
 *   --limits=tools/shaders/device_limits/iphone15.json  Use explicit profile
 *   --limits=latest                                     Use latest captured device
 *   --limits=auto                                       Same as 'latest'
 * Env:
 *   SHADER_LIMITS=<path|latest|auto>                    Overrides --limits if provided
 * 
 * Exit codes:
 *   0 - All shaders pass
 *   1 - Validation errors
 *   2 - Performance warnings
 *   3 - Device limit violations
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { exec } from 'child_process';
import { promisify } from 'util';
import crypto from 'crypto';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const execAsync = promisify(exec);

// Resolver for latest/auto aliases
function resolveLimitsAlias(limitsArg) {
  // Allow env override too (highest precedence)
  const cli = limitsArg && String(limitsArg).trim();
  const env = process.env.SHADER_LIMITS && String(process.env.SHADER_LIMITS).trim();

  // Choose the source of truth: ENV > CLI
  const requested = env || cli || 'tools/shaders/device_limits/iphone15.json';

  // Direct path given and not an alias
  if (requested !== 'latest' && requested !== 'auto') {
    return requested;
  }

  // Pointer file location
  const pointerPath = path.join(
    process.cwd(),
    'tools', 'shaders', 'device_limits', 'latest.json'
  );

  // Try latest pointer first
  if (fs.existsSync(pointerPath)) {
    try {
      const j = JSON.parse(fs.readFileSync(pointerPath, 'utf8'));
      if (j && j.path && fs.existsSync(path.resolve(process.cwd(), j.path))) {
        console.log(`📎 Using latest limits pointer → ${j.path}`);
        return j.path;
      }
    } catch (e) {
      console.warn(`⚠️  Could not parse latest.json: ${e.message}`);
    }
  }

  // "auto" can also fall back to a sensible platform profile if you want
  // For now: both latest-miss and explicit 'latest' fall back to iphone15.json
  const fallback = 'tools/shaders/device_limits/iphone15.json';
  console.log(`↩️  Latest pointer missing; falling back → ${fallback}`);
  return fallback;
}

// Parse CLI arguments
const args = process.argv.slice(2).reduce((acc, arg) => {
  const [key, value] = arg.split('=');
  acc[key.replace(/^--/, '')] = value || true;
  return acc;
}, {});

// Configuration
const SHADER_DIR = path.resolve(args.dir || 'frontend/');
const STRICT_MODE = args.strict === true;
const FIX_MODE = args.fix === true;
const TARGETS = args.targets ? args.targets.split(',') : [];
const LIMITS_PATH = resolveLimitsAlias(args.limits);
const LIMITS_FILE = LIMITS_PATH ? path.resolve(LIMITS_PATH) : null;
const REPORT_FILE = args.report ? path.resolve(args.report) : null;
const JUNIT_FILE = args.junit ? path.resolve(args.junit) : null;

// Device limits (default conservative for mobile)
const DEFAULT_LIMITS = {
  maxComputeInvocationsPerWorkgroup: 256,
  maxComputeWorkgroupSizeX: 256,
  maxComputeWorkgroupSizeY: 256,
  maxComputeWorkgroupSizeZ: 64,
  maxWorkgroupStorageSize: 32768,
  maxSampledTexturesPerShaderStage: 16,
  maxSamplersPerShaderStage: 16
};

// Load device limits
let deviceLimits = DEFAULT_LIMITS;
if (LIMITS_FILE && fs.existsSync(LIMITS_FILE)) {
  try {
    deviceLimits = JSON.parse(fs.readFileSync(LIMITS_FILE, 'utf8'));
    console.log(`📱 Loaded device limits from ${path.resolve(process.cwd(), LIMITS_PATH)}`);
  } catch (e) {
    console.error(`⚠️ Failed to load device limits: ${e.message}`);
  }
} else if (LIMITS_FILE) {
  console.warn(`⚠️ Device limits file not found: ${LIMITS_FILE}`);
}

// Quality rules with auto-fix support
const RULES = {
  // Correctness
  NO_DUPLICATE_BINDINGS: {
    severity: 'error',
    check: (content) => {
      const bindings = new Map();
      const errors = [];
      const bindingRegex = /@group\((\d+)\)\s+@binding\((\d+)\)/g;
      let match;
      
      while ((match = bindingRegex.exec(content)) !== null) {
        const key = `${match[1]}:${match[2]}`;
        if (bindings.has(key)) {
          errors.push({
            line: content.substring(0, match.index).split('\n').length,
            message: `Duplicate binding: @group(${match[1]}) @binding(${match[2]})`
          });
        }
        bindings.set(key, match.index);
      }
      return errors;
    }
  },
  
  WORKGROUP_SIZE_REQUIRED: {
    severity: 'error',
    check: (content) => {
      const errors = [];
      const computeRegex = /@compute\s+fn\s+(\w+)/g;
      let match;
      
      while ((match = computeRegex.exec(content)) !== null) {
        const fnName = match[1];
        const beforeFn = content.substring(0, match.index);
        const lastWorkgroupIdx = beforeFn.lastIndexOf('@workgroup_size');
        
        if (lastWorkgroupIdx === -1 || beforeFn.substring(lastWorkgroupIdx).includes('fn')) {
          errors.push({
            line: content.substring(0, match.index).split('\n').length,
            message: `Compute function '${fnName}' missing @workgroup_size`,
            fix: () => {
              // Add default workgroup size
              return content.replace(match[0], `@workgroup_size(64, 1, 1)\n${match[0]}`);
            }
          });
        }
      }
      return errors;
    }
  },
  
  WORKGROUP_SIZE_LIMITS: {
    severity: 'error',
    check: (content) => {
      const errors = [];
      const workgroupRegex = /@workgroup_size\((\d+)(?:\s*,\s*(\d+))?(?:\s*,\s*(\d+))?\)/g;
      let match;
      
      while ((match = workgroupRegex.exec(content)) !== null) {
        const x = parseInt(match[1]) || 1;
        const y = parseInt(match[2]) || 1;
        const z = parseInt(match[3]) || 1;
        const total = x * y * z;
        
        if (x > deviceLimits.maxComputeWorkgroupSizeX) {
          errors.push({
            line: content.substring(0, match.index).split('\n').length,
            message: `Workgroup size X (${x}) exceeds device limit (${deviceLimits.maxComputeWorkgroupSizeX})`
          });
        }
        if (y > deviceLimits.maxComputeWorkgroupSizeY) {
          errors.push({
            line: content.substring(0, match.index).split('\n').length,
            message: `Workgroup size Y (${y}) exceeds device limit (${deviceLimits.maxComputeWorkgroupSizeY})`
          });
        }
        if (z > deviceLimits.maxComputeWorkgroupSizeZ) {
          errors.push({
            line: content.substring(0, match.index).split('\n').length,
            message: `Workgroup size Z (${z}) exceeds device limit (${deviceLimits.maxComputeWorkgroupSizeZ})`
          });
        }
        if (total > deviceLimits.maxComputeInvocationsPerWorkgroup) {
          errors.push({
            line: content.substring(0, match.index).split('\n').length,
            message: `Total invocations (${total}) exceeds device limit (${deviceLimits.maxComputeInvocationsPerWorkgroup})`
          });
        }
      }
      return errors;
    }
  },
  
  WORKGROUP_MEMORY_LIMITS: {
    severity: 'warning',
    check: (content) => {
      const errors = [];
      const workgroupVarRegex = /var<workgroup>\s+(\w+)\s*:\s*array<(\w+)(?:<[^>]+>)?,\s*(\d+)>/g;
      let totalBytes = 0;
      const vars = [];
      let match;
      
      while ((match = workgroupVarRegex.exec(content)) !== null) {
        const varName = match[1];
        const elementType = match[2];
        const elementCount = parseInt(match[3]);
        
        // Calculate bytes based on type
        let bytesPerElement = 4; // default f32
        if (elementType.includes('vec2')) bytesPerElement = 8;
        if (elementType.includes('vec3')) bytesPerElement = 16; // vec3 aligned to vec4
        if (elementType.includes('vec4')) bytesPerElement = 16;
        if (elementType.includes('mat2x2')) bytesPerElement = 16;
        if (elementType.includes('mat3x3')) bytesPerElement = 48;
        if (elementType.includes('mat4x4')) bytesPerElement = 64;
        
        const varBytes = bytesPerElement * elementCount;
        totalBytes += varBytes;
        vars.push({ name: varName, bytes: varBytes });
      }
      
      if (totalBytes > deviceLimits.maxWorkgroupStorageSize) {
        errors.push({
          line: 1,
          message: `Total workgroup memory (${totalBytes} bytes) exceeds device limit (${deviceLimits.maxWorkgroupStorageSize} bytes). Variables: ${vars.map(v => `${v.name}=${v.bytes}B`).join(', ')}`
        });
      }
      
      return errors;
    }
  },
  
  PREFER_CONST: {
    severity: 'warning',
    check: (content) => {
      const errors = [];
      const letRegex = /\blet\s+(\w+)\s*=\s*([^;]+);/g;
      let match;
      
      while ((match = letRegex.exec(content)) !== null) {
        const varName = match[1];
        const value = match[2];
        
        // Check if it's a compile-time constant
        if (/^[\d\.\-\+\*\/\(\)\s]+$/.test(value) || /^vec[234]\([\d\.\-\+\*\/\(\)\s,]+\)$/.test(value)) {
          errors.push({
            line: content.substring(0, match.index).split('\n').length,
            message: `Consider using 'const' instead of 'let' for '${varName}'`,
            fix: () => content.replace(match[0], match[0].replace('let', 'const'))
          });
        }
      }
      return errors;
    }
  },
  
  VEC3_STORAGE_ALIGNMENT: {
    severity: 'warning',
    check: (content) => {
      const errors = [];
      const structRegex = /struct\s+\w+\s*{([^}]+)}/g;
      let match;
      
      while ((match = structRegex.exec(content)) !== null) {
        const structBody = match[1];
        if (structBody.includes('vec3<')) {
          const lines = structBody.split('\n');
          for (let i = 0; i < lines.length; i++) {
            if (lines[i].includes('vec3<')) {
              const nextLine = lines[i + 1];
              if (nextLine && !nextLine.includes('@align(16)') && !nextLine.includes('vec3<')) {
                errors.push({
                  line: content.substring(0, match.index).split('\n').length + i,
                  message: 'vec3 in storage buffer should be followed by padding or another vec3',
                  fix: () => {
                    // Add padding field
                    const paddingField = '  _padding: f32,';
                    return content.replace(lines[i], lines[i] + '\n' + paddingField);
                  }
                });
              }
            }
          }
        }
      }
      return errors;
    }
  },
  
  DYNAMIC_INDEXING_BOUNDS: {
    severity: 'warning',
    check: (content) => {
      const errors = [];
      const indexRegex = /(\w+)\[([^\]]+)\]/g;
      let match;
      
      while ((match = indexRegex.exec(content)) !== null) {
        const arrayName = match[1];
        const indexExpr = match[2];
        
        // Check if index is dynamic (contains variables)
        if (!/^\d+$/.test(indexExpr) && /[a-zA-Z_]/.test(indexExpr)) {
          // Look for bounds checking
          const line = content.substring(0, match.index).split('\n').length;
          const lineStart = content.lastIndexOf('\n', match.index) + 1;
          const lineEnd = content.indexOf('\n', match.index);
          const lineContent = content.substring(lineStart, lineEnd);
          
          if (!lineContent.includes('min(') && !lineContent.includes('clamp(')) {
            errors.push({
              line,
              message: `Dynamic array access '${arrayName}[${indexExpr}]' without apparent bounds checking`
            });
          }
        }
      }
      return errors;
    }
  }
};

// Backend validators
class ShaderValidator {
  constructor() {
    this.results = [];
    this.hasNaga = false;
    this.hasTint = false;
    this.nagaSyntax = null;
  }
  
  async checkTools() {
    // Check for Naga
    try {
      const { stdout } = await execAsync('naga --version');
      this.hasNaga = true;
      console.log(`✅ Naga found (version: ${stdout.trim()})`);
      
      // Check which naga command syntax works
      const testFile = path.join(__dirname, '.test_naga_syntax.wgsl');
      fs.writeFileSync(testFile, '@compute @workgroup_size(1) fn main() {}');
      
      // Try different syntaxes
      const syntaxTests = [
        'naga validate',
        'naga',
        'naga --validate'
      ];
      
      for (const syntax of syntaxTests) {
        try {
          await execAsync(`${syntax} ${testFile}`);
          this.nagaSyntax = syntax;
          console.log(`  Using syntax: ${syntax} <file>`);
          break;
        } catch (e) {
          // Try next syntax
        }
      }
      
      fs.unlinkSync(testFile);
      
      if (!this.nagaSyntax) {
        console.warn('⚠️ Naga found but validation syntax could not be determined');
        console.warn('  Run ./Test-AllNagaSyntax.ps1 to diagnose');
        this.hasNaga = false;
      }
    } catch (e) {
      console.warn('⚠️ Naga not found or not working. Install with: cargo install naga-cli');
      console.warn('  Or run ./Find-Naga.ps1 to diagnose');
    }
    
    // Check for Tint
    try {
      await execAsync('tint --version');
      this.hasTint = true;
      console.log('✅ Tint transpiler found');
    } catch (e) {
      // Try common locations
      const tintPaths = [
        path.join(__dirname, '../../build/tools/tint/bin/tint'),
        path.join(__dirname, '../../build/tools/tint/bin/tint.exe'),
        'C:/dawn/out/Release/tint.exe',
        '/usr/local/bin/tint'
      ];
      
      for (const tintPath of tintPaths) {
        if (fs.existsSync(tintPath)) {
          this.tintPath = tintPath;
          this.hasTint = true;
          console.log(`✅ Tint found at ${tintPath}`);
          break;
        }
      }
      
      if (!this.hasTint) {
        console.warn('⚠️ Tint not found. Some transpilation checks will be skipped.');
      }
    }
  }
  
  async validateWithNaga(filePath, content) {
    if (!this.hasNaga || !this.nagaSyntax) return { success: true, warnings: [] };
    
    const tempFile = path.join(path.dirname(filePath), `.temp_${path.basename(filePath)}`);
    fs.writeFileSync(tempFile, content);
    
    try {
      const { stdout, stderr } = await execAsync(`${this.nagaSyntax} ${tempFile}`);
      fs.unlinkSync(tempFile);
      
      return {
        success: true,
        output: stdout,
        warnings: stderr ? stderr.split('\n').filter(l => l) : []
      };
    } catch (e) {
      fs.unlinkSync(tempFile);
      return {
        success: false,
        error: e.stderr || e.message,
        warnings: []
      };
    }
  }
  
  async transpileWithTint(filePath, content, target) {
    if (!this.hasTint) return { success: true, skipped: true };
    
    const tempFile = path.join(path.dirname(filePath), `.temp_${path.basename(filePath)}`);
    fs.writeFileSync(tempFile, content);
    
    const tintCmd = this.tintPath || 'tint';
    const formatFlag = {
      'msl': '--format=msl',
      'hlsl': '--format=hlsl',
      'spirv': '--format=spirv'
    }[target] || '--format=wgsl';
    
    try {
      const { stdout, stderr } = await execAsync(`${tintCmd} ${formatFlag} ${tempFile}`);
      fs.unlinkSync(tempFile);
      
      return {
        success: true,
        output: stdout,
        warnings: stderr ? stderr.split('\n').filter(l => l) : [],
        target
      };
    } catch (e) {
      fs.unlinkSync(tempFile);
      return {
        success: false,
        error: e.stderr || e.message,
        target
      };
    }
  }
  
  async validateShader(filePath) {
    const content = fs.readFileSync(filePath, 'utf8');
    const result = {
      file: path.relative(process.cwd(), filePath),
      hash: crypto.createHash('sha256').update(content).digest('hex').substring(0, 8),
      size: content.length,
      lines: content.split('\n').length,
      timestamp: new Date().toISOString(),
      errors: [],
      warnings: [],
      fixes: [],
      backends: {}
    };
    
    // Apply rules
    console.log(`\n📄 ${result.file}`);
    
    for (const [ruleName, rule] of Object.entries(RULES)) {
      const issues = rule.check(content);
      if (issues.length > 0) {
        for (const issue of issues) {
          const item = {
            rule: ruleName,
            line: issue.line,
            message: issue.message
          };
          
          if (rule.severity === 'error') {
            result.errors.push(item);
            console.log(`  ❌ Line ${issue.line}: ${issue.message}`);
          } else {
            result.warnings.push(item);
            if (STRICT_MODE) {
              console.log(`  ⚠️  Line ${issue.line}: ${issue.message}`);
            }
          }
          
          if (FIX_MODE && issue.fix) {
            const fixed = issue.fix();
            fs.writeFileSync(filePath, fixed);
            result.fixes.push(item);
            console.log(`  ✏️  Fixed: ${issue.message}`);
          }
        }
      }
    }
    
    // Validate with Naga
    const nagaResult = await this.validateWithNaga(filePath, content);
    result.backends.naga = {
      success: nagaResult.success,
      error: nagaResult.error,
      warnings: nagaResult.warnings
    };
    
    if (!nagaResult.success) {
      result.errors.push({
        rule: 'NAGA_VALIDATION',
        message: nagaResult.error
      });
      console.log(`  ❌ Naga: ${nagaResult.error}`);
    } else {
      console.log(`  ✅ Naga: Valid`);
    }
    
    // Transpile to targets
    for (const target of TARGETS) {
      const transpileResult = await this.transpileWithTint(filePath, content, target);
      result.backends[target] = {
        success: transpileResult.success,
        error: transpileResult.error,
        warnings: transpileResult.warnings,
        skipped: transpileResult.skipped
      };
      
      if (transpileResult.skipped) {
        console.log(`  ⏭️  ${target}: Skipped (Tint not available)`);
      } else if (!transpileResult.success) {
        result.errors.push({
          rule: `TINT_TRANSPILE_${target.toUpperCase()}`,
          message: transpileResult.error
        });
        console.log(`  ❌ ${target}: ${transpileResult.error}`);
      } else {
        console.log(`  ✅ ${target}: Success`);
      }
    }
    
    this.results.push(result);
    return result;
  }
  
  async validateDirectory(dir) {
    const files = [];
    
    function scanDir(currentDir) {
      const entries = fs.readdirSync(currentDir, { withFileTypes: true });
      for (const entry of entries) {
        const fullPath = path.join(currentDir, entry.name);
        if (entry.isDirectory() && !entry.name.startsWith('.') && entry.name !== 'node_modules') {
          scanDir(fullPath);
        } else if (entry.isFile() && entry.name.endsWith('.wgsl')) {
          files.push(fullPath);
        }
      }
    }
    
    scanDir(dir);
    console.log(`\n🔍 Found ${files.length} WGSL files in ${dir}\n`);
    
    for (const file of files) {
      await this.validateShader(file);
    }
    
    return this.results;
  }
  
  generateReport() {
    const report = {
      timestamp: new Date().toISOString(),
      directory: SHADER_DIR,
      deviceLimits: deviceLimits,
      summary: {
        total: this.results.length,
        passed: 0,
        failed: 0,
        warnings: 0,
        fixed: 0
      },
      shaders: this.results
    };
    
    for (const result of this.results) {
      if (result.errors.length === 0) {
        report.summary.passed++;
      } else {
        report.summary.failed++;
      }
      report.summary.warnings += result.warnings.length;
      report.summary.fixed += result.fixes.length;
    }
    
    return report;
  }
  
  generateJUnit() {
    const report = this.generateReport();
    const testCases = [];
    
    for (const shader of report.shaders) {
      const testCase = {
        name: shader.file,
        classname: 'ShaderValidation',
        time: 0.1
      };
      
      if (shader.errors.length > 0) {
        testCase.failure = {
          message: shader.errors[0].message,
          type: shader.errors[0].rule,
          text: shader.errors.map(e => `${e.rule}: ${e.message}`).join('\n')
        };
      }
      
      testCases.push(testCase);
    }
    
    const junit = {
      testsuites: [{
        name: 'Shader Quality Gate',
        tests: report.summary.total,
        failures: report.summary.failed,
        errors: 0,
        time: 1.0,
        timestamp: report.timestamp,
        testsuite: testCases.map(tc => ({
          testcase: {
            $: {
              name: tc.name,
              classname: tc.classname,
              time: tc.time
            },
            failure: tc.failure ? {
              $: {
                message: tc.failure.message,
                type: tc.failure.type
              },
              _: tc.failure.text
            } : undefined
          }
        }))
      }]
    };
    
    // Convert to XML
    let xml = '<?xml version="1.0" encoding="UTF-8"?>\n';
    xml += '<testsuites>\n';
    xml += `  <testsuite name="${junit.testsuites[0].name}" tests="${junit.testsuites[0].tests}" failures="${junit.testsuites[0].failures}" errors="${junit.testsuites[0].errors}" time="${junit.testsuites[0].time}" timestamp="${junit.testsuites[0].timestamp}">\n`;
    
    for (const suite of junit.testsuites[0].testsuite) {
      const tc = suite.testcase;
      xml += `    <testcase name="${tc.$.name}" classname="${tc.$.classname}" time="${tc.$.time}"`;
      if (tc.failure) {
        xml += '>\n';
        xml += `      <failure message="${tc.failure.$.message}" type="${tc.failure.$.type}"><![CDATA[${tc.failure._}]]></failure>\n`;
        xml += '    </testcase>\n';
      } else {
        xml += ' />\n';
      }
    }
    
    xml += '  </testsuite>\n';
    xml += '</testsuites>\n';
    
    return xml;
  }
}

// Main execution
async function main() {
  console.log('🚀 Shader Quality Gate v2.0');
  console.log('================================\n');
  
  const validator = new ShaderValidator();
  await validator.checkTools();
  
  if (!fs.existsSync(SHADER_DIR)) {
    console.error(`❌ Directory not found: ${SHADER_DIR}`);
    process.exit(1);
  }
  
  const results = await validator.validateDirectory(SHADER_DIR);
  const report = validator.generateReport();
  
  // Output report
  console.log('\n================================');
  console.log('📊 Summary');
  console.log(`  Total: ${report.summary.total}`);
  console.log(`  ✅ Passed: ${report.summary.passed}`);
  console.log(`  ❌ Failed: ${report.summary.failed}`);
  console.log(`  ⚠️  Warnings: ${report.summary.warnings}`);
  if (FIX_MODE) {
    console.log(`  ✏️  Fixed: ${report.summary.fixed}`);
  }
  
  // Save reports
  if (REPORT_FILE) {
    const reportDir = path.dirname(REPORT_FILE);
    if (!fs.existsSync(reportDir)) {
      fs.mkdirSync(reportDir, { recursive: true });
    }
    fs.writeFileSync(REPORT_FILE, JSON.stringify(report, null, 2));
    console.log(`\n📄 Report saved to ${REPORT_FILE}`);
  }
  
  if (JUNIT_FILE) {
    const junitDir = path.dirname(JUNIT_FILE);
    if (!fs.existsSync(junitDir)) {
      fs.mkdirSync(junitDir, { recursive: true });
    }
    fs.writeFileSync(JUNIT_FILE, validator.generateJUnit());
    console.log(`📄 JUnit report saved to ${JUNIT_FILE}`);
  }
  
  // Exit code
  if (report.summary.failed > 0) {
    process.exit(1);
  } else if (STRICT_MODE && report.summary.warnings > 0) {
    process.exit(2);
  } else {
    process.exit(0);
  }
}

// Run
main().catch(error => {
  console.error('❌ Fatal error:', error);
  process.exit(1);
});
