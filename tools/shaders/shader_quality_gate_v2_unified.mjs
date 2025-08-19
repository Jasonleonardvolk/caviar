#!/usr/bin/env node
/**
 * Production Shader Quality Gate v2.1
 * Multi-backend validation with device limits and CI integration
 * Now with shared validation library
 * 
 * Usage:
 *   node tools/shaders/shader_quality_gate_v2.mjs [options]
 * 
 * Options:
 *   --dir=<path>           Directory to scan (default: frontend/)
 *   --strict               Fail on warnings
 *   --fix                  Auto-fix common issues
 *   --noSuppress           Disable warning suppression
 *   --targets=msl,hlsl,... Transpile targets (msl,hlsl,spirv)
 *   --limits=<path|latest|auto>  Device limits JSON file or alias
 *   --report=<file>        Output JSON report
 *   --junit=<file>         Output JUnit XML report
 * 
 * Limits:
 *   --limits=tools/shaders/device_limits/iphone15.json  Use explicit profile
 *   --limits=latest                                     Use latest captured device
 *   --limits=latest.ios                                 Use latest iOS device
 *   --limits=latest.android                             Use latest Android device
 *   --limits=auto                                       Auto-detect platform
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
import { exec, execSync } from 'child_process';
import { promisify } from 'util';
import crypto from 'crypto';
import { loadAndValidateLimits } from './limits_resolver_v2.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const execAsync = promisify(exec);

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
const NO_SUPPRESS = args.noSuppress === true;
const TARGETS = args.targets ? args.targets.split(',') : [];
const REPORT_FILE = args.report ? path.resolve(args.report) : null;
const JUNIT_FILE = args.junit ? path.resolve(args.junit) : null;

// Load and validate limits using shared resolver
const limitsArg = args.limits || 'latest';
const { limits: deviceLimits, path: resolvedLimitsPath, platform: detectedPlatform } = 
  loadAndValidateLimits(limitsArg, { verbose: true });

// Print tool versions
console.log('\n📦 TOOL VERSIONS:');
console.log('─'.repeat(60));
console.log(`Node: ${process.version}`);

try {
  const nagaVersion = execSync('naga-cli --version', { encoding: 'utf8' }).trim();
  console.log(`Naga: ${nagaVersion}`);
} catch {
  console.log('Naga: not found');
}

try {
  const tintVersion = execSync('tint --version', { encoding: 'utf8' }).trim();
  console.log(`Tint: ${tintVersion}`);
} catch {
  console.log('Tint: not found');
}

console.log('─'.repeat(60));

// Print resolved limits
console.log('\n📊 DEVICE LIMITS:');
console.log('─'.repeat(60));
console.log(`Requested: ${limitsArg}`);
console.log(`Resolved: ${resolvedLimitsPath}`);
console.log(`Platform: ${detectedPlatform}`);
console.log(`Values:`);
console.log(`  maxComputeInvocationsPerWorkgroup: ${deviceLimits.maxComputeInvocationsPerWorkgroup}`);
console.log(`  maxComputeWorkgroupSizeX: ${deviceLimits.maxComputeWorkgroupSizeX}`);
console.log(`  maxComputeWorkgroupSizeY: ${deviceLimits.maxComputeWorkgroupSizeY}`);
console.log(`  maxComputeWorkgroupSizeZ: ${deviceLimits.maxComputeWorkgroupSizeZ}`);
if (deviceLimits.maxComputeWorkgroupStorageSize) {
  console.log(`  maxComputeWorkgroupStorageSize: ${deviceLimits.maxComputeWorkgroupStorageSize} (${(deviceLimits.maxComputeWorkgroupStorageSize/1024).toFixed(1)} KiB)`);
}
console.log('─'.repeat(60) + '\n');

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
      if (!content.includes('@compute') || !content.includes('fn ')) {
        return [];
      }
      
      if (!content.includes('@workgroup_size')) {
        return [{
          line: 1,
          message: 'Compute shader missing @workgroup_size declaration'
        }];
      }
      return [];
    }
  },
  
  WORKGROUP_SIZE_LIMITS: {
    severity: 'error',
    check: (content) => {
      const errors = [];
      const workgroupRegex = /@workgroup_size\((\d+)(?:,\s*(\d+))?(?:,\s*(\d+))?\)/g;
      let match;
      
      while ((match = workgroupRegex.exec(content)) !== null) {
        const x = parseInt(match[1]) || 1;
        const y = parseInt(match[2]) || 1;
        const z = parseInt(match[3]) || 1;
        const line = content.substring(0, match.index).split('\n').length;
        
        if (x > deviceLimits.maxComputeWorkgroupSizeX) {
          errors.push({
            line,
            message: `Workgroup size X (${x}) exceeds limit (${deviceLimits.maxComputeWorkgroupSizeX})`
          });
        }
        
        if (y > deviceLimits.maxComputeWorkgroupSizeY) {
          errors.push({
            line,
            message: `Workgroup size Y (${y}) exceeds limit (${deviceLimits.maxComputeWorkgroupSizeY})`
          });
        }
        
        if (z > deviceLimits.maxComputeWorkgroupSizeZ) {
          errors.push({
            line,
            message: `Workgroup size Z (${z}) exceeds limit (${deviceLimits.maxComputeWorkgroupSizeZ})`
          });
        }
        
        const invocations = x * y * z;
        if (invocations > deviceLimits.maxComputeInvocationsPerWorkgroup) {
          errors.push({
            line,
            message: `Total invocations (${invocations}) exceeds limit (${deviceLimits.maxComputeInvocationsPerWorkgroup})`
          });
        }
      }
      
      return errors;
    },
    fix: (content) => {
      // Auto-fix by clamping to limits
      return content.replace(/@workgroup_size\((\d+)(?:,\s*(\d+))?(?:,\s*(\d+))?\)/g, (match, x, y, z) => {
        const newX = Math.min(parseInt(x) || 1, deviceLimits.maxComputeWorkgroupSizeX);
        const newY = Math.min(parseInt(y) || 1, deviceLimits.maxComputeWorkgroupSizeY);
        const newZ = Math.min(parseInt(z) || 1, deviceLimits.maxComputeWorkgroupSizeZ);
        
        // Ensure total invocations don't exceed limit
        let invocations = newX * newY * newZ;
        if (invocations > deviceLimits.maxComputeInvocationsPerWorkgroup) {
          const scale = Math.cbrt(deviceLimits.maxComputeInvocationsPerWorkgroup / invocations);
          const finalX = Math.max(1, Math.floor(newX * scale));
          const finalY = Math.max(1, Math.floor(newY * scale));
          const finalZ = Math.max(1, Math.floor(newZ * scale));
          
          if (finalY === 1 && finalZ === 1) {
            return `@workgroup_size(${finalX})`;
          } else if (finalZ === 1) {
            return `@workgroup_size(${finalX}, ${finalY})`;
          } else {
            return `@workgroup_size(${finalX}, ${finalY}, ${finalZ})`;
          }
        }
        
        if (newY === 1 && newZ === 1) {
          return `@workgroup_size(${newX})`;
        } else if (newZ === 1) {
          return `@workgroup_size(${newX}, ${newY})`;
        } else {
          return `@workgroup_size(${newX}, ${newY}, ${newZ})`;
        }
      });
    }
  },
  
  // Performance
  AVOID_DYNAMIC_INDEXING: {
    severity: 'warning',
    suppressible: true,
    check: (content) => {
      const warnings = [];
      const dynamicIndexRegex = /\[([^0-9\]]+)\]/g;
      let match;
      
      while ((match = dynamicIndexRegex.exec(content)) !== null) {
        const line = content.substring(0, match.index).split('\n').length;
        warnings.push({
          line,
          message: `Dynamic array indexing: [${match[1]}] - consider bounds checking`
        });
      }
      
      return warnings;
    }
  },
  
  NO_STORAGE_VEC3: {
    severity: 'warning',
    check: (content) => {
      const warnings = [];
      const vec3Regex = /struct\s+\w+\s*\{[^}]*vec3<[^>]+>[^}]*\}/g;
      
      if (vec3Regex.test(content)) {
        warnings.push({
          line: 1,
          message: 'vec3 in storage buffer may cause alignment issues'
        });
      }
      
      return warnings;
    }
  }
};

// File validation
async function validateFile(filePath) {
  const content = fs.readFileSync(filePath, 'utf8');
  const fileName = path.basename(filePath);
  const results = {
    file: fileName,
    errors: [],
    warnings: [],
    fixed: false
  };
  
  // Apply rules
  for (const [ruleName, rule of Object.entries(RULES)) {
    const issues = rule.check(content);
    
    if (issues.length > 0) {
      const category = rule.severity === 'error' ? 'errors' : 'warnings';
      
      // Apply suppression for warnings if enabled
      if (category === 'warnings' && rule.suppressible && !NO_SUPPRESS) {
        // Suppressed - don't add to results
        continue;
      }
      
      results[category].push(...issues.map(issue => ({
        rule: ruleName,
        ...issue
      })));
      
      // Auto-fix if enabled and fix available
      if (FIX_MODE && rule.fix) {
        const fixed = rule.fix(content);
        if (fixed !== content) {
          fs.writeFileSync(filePath, fixed);
          results.fixed = true;
          console.log(`  🔧 Fixed: ${ruleName}`);
        }
      }
    }
  }
  
  // Run external validators
  for (const target of TARGETS) {
    try {
      await execAsync(`${target} ${filePath}`);
      console.log(`  ✅ ${target}: Valid`);
    } catch (error) {
      results.errors.push({
        rule: `${target.toUpperCase()}_VALIDATION`,
        message: error.message
      });
    }
  }
  
  return results;
}

// Main validation
async function main() {
  console.log(`📂 Scanning directory: ${SHADER_DIR}`);
  
  const files = fs.readdirSync(SHADER_DIR, { recursive: true })
    .filter(f => f.endsWith('.wgsl'))
    .map(f => path.join(SHADER_DIR, f));
  
  console.log(`📊 Found ${files.length} WGSL files\n`);
  
  const allResults = [];
  let totalErrors = 0;
  let totalWarnings = 0;
  
  for (const file of files) {
    console.log(`Validating: ${path.relative(SHADER_DIR, file)}`);
    const results = await validateFile(file);
    allResults.push(results);
    
    totalErrors += results.errors.length;
    totalWarnings += results.warnings.length;
    
    if (results.errors.length > 0) {
      console.log(`  ❌ ${results.errors.length} errors`);
      if (STRICT_MODE) {
        results.errors.forEach(e => console.log(`     ${e.rule}: ${e.message}`));
      }
    }
    
    if (results.warnings.length > 0) {
      console.log(`  ⚠️  ${results.warnings.length} warnings`);
      if (STRICT_MODE && !NO_SUPPRESS) {
        console.log(`     (suppressed - use --noSuppress to see details)`);
      } else if (NO_SUPPRESS) {
        results.warnings.forEach(w => console.log(`     ${w.rule}: ${w.message}`));
      }
    }
    
    if (results.errors.length === 0 && results.warnings.length === 0) {
      console.log(`  ✅ Clean`);
    }
  }
  
  // Generate reports
  if (REPORT_FILE) {
    const report = {
      timestamp: new Date().toISOString(),
      meta: {
        limits: {
          requested: limitsArg,
          resolved: resolvedLimitsPath,
          platform: detectedPlatform,
          values: deviceLimits
        },
        tooling: {
          node: process.version,
          targets: TARGETS
        },
        options: {
          strict: STRICT_MODE,
          fix: FIX_MODE,
          noSuppress: NO_SUPPRESS
        }
      },
      results: allResults,
      summary: {
        files: allResults.length,
        errors: totalErrors,
        warnings: totalWarnings
      }
    };
    
    fs.mkdirSync(path.dirname(REPORT_FILE), { recursive: true });
    fs.writeFileSync(REPORT_FILE, JSON.stringify(report, null, 2));
    console.log(`\n📄 Report saved to: ${REPORT_FILE}`);
  }
  
  // Summary
  console.log('\n' + '='.repeat(60));
  console.log('VALIDATION SUMMARY');
  console.log('='.repeat(60));
  console.log(`Files:    ${allResults.length}`);
  console.log(`Errors:   ${totalErrors}`);
  console.log(`Warnings: ${totalWarnings} ${NO_SUPPRESS ? '' : '(suppressed)'}`);
  console.log(`Limits:   ${resolvedLimitsPath}`);
  console.log('='.repeat(60));
  
  // Exit codes
  if (totalErrors > 0) {
    process.exit(1);
  } else if (STRICT_MODE && totalWarnings > 0) {
    process.exit(2);
  } else {
    process.exit(0);
  }
}

// Run
main().catch(error => {
  console.error('Fatal error:', error);
  process.exit(3);
});
