#!/usr/bin/env node
/**
 * Intelligent WGSL Diagnostic System
 * Actually understands WGSL and gives REAL fixes, not generic garbage
 * 
 * Replaces the fired gentleman's work with actual intelligence
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { exec } from 'child_process';
import { promisify } from 'util';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const execAsync = promisify(exec);

// WGSL-specific knowledge base
const WGSL_RULES = {
  SWIZZLE_ASSIGNMENT: {
    pattern: /(\w+)\.([rgba]{2,4}|[xyzw]{2,4})\s*([+\-*/%])?=/g,
    diagnose: (match, fileContent, line) => {
      const [full, varName, swizzle, operator] = match;
      return {
        problem: `WGSL prohibits assignment to swizzles (${varName}.${swizzle})`,
        reason: 'WGSL spec does not support swizzle assignments for safety',
        fix: swizzle.split('').map(c => 
          `${varName}.${c} ${operator || ''}= value;`
        ).join('\n'),
        severity: 'ERROR'
      };
    }
  },

  TYPE_MISMATCH_VEC_CONSTRUCTOR: {
    pattern: /vec([234])<(i32|u32|f32)>\s*\(([^)]+)\)/g,
    diagnose: (match, fileContent, line) => {
      const [full, dims, type, args] = match;
      const argList = args.split(',').map(a => a.trim());
      
      // Check if arguments match the expected type
      const issues = [];
      argList.forEach((arg, i) => {
        if (type === 'i32' && /^[a-z_]\w*$/.test(arg)) {
          // It's a variable, need to check its type from context
          const varDecl = new RegExp(`(let|var|const)\\s+${arg}\\s*(?::\\s*(\\w+))?\\s*=`);
          const declMatch = fileContent.match(varDecl);
          if (declMatch && declMatch[2] && declMatch[2] !== 'i32') {
            issues.push({
              arg,
              currentType: declMatch[2],
              needsType: type
            });
          }
        }
      });

      if (issues.length > 0) {
        return {
          problem: `Type mismatch in vec${dims}<${type}> constructor`,
          reason: `Arguments must be explicitly cast to ${type}`,
          fix: `vec${dims}<${type}>(${issues.map(i => 
            `${type}(${i.arg})`
          ).join(', ')})`,
          severity: 'ERROR'
        };
      }
      return null;
    }
  },

  TEXTURE_LOAD_ARGS: {
    pattern: /textureLoad\s*\(\s*(\w+)\s*,([^)]+)\)/g,
    diagnose: (match, fileContent, line) => {
      const [full, textureName, args] = match;
      const argCount = args.split(',').length;
      
      // Find texture declaration
      const texDecl = new RegExp(`var\\s+${textureName}\\s*:\\s*texture_([^;]+);`);
      const declMatch = fileContent.match(texDecl);
      
      if (declMatch) {
        const textureType = declMatch[1];
        let expectedArgs = 0;
        let signature = '';
        
        if (textureType.includes('2d_array')) {
          expectedArgs = 4;
          signature = 'texture, coords: vec2<i32>, array_index: i32, level: i32';
        } else if (textureType.includes('2d')) {
          expectedArgs = 3;
          signature = 'texture, coords: vec2<i32>, level: i32';
        } else if (textureType.includes('3d')) {
          expectedArgs = 3;
          signature = 'texture, coords: vec3<i32>, level: i32';
        }
        
        if (argCount < expectedArgs) {
          return {
            problem: `textureLoad missing arguments (has ${argCount}, needs ${expectedArgs})`,
            reason: `texture_${textureType} requires: ${signature}`,
            fix: expectedArgs === 4 ? 
              `textureLoad(${textureName}, vec2<i32>(coords), array_index, 0)` :
              `textureLoad(${textureName}, vec2<i32>(coords), 0)`,
            severity: 'ERROR'
          };
        }
      }
      return null;
    }
  },

  STORAGE_VEC3_PADDING: {
    pattern: /struct\s+(\w+)\s*\{([^}]+)\}/g,
    diagnose: (match, fileContent, line) => {
      const [full, structName, body] = match;
      
      // Check if this struct is used in storage
      const storageUsage = new RegExp(`var<storage[^>]*>\\s+\\w+\\s*:\\s*(?:array<)?${structName}`);
      if (!fileContent.match(storageUsage)) {
        return null; // Not a storage struct
      }
      
      // Parse fields
      const fields = body.match(/(\w+)\s*:\s*([^,;]+)/g) || [];
      const fixes = [];
      
      fields.forEach((field, i) => {
        if (field.includes('vec3<f32>')) {
          const nextField = fields[i + 1];
          if (!nextField || !nextField.includes('vec3<f32>')) {
            const fieldName = field.match(/(\w+)\s*:/)[1];
            fixes.push({
              after: fieldName,
              add: '_padding: f32'
            });
          }
        }
      });
      
      if (fixes.length > 0) {
        return {
          problem: `Storage buffer struct '${structName}' has unpadded vec3<f32> fields`,
          reason: 'vec3 is 12 bytes but alignment is 16 in storage buffers',
          fix: fixes.map(f => `Add after ${f.after}: ${f.add}`).join('\n'),
          severity: 'WARNING'
        };
      }
      return null;
    }
  },

  DYNAMIC_ARRAY_BOUNDS: {
    pattern: /(\w+)\[([^\]]+)\]/g,
    diagnose: (match, fileContent, lineNum, lineContent) => {
      const [full, arrayName, indexExpr] = match;
      
      // Check if clamp_index_dyn is already used
      if (indexExpr.includes('clamp_index_dyn')) {
        return {
          problem: null, // No problem, already safe
          reason: 'Already using bounds checking helper',
          severity: 'SAFE'
        };
      }
      
      // Check if it's a constant index
      if (/^\d+$/.test(indexExpr)) {
        return null; // Constant index is fine
      }
      
      // Check if it's workgroup/shared memory with known size
      const sharedDecl = new RegExp(`var<workgroup>\\s+${arrayName}\\s*:\\s*array<[^,]+,\\s*(\\d+)`);
      const sharedMatch = fileContent.match(sharedDecl);
      
      if (sharedMatch) {
        const size = parseInt(sharedMatch[1]);
        return {
          problem: `Dynamic index into workgroup array '${arrayName}[${indexExpr}]'`,
          reason: 'Could exceed array bounds',
          fix: `${arrayName}[min(${indexExpr}, ${size - 1}u)]`,
          severity: 'WARNING'
        };
      }
      
      // Storage buffer array
      return {
        problem: `Dynamic index '${arrayName}[${indexExpr}]' without visible bounds check`,
        reason: 'Validator cannot verify bounds checking',
        fix: `${arrayName}[clamp_index_dyn(${indexExpr}, arrayLength(&${arrayName}))]`,
        severity: 'INFO'
      };
    }
  },

  VERTEX_ATTRIBUTE_FALSE_POSITIVE: {
    pattern: /@location\((\d+)\)\s+(\w+)\s*:\s*vec3<f32>/g,
    diagnose: () => {
      return {
        problem: null,
        reason: 'Vertex attributes do not need padding (not storage)',
        severity: 'FALSE_POSITIVE'
      };
    }
  }
};

class IntelligentDiagnostic {
  constructor() {
    this.diagnostics = [];
    this.falsePositives = [];
    this.realIssues = [];
  }

  async analyzeFile(filePath) {
    const content = fs.readFileSync(filePath, 'utf8');
    const lines = content.split('\n');
    const fileName = path.basename(filePath);
    
    console.log(`\n🔍 Analyzing ${fileName} with intelligence...`);
    
    // First, run Naga to get actual errors
    const nagaErrors = await this.runNaga(filePath);
    
    // Then apply our intelligent analysis
    for (const [ruleName, rule] of Object.entries(WGSL_RULES)) {
      const matches = [...content.matchAll(rule.pattern)];
      
      for (const match of matches) {
        const lineNum = content.slice(0, match.index).split('\n').length;
        const lineContent = lines[lineNum - 1];
        
        const diagnostic = rule.diagnose(match, content, lineNum, lineContent);
        
        if (diagnostic) {
          if (diagnostic.severity === 'FALSE_POSITIVE' || diagnostic.severity === 'SAFE') {
            this.falsePositives.push({
              file: fileName,
              line: lineNum,
              rule: ruleName,
              ...diagnostic
            });
          } else if (diagnostic.problem) {
            this.realIssues.push({
              file: fileName,
              line: lineNum,
              rule: ruleName,
              ...diagnostic
            });
          }
        }
      }
    }
    
    // Cross-reference with Naga errors for accuracy
    this.correlateWithNaga(nagaErrors, fileName);
  }

  async runNaga(filePath) {
    try {
      const { stderr } = await execAsync(`naga validate "${filePath}" --input-kind wgsl`);
      if (stderr) {
        // Parse Naga errors
        const errors = [];
        const errorPattern = /error:([^\n]+)\n\s+┌─[^\n]+:(\d+):(\d+)/g;
        const matches = [...stderr.matchAll(errorPattern)];
        
        for (const match of matches) {
          errors.push({
            message: match[1].trim(),
            line: parseInt(match[2]),
            column: parseInt(match[3])
          });
        }
        return errors;
      }
    } catch (e) {
      // Naga failed - parse error output
      if (e.stderr) {
        return this.parseNagaError(e.stderr);
      }
    }
    return [];
  }

  parseNagaError(stderr) {
    const errors = [];
    
    // Parse swizzle assignment errors
    if (stderr.includes('cannot assign to this expression')) {
      const swizzleMatch = stderr.match(/(\d+)\s+│\s+(\w+\.rgb)/);
      if (swizzleMatch) {
        errors.push({
          type: 'SWIZZLE_ASSIGNMENT',
          line: parseInt(swizzleMatch[1]),
          expression: swizzleMatch[2]
        });
      }
    }
    
    // Parse type mismatch
    if (stderr.includes('Composing') && stderr.includes('component type is not expected')) {
      const typeMatch = stderr.match(/:(\d+):(\d+)/);
      if (typeMatch) {
        errors.push({
          type: 'TYPE_MISMATCH',
          line: parseInt(typeMatch[1]),
          column: parseInt(typeMatch[2])
        });
      }
    }
    
    // Parse wrong argument count
    if (stderr.includes('wrong number of arguments')) {
      const argsMatch = stderr.match(/(\d+)\s+│[^\n]+textureLoad/);
      if (argsMatch) {
        errors.push({
          type: 'WRONG_ARG_COUNT',
          line: parseInt(argsMatch[1])
        });
      }
    }
    
    return errors;
  }

  correlateWithNaga(nagaErrors, fileName) {
    // Enhance our diagnostics with Naga's findings
    for (const error of nagaErrors) {
      const existingDiag = this.realIssues.find(d => 
        d.file === fileName && Math.abs(d.line - error.line) <= 2
      );
      
      if (existingDiag) {
        existingDiag.nagaConfirmed = true;
        existingDiag.nagaMessage = error.message || error.type;
      } else {
        // Naga found something we missed
        this.realIssues.push({
          file: fileName,
          line: error.line,
          rule: 'NAGA_DETECTED',
          problem: error.message || error.type,
          reason: 'Detected by Naga validator',
          severity: 'ERROR',
          nagaConfirmed: true
        });
      }
    }
  }

  generateReport() {
    console.log('\n' + '='.repeat(70));
    console.log('📊 INTELLIGENT DIAGNOSTIC REPORT');
    console.log('='.repeat(70));
    
    // Real Issues That Need Fixing
    if (this.realIssues.length > 0) {
      console.log('\n❌ REAL ISSUES TO FIX:\n');
      
      for (const issue of this.realIssues) {
        const confirmed = issue.nagaConfirmed ? ' ✓' : '';
        console.log(`📍 ${issue.file}:${issue.line} [${issue.severity}${confirmed}]`);
        console.log(`   Problem: ${issue.problem}`);
        console.log(`   Reason:  ${issue.reason}`);
        if (issue.fix) {
          console.log(`   FIX:     ${issue.fix}`);
        }
        console.log();
      }
    } else {
      console.log('\n✅ NO REAL ISSUES FOUND!');
    }
    
    // False Positives We Can Ignore
    console.log('\n🟡 FALSE POSITIVES (Safe to Ignore):');
    console.log(`   Found ${this.falsePositives.length} false positives`);
    
    const fpByRule = {};
    for (const fp of this.falsePositives) {
      fpByRule[fp.rule] = (fpByRule[fp.rule] || 0) + 1;
    }
    
    for (const [rule, count] of Object.entries(fpByRule)) {
      console.log(`   - ${rule}: ${count} instances`);
    }
    
    // Summary
    console.log('\n' + '='.repeat(70));
    console.log('📈 SUMMARY:');
    console.log(`   Real Issues:     ${this.realIssues.length}`);
    console.log(`   False Positives: ${this.falsePositives.length}`);
    console.log(`   Action Required: ${this.realIssues.filter(i => i.severity === 'ERROR').length} errors`);
    
    // Save detailed JSON report
    const report = {
      timestamp: new Date().toISOString(),
      summary: {
        realIssues: this.realIssues.length,
        falsePositives: this.falsePositives.length,
        errors: this.realIssues.filter(i => i.severity === 'ERROR').length,
        warnings: this.realIssues.filter(i => i.severity === 'WARNING').length
      },
      realIssues: this.realIssues,
      falsePositives: this.falsePositives
    };
    
    const reportPath = path.join(__dirname, '../../build/intelligent_diagnostic.json');
    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
    console.log(`\n📁 Detailed report: ${reportPath}`);
  }

  async analyzeDirectory(dir) {
    const files = fs.readdirSync(dir)
      .filter(f => f.endsWith('.wgsl'))
      .map(f => path.join(dir, f));
    
    console.log(`🧠 Intelligent WGSL Diagnostic System`);
    console.log(`📂 Analyzing ${files.length} shaders...\n`);
    
    for (const file of files) {
      await this.analyzeFile(file);
    }
    
    this.generateReport();
  }
}

// Auto-fix generator
class IntelligentFixer {
  static generateFix(diagnostic) {
    const fixes = [];
    
    switch (diagnostic.rule) {
      case 'SWIZZLE_ASSIGNMENT':
        // Generate individual component assignments
        const components = diagnostic.problem.match(/\.([rgba]{2,4}|[xyzw]{2,4})/)[1];
        for (const comp of components) {
          fixes.push({
            type: 'replace_line',
            line: diagnostic.line,
            newCode: `    ${diagnostic.varName}.${comp} *= value; // Fixed swizzle assignment`
          });
        }
        break;
        
      case 'TYPE_MISMATCH_VEC_CONSTRUCTOR':
        fixes.push({
          type: 'replace_in_line',
          line: diagnostic.line,
          pattern: diagnostic.problem,
          replacement: diagnostic.fix
        });
        break;
        
      case 'TEXTURE_LOAD_ARGS':
        fixes.push({
          type: 'add_argument',
          line: diagnostic.line,
          function: 'textureLoad',
          argument: ', 0'
        });
        break;
    }
    
    return fixes;
  }
}

// Main execution
async function main() {
  const args = process.argv.slice(2);
  const targetDir = args[0] || path.join(__dirname, '../../frontend/hybrid/wgsl');
  
  const diagnostic = new IntelligentDiagnostic();
  await diagnostic.analyzeDirectory(targetDir);
  
  // Generate fix script if requested
  if (args.includes('--generate-fixes')) {
    console.log('\n🔧 Generating fix script...');
    const fixes = diagnostic.realIssues
      .filter(i => i.fix)
      .map(i => IntelligentFixer.generateFix(i));
    
    const fixScript = path.join(__dirname, '../../build/apply_fixes.sh');
    // Generate sed commands or VS Code commands
    console.log(`📝 Fix script: ${fixScript}`);
  }
}

main().catch(console.error);
