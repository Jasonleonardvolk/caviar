// suppression_patch.cjs
// CommonJS version of suppression system for legacy tooling compatibility

const fs = require('fs');
const path = require('path');

class SuppressionSystemCJS {
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
      console.log(`📋 Loaded ${data.rules?.length || 0} suppression rules`);
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
      const ruleName = rule.name || rule.reason;
      this.stats.byRule.set(ruleName, (this.stats.byRule.get(ruleName) || 0) + 1);
      
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

  // Static method for quick integration
  static filterWarnings(warnings, filePath, rulesPath) {
    const suppressor = new SuppressionSystemCJS(rulesPath);
    const filtered = [];
    
    for (const warning of warnings) {
      const message = typeof warning === 'string' ? warning : warning.message;
      if (!suppressor.shouldSuppress(filePath, message)) {
        filtered.push(warning);
      }
    }
    
    return {
      filtered,
      suppressedCount: warnings.length - filtered.length,
      suppressor
    };
  }
}

module.exports = SuppressionSystemCJS;

// CLI usage if run directly
if (require.main === module) {
  const args = process.argv.slice(2);
  
  if (args.length < 1) {
    console.log('Usage: node suppression_patch.cjs <shader-file> [rules-file]');
    process.exit(1);
  }
  
  const shaderFile = args[0];
  const rulesFile = args[1];
  
  // Example usage
  const suppressor = new SuppressionSystemCJS(rulesFile);
  
  // Simulate some warnings
  const testWarnings = [
    "Dynamic array access 'array[idx]' without apparent bounds checking",
    "vec3 in storage buffer should be followed by padding"
  ];
  
  console.log(`\nTesting suppression for: ${shaderFile}`);
  for (const warning of testWarnings) {
    const suppressed = suppressor.shouldSuppress(shaderFile, warning, true);
    console.log(`  ${suppressed ? '🔇' : '⚠️'} ${warning}`);
  }
  
  suppressor.printStats();
}
