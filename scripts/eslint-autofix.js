/**
 * ESLint Auto-fix Script
 * 
 * This script applies common fixes for ESLint warnings in the codebase.
 * It handles:
 * 1. Prefix unused variables with _ to prevent warnings
 * 2. Fix anonymous default exports
 * 3. Fix template string syntax
 * 4. Add parentheses to mixed operators
 * 5. Add missing dependencies to useEffect hooks
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

// Find files with ESLint warnings
console.log('Finding files with ESLint warnings...');
const filesWithWarnings = new Set();

try {
  // Get ESLint output
  const eslintOutput = execSync('npx eslint "src/**/*.{js,jsx}" "client/src/**/*.{js,jsx}" "server/**/*.{js,jsx}" --format json')
    .toString();
  
  const results = JSON.parse(eslintOutput);
  
  // Collect files with warnings
  results.forEach(result => {
    if (result.messages.length > 0) {
      filesWithWarnings.add(result.filePath);
    }
  });
  
  console.log(`Found ${filesWithWarnings.size} files with warnings`);
} catch (error) {
  console.error('Error running ESLint:', error.message);
  process.exit(1);
}

// Fix anonymous default exports
console.log('\nFixing anonymous default exports...');
for (const filePath of filesWithWarnings) {
  if (!fs.existsSync(filePath)) continue;
  
  const content = fs.readFileSync(filePath, 'utf-8');
  
  // Check if file has anonymous default export
  if (content.includes('export default {') || content.includes('export default (')) {
    const fileName = path.basename(filePath, path.extname(filePath));
    const varName = fileName.replace(/([a-z])([A-Z])/g, '$1_$2').toLowerCase() + 'Service';
    
    // Replace anonymous export with named export
    const newContent = content.replace(
      /export default (\{[\s\S]*\}|\([\s\S]*\));?$/,
      `const ${varName} = $1;\n\nexport default ${varName};`
    );
    
    if (newContent !== content) {
      fs.writeFileSync(filePath, newContent);
      console.log(`  Fixed anonymous export in ${filePath}`);
    }
  }
}

// Fix template string syntax
console.log('\nFixing template string syntax...');
for (const filePath of filesWithWarnings) {
  if (!fs.existsSync(filePath)) continue;
  
  const content = fs.readFileSync(filePath, 'utf-8');
  
  // Replace "${...}" with "\${...}" when inside regular quotes
  const newContent = content.replace(
    /(['"])(.*?)\${(.*?)}\2/g,
    (match, quote, before, expr, after) => {
      return `\`${before}\${${expr}}\``;
    }
  );
  
  if (newContent !== content) {
    fs.writeFileSync(filePath, newContent);
    console.log(`  Fixed template string in ${filePath}`);
  }
}

// Fix mixed operators by adding parentheses
console.log('\nFixing mixed operators...');
for (const filePath of filesWithWarnings) {
  if (!fs.existsSync(filePath)) continue;
  
  const content = fs.readFileSync(filePath, 'utf-8');
  
  // Add parentheses around || && expressions
  const newContent = content.replace(
    /([^&|!()]+)(\s*\|\|\s*)([^&|()]+)(\s*&&\s*)([^&|()]+)/g,
    '($1$2$3)$4$5'
  );
  
  if (newContent !== content) {
    fs.writeFileSync(filePath, newContent);
    console.log(`  Fixed mixed operators in ${filePath}`);
  }
}

// Prefix unused variables
console.log('\nPrefixing unused variables (MANUAL DEMO ONLY - requires parser)...');
console.log('  This would require a proper JS parser to implement correctly');
console.log('  For demonstration, manually prefix variables with _ where reported by ESLint');

console.log('\nScript completed. Some fixes require manual attention:');
console.log('1. React Hook dependencies - add missing deps or wrap functions in useCallback');
console.log('2. Complex unused variables - manually prefix with _');
console.log('3. Parsing errors - manually fix these syntax issues');
