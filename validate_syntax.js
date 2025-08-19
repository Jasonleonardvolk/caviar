#!/usr/bin/env node
// Simple syntax validator for common TypeScript/JavaScript errors
// Run with: node validate_syntax.js

const fs = require('fs');
const path = require('path');

const COLORS = {
  RED: '\x1b[31m',
  GREEN: '\x1b[32m',
  YELLOW: '\x1b[33m',
  RESET: '\x1b[0m'
};

// Patterns to check
const PATTERNS = [
  {
    name: 'Missing closing parenthesis after .slice()',
    regex: /\.slice\(\)[^)]*;/g,
    message: 'Found .slice() with missing closing parenthesis for outer function call'
  },
  {
    name: 'Static keyword in function',
    regex: /function[^{]*{[^}]*\bstatic\s+\w+\s*=/g,
    message: 'Found "static" keyword inside a regular function (should be in class)'
  },
  {
    name: 'Unclosed writeBuffer calls',
    regex: /writeBuffer\([^)]*\.slice\(\);/g,
    message: 'Found writeBuffer with unclosed parenthesis after .slice()'
  }
];

function checkFile(filePath) {
  const content = fs.readFileSync(filePath, 'utf8');
  const issues = [];
  
  PATTERNS.forEach(pattern => {
    const matches = content.match(pattern.regex);
    if (matches) {
      matches.forEach(match => {
        // Find line number
        const lines = content.substring(0, content.indexOf(match)).split('\n');
        const lineNum = lines.length;
        issues.push({
          pattern: pattern.name,
          message: pattern.message,
          line: lineNum,
          match: match.trim()
        });
      });
    }
  });
  
  return issues;
}

function scanDirectory(dir, extensions = ['.ts', '.tsx', '.js', '.jsx']) {
  let totalIssues = 0;
  const results = [];
  
  function scan(currentPath) {
    const items = fs.readdirSync(currentPath);
    
    items.forEach(item => {
      const fullPath = path.join(currentPath, item);
      const stat = fs.statSync(fullPath);
      
      // Skip node_modules and .git directories
      if (stat.isDirectory() && !item.startsWith('.') && item !== 'node_modules') {
        scan(fullPath);
      } else if (stat.isFile() && extensions.includes(path.extname(item))) {
        const issues = checkFile(fullPath);
        if (issues.length > 0) {
          results.push({
            file: path.relative(dir, fullPath),
            issues
          });
          totalIssues += issues.length;
        }
      }
    });
  }
  
  scan(dir);
  
  return { results, totalIssues };
}

// Main execution
const targetDir = process.argv[2] || '.';

console.log(`${COLORS.YELLOW}Scanning for common syntax issues in: ${targetDir}${COLORS.RESET}\n`);

const { results, totalIssues } = scanDirectory(targetDir);

if (totalIssues === 0) {
  console.log(`${COLORS.GREEN}✓ No syntax issues found!${COLORS.RESET}`);
} else {
  console.log(`${COLORS.RED}Found ${totalIssues} potential issues:${COLORS.RESET}\n`);
  
  results.forEach(({ file, issues }) => {
    console.log(`${COLORS.YELLOW}File: ${file}${COLORS.RESET}`);
    issues.forEach(issue => {
      console.log(`  Line ${issue.line}: ${issue.message}`);
      console.log(`    ${COLORS.RED}${issue.match}${COLORS.RESET}`);
    });
    console.log();
  });
  
  console.log(`\n${COLORS.YELLOW}Run 'npx tsc --noEmit' for full TypeScript validation${COLORS.RESET}`);
}

// Usage instructions
if (process.argv.length < 3) {
  console.log('\nUsage: node validate_syntax.js [directory]');
  console.log('Example: node validate_syntax.js frontend/');
}
