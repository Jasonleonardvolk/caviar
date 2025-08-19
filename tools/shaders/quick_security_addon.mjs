import fs from 'fs/promises';
import path from 'path';

const CRITICAL_PATTERNS = {
  arraySizeMismatch: {
    pattern: /shared_\w+\[clamp_index_dyn\(i,\s*(\d+)u?\)/g,
    check: (match) => match[1] === '256',
    fix: 'Use MAX_OSCILLATORS instead of 256'
  },
  selfAssignment: {
    pattern: /(\w+)\s*=\s*\1\s*;/g,
    check: () => true,
    fix: 'Remove self-assignment'
  }
};

async function scanShader(filePath) {
  const code = await fs.readFile(filePath, 'utf8');
  const issues = [];
  
  for (const [name, rule] of Object.entries(CRITICAL_PATTERNS)) {
    const matches = [...code.matchAll(rule.pattern)];
    for (const match of matches) {
      if (rule.check(match)) {
        const line = code.substring(0, match.index).split('\n').length;
        issues.push({
          file: path.basename(filePath),
          line,
          type: name,
          fix: rule.fix
        });
      }
    }
  }
  return issues;
}

async function main() {
  const dir = process.argv[2] || 'frontend/lib/webgpu/shaders';
  console.log('Scanning shaders in:', dir);
  
  const files = await fs.readdir(dir);
  let found = false;
  
  for (const file of files.filter(f => f.endsWith('.wgsl'))) {
    const issues = await scanShader(path.join(dir, file));
    if (issues.length > 0) {
      found = true;
      console.log(`\n${file}:`);
      issues.forEach(i => console.log(`  Line ${i.line}: ${i.fix}`));
    }
  }
  
  if (!found) console.log('\nNo issues found!');
  process.exit(found ? 1 : 0);
}

main().catch(console.error);
