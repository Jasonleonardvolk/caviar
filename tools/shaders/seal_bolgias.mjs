// tools/shaders/seal_bolgias.mjs
// Removes tracked public shader copies and adds ignore rule.
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
const repoRoot = process.cwd();
const ignoreLine = '/frontend/public/hybrid/wgsl/';
const gitignore = path.join(repoRoot, '.gitignore');

function run(cmd, args) {
  return spawnSync(cmd, args, { stdio: 'inherit', shell: process.platform === 'win32' }).status ?? 1;
}

// Remove tracked files if present
run('git', ['rm', '-r', '--cached', 'frontend/public/hybrid/wgsl']);
// Ensure ignore rule
let contents = '';
if (fs.existsSync(gitignore)) contents = fs.readFileSync(gitignore, 'utf8');
if (!contents.includes(ignoreLine)) {
  fs.writeFileSync(gitignore, (contents ? contents.trimEnd() + '\n' : '') + ignoreLine + '\n');
  console.log('[seal_bolgias] Added ignore rule to .gitignore');
} else {
  console.log('[seal_bolgias] .gitignore already contains ignore rule');
}
console.log('[seal_bolgias] Done.');
