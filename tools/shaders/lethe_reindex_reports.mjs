// tools/shaders/lethe_reindex_reports.mjs
// Cleans and sets shader_validation_latest.json to point at most recent JSON report.
import fs from 'node:fs';
import path from 'node:path';

const repoRoot = process.cwd();
const reportsDir = path.join(repoRoot, 'tools', 'shaders', 'reports');
fs.mkdirSync(reportsDir, { recursive: true });
const files = fs.readdirSync(reportsDir).filter(f => f.startsWith('shader_validation_') && f.endsWith('.json'));
if (!files.length) {
  console.error('[lethe] No reports found.');
  process.exit(1);
}
files.sort(); // timestamped names sort lexicographically
const latest = files[files.length - 1];
const latestPath = path.join(reportsDir, latest);
const content = fs.readFileSync(latestPath, 'utf8');
fs.writeFileSync(path.join(reportsDir, 'shader_validation_latest.json'), content);
console.log(`[lethe] Set latest -> ${latest}`);
