// tools/shaders/import_device_limits.mjs
// Usage:
//   node tools/shaders/import_device_limits.mjs --in=path/to/limits.json --name=iphone15
// Writes to tools/shaders/device_limits/<name>.json with normalized keys.

import fs from 'node:fs';
import path from 'node:path';

const args = Object.fromEntries(process.argv.slice(2).map(a => {
  const [k, v] = a.split('=');
  return [k.replace(/^--/, ''), v];
}));

if (!args.in || !args.name) {
  console.error('Usage: node tools/shaders/import_device_limits.mjs --in=path/to/limits.json --name=iphone15');
  process.exit(2);
}

const inputPath = path.resolve(process.cwd(), args.in);
if (!fs.existsSync(inputPath)) {
  console.error('Input file not found:', inputPath);
  process.exit(2);
}

let raw;
try {
  raw = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
} catch (e) {
  console.error('Failed to parse JSON:', e.message);
  process.exit(2);
}

// Accept either a plain limits dump or the normalized format from our probe
const limits = raw.__raw || raw;

function pick(k, fallbackKeys = []) {
  for (const key of [k, ...fallbackKeys]) {
    if (limits[key] != null) return limits[key];
  }
  return null;
}

// Normalize most important constraints we enforce
const normalized = {
  maxComputeInvocationsPerWorkgroup: pick('maxComputeInvocationsPerWorkgroup'),
  maxComputeWorkgroupSizeX: pick('maxComputeWorkgroupSizeX'),
  maxComputeWorkgroupSizeY: pick('maxComputeWorkgroupSizeY'),
  maxComputeWorkgroupSizeZ: pick('maxComputeWorkgroupSizeZ'),
  maxComputeWorkgroupStorageSize: pick('maxComputeWorkgroupStorageSize', ['maxWorkgroupStorageSize']),
  maxSampledTexturesPerShaderStage: pick('maxSampledTexturesPerShaderStage'),
  maxSamplersPerShaderStage: pick('maxSamplersPerShaderStage'),
  label: args.name
};

const outDir = path.join(process.cwd(), 'tools', 'shaders', 'device_limits');
fs.mkdirSync(outDir, { recursive: true });
const outPath = path.join(outDir, `${args.name}.json`);
fs.writeFileSync(outPath, JSON.stringify(normalized, null, 2));
console.log('Wrote', outPath);