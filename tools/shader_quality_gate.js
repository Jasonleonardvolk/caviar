# Create an upgraded shader quality gate script with multi-backend checks and JSON/JUnit reporting
from pathlib import Path

code = r"""#!/usr/bin/env node
/**
 * shader_quality_gate_v2.mjs
 * Production shader validator for WebGPU (WGSL) with cross-backend checks.
 *
 * Features
 * - Validates WGSL with Naga (syntax/semantic)
 * - Optionally translates via Tint to HLSL/MSL/SPIR-V for backend sanity
 * - Enforces device limits via a --limits JSON (workgroup size, workgroup memory, bindings)
 * - Heuristic perf metrics & lint rules (regex-based; safe & fast)
 * - Emits JSON (--report) and/or JUnit XML (--junit) for CI
 *
 * Usage examples:
 *   node tools/shaders/shader_quality_gate_v2.mjs --dir frontend/shaders --strict --targets=msl,hlsl --limits=tools/shaders/device_limits.iphone15.json --report=build/shader_report.json
 *   node tools/shaders/shader_quality_gate_v2.mjs --dir frontend/shaders --fix
 *
 * Exit codes: 0 OK, 1 errors, 2 warnings (in --strict)
 */
import fs from 'fs';
import path from 'path';
import {fileURLToPath} from 'url';
import {exec} from 'child_process';
import {promisify} from 'util';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const run = promisify(exec);

const args = new Map(process.argv.slice(2).map(a => {
  const [k, v] = a.includes('=') ? a.split('=') : [a, true];
  return [k, v];
}));

const ROOT = process.cwd();
const TARGET_DIR = path.resolve(ROOT, (args.get('--dir') || 'frontend/public/hybrid/wgsl'));
const STRICT = !!args.get('--strict');
const FIX = !!args.get('--fix');
const REPORT_PATH = args.get('--report') ? path.resolve(String(args.get('--report'))) : null;
const JUNIT_PATH  = args.get('--junit')  ? path.resolve(String(args.get('--junit')))  : null;
const TARGETS = String(args.get('--targets') || '').split(',').filter(Boolean); // 'msl','hlsl','spirv'
const LIMITS = args.get('--limits') ? JSON.parse(fs.readFileSync(path.resolve(String(args.get('--limits'))), 'utf-8')) : null;

const LIMIT_DEFAULTS = {
  maxComputeInvocationsPerWorkgroup: 256,
  maxComputeWorkgroupSizeX: 256,
  maxComputeWorkgroupSizeY: 256,
  maxComputeWorkgroupSizeZ: 64,
  maxComputeWorkgroupsPerDimension: 65535,
  maxStorageBuffersPerShaderStage: 8,
  maxUniformBuffersPerShaderStage: 12,
  maxSampledTexturesPerShaderStage: 16,
  maxSamplersPerShaderStage: 16,
  maxWorkgroupStorageSize: 32768, // 32KB
};

const LIMITS_MERGED = Object.assign({}, LIMIT_DEFAULTS, LIMITS || {});

// ---------- Utility
const globWGSL = (dir) => fs.readdirSync(dir)
  .filter(f => f.endsWith('.wgsl'))
  .map(f => path.join(dir, f));

const safeRun = async (cmd) => {
  try { return await run(cmd); } catch (e) { return {stdout: e.stdout||'', stderr: e.stderr||e.message||String(e)}; }
};

const hasBin = async (name, versionFlag='--version') => {
  const {stdout, stderr} = await safeRun(`${name} ${versionFlag}`);
  return !(stderr && !stdout) && (stdout || '').length > 0;
};

// ---------- Rules (regex)
const RULES = [
  {
    id: 'REQUIRE_WORKGROUP_SIZE_FOR_COMPUTE',
    level: 'error',
    test: (code) => (code.includes('@compute') && !/@workgroup_size\s*\(/.test(code)) ? 'Compute shader missing @workgroup_size' : null,
  },
  {
    id: 'NO_DUP_BINDINGS_WITHIN_GROUP',
    level: 'error',
    test: (code) => {
      const re = /@group\((\d+)\)\s*@binding\((\d+)\)/g;
      const seen = new Set();
      let m; 
      while ((m = re.exec(code))) {
        const key = `${m[1]}:${m[2]}`;
        if (seen.has(key)) return `Duplicate @group(${m[1]})/@binding(${m[2]})`;
        seen.add(key);
      }
      return null;
    }
  },
  {
    id: 'AVOID_DYNAMIC_INDEXING',
    level: 'warning',
    test: (code) => /\[[^\]\d]*[a-zA-Z_][^\]]*\]/.test(code) ? 'Dynamic array indexing can be slow on some GPUs' : null,
  },
  {
    id: 'ALIGN_VEC3_IN_STORAGE_ARRAYS',
    level: 'error',
    test: (code) => /var<storage[^>]*>\s+\w+:\s*array<\s*vec3<f32>\s*>/.test(code) ? 'array<vec3<f32>> in storage likely needs explicit @align(16)' : null,
    fix: (code) => code.replace(/array<\s*vec3<f32>\s*>/g, 'array<vec3<f32>, @align(16)>'),
  },
  {
    id: 'PREFER_CONST_WHERE_POSSIBLE',
    level: 'info',
    test: (code) => {
      // Cheap heuristic: let NAME = ... ; not reassigned elsewhere
      const re = /\blet\s+([A-Za-z_]\w*)\s*=/g;
      let m, hints = [];
      while ((m = re.exec(code))) {
        const name = m[1];
        const reassign = new RegExp(`\\b${name}\\s*=`, 'g');
        const count = (code.match(reassign)||[]).length;
        if (count === 1) hints.push(`'${name}' could be const`);
      }
      return hints.length ? hints.join('; ') : null;
    },
    fix: (code) => code.replace(/\blet\s+([A-Za-z_]\w*)\s*=/g, 'const $1 ='),
  },
];

// ---------- Metrics (very rough heuristics)
const METRICS = {
  INSTRUCTION_LIKE: (code) => (code.match(/[+\-*/%=<>!&|]+/g)||[]).length + (code.match(/\w+\(/g)||[]).length*2,
  LOOPS: (code) => (code.match(/\bfor\b|\bwhile\b/g)||[]).length,
  COMPLEX_MATH: (code) => (code.match(/\b(sin|cos|tan|asin|acos|atan|exp|log|pow|sqrt|inverseSqrt)\b/g)||[]).length,
  TEXTURE_SAMPLES: (code) => (code.match(/\btextureSample(Compare)?\b/g)||[]).length,
};

const METRIC_TARGETS = {INSTRUCTION_LIKE: 2000, LOOPS: 8, COMPLEX_MATH: 64, TEXTURE_SAMPLES: 64};

// ---------- Device limit checks (best-effort static checks)
const deviceLimitChecks = (code) => {
  const issues = [];
  // @workgroup_size(x,y,z)
  const m = code.match(/@workgroup_size\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)/);
  if (m) {
    const x = parseInt(m[1],10), y = parseInt(m[2],10), z = parseInt(m[3],10);
    const invocations = x*y*z;
    if (x > LIMITS_MERGED.maxComputeWorkgroupSizeX ||
        y > LIMITS_MERGED.maxComputeWorkgroupSizeY ||
        z > LIMITS_MERGED.maxComputeWorkgroupSizeZ) {
      issues.push({level:'error', msg:`@workgroup_size(${x},${y},${z}) exceeds device per-dimension limits`});
    }
    if (invocations > LIMITS_MERGED.maxComputeInvocationsPerWorkgroup) {
      issues.push({level:'error', msg:`@workgroup_size total invocations ${invocations} exceeds ${LIMITS_MERGED.maxComputeInvocationsPerWorkgroup}`});
    }
  }
  // Shared memory quick estimate: count var<workgroup> array<vecN<f32>, K>
  const workgroupDecls = code.match(/var<workgroup>\s+\w+:\s*array<\s*vec[234]<f32>\s*,\s*(\d+)\s*>/g) || [];
  const bytes = workgroupDecls.reduce((sum, decl) => {
    const m = decl.match(/vec([234])<f32>.*?,\s*(\d+)/);
    if (!m) return sum;
    const lanes = parseInt(m[1],10);
    const count = parseInt(m[2],10);
    return sum + (lanes*4)*count;
  }, 0);
  if (bytes > 0 && bytes > LIMITS_MERGED.maxWorkgroupStorageSize) {
    issues.push({level:'error', msg:`var<workgroup> usage ~${bytes} bytes exceeds device maxWorkgroupStorageSize=${LIMITS_MERGED.maxWorkgroupStorageSize}`});
  }
  return issues;
};

// ---------- Naga/Tint helpers
async function nagaValidate(file) {
  if (!(await hasBin('naga', '--version'))) {
    return {ok:false, note:'naga not found (cargo install naga-cli)'};
  }
  const {stdout, stderr} = await safeRun(`naga validate "${file}" --input-kind wgsl`);
  const all = `${stdout}\n${stderr}`.trim();
  const ok = !/error/i.test(all);
  return {ok, log: all};
}

async function tintTranspile(file, fmt) {
  if (!(await hasBin('tint', '--version'))) {
    return {ok:false, note:'tint not found (see Dawn build or get tint from Chromium)'};
  }
  // tint expects: tint --format=<hlsl|msl|spirv> input.wgsl -o -
  const {stdout, stderr} = await safeRun(`tint --format=${fmt} "${file}" -o -`);
  const all = `${stdout}\n${stderr}`.trim();
  const ok = all.length > 0 && !/error/i.test(all);
  return {ok, log: all};
}

// ---------- Fixer
function applyFixes(code) {
  if (!FIX) return code;
  let out = code;
  for (const r of RULES) {
    if (typeof r.fix === 'function') out = r.fix(out);
  }
  return out;
}

// ---------- Main
(async () => {
  if (!fs.existsSync(TARGET_DIR)) {
    console.error(`Target directory not found: ${TARGET_DIR}`);
    process.exit(1);
  }
  const files = globWGSL(TARGET_DIR);
  console.log(`🔍 Shader Quality Gate v2 — scanning ${files.length} files from ${path.relative(ROOT, TARGET_DIR)}`);

  const report = {
    dir: TARGET_DIR,
    files: [],
    totals: {errors:0, warnings:0, info:0},
    meta: {strict: STRICT, targets: TARGETS, limits: LIMITS_MERGED}
  };

  for (const file of files) {
    const src = fs.readFileSync(file, 'utf-8');
    let code = src;
    const fileRel = path.relative(ROOT, file);
    const result = {file: fileRel, errors: [], warnings: [], info: [], metrics: {}};

    // Rules & fixes
    code = applyFixes(code);
    if (code !== src) {
      fs.writeFileSync(file, code, 'utf-8');
      result.info.push('Auto-fixed with built-in fixer');
    }
    for (const rule of RULES) {
      const msg = rule.test(code);
      if (!msg) continue;
      if (rule.level === 'error') result.errors.push(`${rule.id}: ${msg}`);
      else if (rule.level === 'warning') result.warnings.push(`${rule.id}: ${msg}`);
      else result.info.push(`${rule.id}: ${msg}`);
    }

    // Device limit checks
    for (const iss of deviceLimitChecks(code)) {
      (iss.level === 'error' ? result.errors : result.warnings).push(`DEVICE_LIMIT: ${iss.msg}`);
    }

    // Metrics
    for (const [k, fn] of Object.entries(METRICS)) {
      const v = fn(code);
      result.metrics[k] = v;
      const target = METRIC_TARGETS[k];
      if (typeof target === 'number' && v > target) {
        result.warnings.push(`METRIC_${k}: ${v} exceeds target ${target}`);
      }
    }

    // Naga validate
    const naga = await nagaValidate(file);
    if (!naga.ok) {
      result.errors.push(`NAGA: validation failed or missing — ${naga.log || naga.note}`.trim());
    }

    // Tint transpile (optional per-target sanity)
    for (const t of TARGETS) {
      const r = await tintTranspile(file, t);
      if (!r.ok) {
        result.errors.push(`TINT(${t}): transpile failed — ${r.log || r.note}`.trim());
      } else {
        result.info.push(`TINT(${t}): ok`);
      }
    }

    report.files.push(result);
    report.totals.errors += result.errors.length;
    report.totals.warnings += result.warnings.length;
    report.totals.info += result.info.length;

    // Console summary per file
    const status = result.errors.length ? '❌' : (result.warnings.length ? '⚠️' : '✅');
    console.log(`${status} ${fileRel}  (E:${result.errors.length} W:${result.warnings.length})`);
    if (STRICT && result.warnings.length) {
      for (const w of result.warnings) console.log(`   ↳ WARN ${w}`);
    }
    if (result.errors.length) {
      for (const e of result.errors) console.log(`   ↳ ERROR ${e}`);
    }
  }

  // Reports
  if (REPORT_PATH) {
    fs.mkdirSync(path.dirname(REPORT_PATH), {recursive: true});
    fs.writeFileSync(REPORT_PATH, JSON.stringify(report, null, 2));
    console.log(`\n📝 Wrote JSON report → ${path.relative(ROOT, REPORT_PATH)}`);
  }
  if (JUNIT_PATH) {
    const xml = [
      '<?xml version="1.0" encoding="UTF-8"?>',
      `<testsuite name="shader_quality_gate" tests="${report.files.length}" failures="${report.totals.errors}">`,
      ...report.files.map(f => [
        `<testcase name="${f.file}">`,
        ...f.errors.map(e => `<failure message="${escapeXml(e)}"/>`),
        `</testcase>`
      ].join('\n')),
      '</testsuite>'
    ].join('\n');
    fs.mkdirSync(path.dirname(JUNIT_PATH), {recursive: true});
    fs.writeFileSync(JUNIT_PATH, xml);
    console.log(`📝 Wrote JUnit report → ${path.relative(ROOT, JUNIT_PATH)}`);
  }

  // Exit code
  const exit = report.totals.errors ? 1 : (STRICT && report.totals.warnings ? 2 : 0);
  process.exit(exit);
})().catch(e => {
  console.error(e);
  process.exit(1);
});

function escapeXml(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
"""
out = Path("/mnt/data/shader_quality_gate_v2.mjs")
out.write_text(code)
print(str(out))
