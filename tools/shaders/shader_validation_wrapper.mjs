#!/usr/bin/env node
// shader_validation_wrapper.mjs
// Wraps shader_quality_gate_v2.mjs and generates comprehensive reports

import { execSync } from 'child_process';
import { readFileSync, writeFileSync, existsSync, mkdirSync, copyFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = join(__dirname, '..', '..');
const reportsDir = join(__dirname, 'reports');

// Ensure reports directory exists
if (!existsSync(reportsDir)) {
    mkdirSync(reportsDir, { recursive: true });
}

// Get timestamp
const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);

// Run validator
console.log('Running shader validator...');
let exitCode = 0;
let validatorFailed = false;

try {
    execSync(`node ${join(__dirname, 'shader_quality_gate_v2.mjs')} --dir=frontend/ --strict --targets=msl,hlsl,spirv --limits=${join(__dirname, 'device_limits.iphone15.json')} --report=build/shader_report.json --junit=build/shader_report.junit.xml`, {
        cwd: repoRoot,
        stdio: 'inherit'
    });
} catch (error) {
    if (error.status === undefined) {
        validatorFailed = true;
        exitCode = 3;
    } else {
        exitCode = error.status;
    }
}

if (validatorFailed) {
    console.error('Validator tool failed to run');
    process.exit(3);
}

// Read outputs
const jsonPath = join(repoRoot, 'build', 'shader_report.json');
const junitPath = join(repoRoot, 'build', 'shader_report.junit.xml');

if (!existsSync(jsonPath)) {
    console.error('JSON report not generated');
    process.exit(3);
}

const jsonData = JSON.parse(readFileSync(jsonPath, 'utf8'));
const junitData = existsSync(junitPath) ? readFileSync(junitPath, 'utf8') : '';

// Copy to timestamped files
const jsonDest = join(reportsDir, `shader_validation_${timestamp}.json`);
const junitDest = join(reportsDir, `shader_validation_${timestamp}.junit.xml`);
const summaryDest = join(reportsDir, `shader_validation_${timestamp}_summary.txt`);
const latestDest = join(reportsDir, 'shader_validation_latest.json');

writeFileSync(jsonDest, JSON.stringify(jsonData, null, 2));
if (junitData) writeFileSync(junitDest, junitData);
writeFileSync(latestDest, JSON.stringify(jsonData, null, 2));

// Generate summary
let summary = [];
summary.push('=== SHADER VALIDATION SUMMARY ===');
summary.push(`Timestamp: ${new Date().toISOString()}`);
summary.push(`Total Files: ${jsonData.totalFiles || 0}`);

const passed = jsonData.passed || [];
const warnings = jsonData.warnings || [];
const failed = jsonData.failed || [];

summary.push(`Status: ${passed.length} PASSED | ${failed.length} FAILED | ${warnings.length} WARNING`);
summary.push('');

// Failed section
if (failed.length > 0) {
    summary.push(`FAILED (${failed.length}):`);
    failed.forEach(file => {
        summary.push(`  X ${file.file}`);
        if (file.errors) {
            file.errors.forEach(err => {
                summary.push(`    - Line ${err.line || '?'}: ${err.message}`);
                summary.push(`      *Fix:* ${getFixSuggestion(err.message)}`);
            });
        }
    });
    summary.push('');
}

// Warning section
if (warnings.length > 0) {
    summary.push(`WARNING (${warnings.length}):`);
    warnings.forEach(file => {
        summary.push(`  ! ${file.file}`);
        if (file.warnings) {
            file.warnings.forEach(warn => {
                summary.push(`    - Line ${warn.line || '?'}: ${warn.message}`);
            });
        }
    });
    summary.push('');
}

// Passed section
if (passed.length > 0) {
    summary.push(`PASSED (${passed.length}):`);
    passed.forEach(file => {
        summary.push(`  ✓ ${file}`);
    });
}

// Write summary
writeFileSync(summaryDest, summary.join('\n'));
console.log(summary.join('\n'));

// Set exit code
if (failed.length > 0) {
    process.exit(2);
} else if (warnings.length > 0) {
    process.exit(1);
} else {
    process.exit(0);
}

function getFixSuggestion(message) {
    if (message.includes('uniform') && message.includes('stride')) {
        return 'Convert to storage buffer or add vec4 padding';
    }
    if (message.includes('textureLoad') && message.includes('mip')) {
        return 'Add ,0 as last parameter to textureLoad';
    }
    if (message.includes('workgroup') && message.includes('size')) {
        return 'Reduce @workgroup_size to 256 or less';
    }
    return 'Check shader syntax';
}
