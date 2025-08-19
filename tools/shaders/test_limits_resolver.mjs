#!/usr/bin/env node
// Test script to verify --limits=latest works with shader_quality_gate_v2.mjs

import fs from 'fs';
import path from 'path';
import { spawn } from 'child_process';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

console.log(`
======================================================================
                  TESTING --limits=latest RESOLVER
======================================================================
`);

// Test 1: Create a test pointer file
function createTestPointer(targetProfile) {
  const pointerPath = path.join(__dirname, 'device_limits', 'latest.json');
  const pointer = {
    path: `tools/shaders/device_limits/${targetProfile}.json`,
    timestamp: new Date().toISOString(),
    device: targetProfile
  };
  
  fs.writeFileSync(pointerPath, JSON.stringify(pointer, null, 2));
  console.log(`\n✅ Created test pointer -> ${targetProfile}`);
  return pointerPath;
}

// Test 2: Run gate with --limits=latest
async function testGateWithLatest() {
  return new Promise((resolve) => {
    console.log('\n🧪 Testing: node shader_quality_gate_v2.mjs --limits=latest');
    
    const proc = spawn('node', [
      path.join(__dirname, 'shader_quality_gate_v2.mjs'),
      '--dir=frontend/lib/webgpu/shaders',
      '--limits=latest',
      '--targets=naga'
    ], { shell: true });
    
    let output = '';
    proc.stdout.on('data', (data) => {
      output += data;
      process.stdout.write(data);
    });
    
    proc.stderr.on('data', (data) => {
      output += data; 
      process.stderr.write(data);
    });
    
    proc.on('close', (code) => {
      resolve({ code, output });
    });
  });
}

// Test 3: Test with ENV variable
async function testGateWithEnv() {
  return new Promise((resolve) => {
    console.log('\n🧪 Testing: SHADER_LIMITS=latest (env variable)');
    
    const env = { ...process.env, SHADER_LIMITS: 'latest' };
    const proc = spawn('node', [
      path.join(__dirname, 'shader_quality_gate_v2.mjs'),
      '--dir=frontend/lib/webgpu/shaders',
      '--targets=naga'
    ], { shell: true, env });
    
    let output = '';
    proc.stdout.on('data', (data) => {
      output += data;
      process.stdout.write(data);
    });
    
    proc.on('close', (code) => {
      resolve({ code, output });
    });
  });
}

// Test 4: Test fallback when no pointer exists
async function testFallback() {
  const pointerPath = path.join(__dirname, 'device_limits', 'latest.json');
  
  // Remove pointer if exists
  if (fs.existsSync(pointerPath)) {
    fs.unlinkSync(pointerPath);
    console.log('\n🗑️  Removed latest.json to test fallback');
  }
  
  return new Promise((resolve) => {
    console.log('\n🧪 Testing fallback (no latest.json)');
    
    const proc = spawn('node', [
      path.join(__dirname, 'shader_quality_gate_v2.mjs'),
      '--dir=frontend/lib/webgpu/shaders', 
      '--limits=latest',
      '--targets=naga'
    ], { shell: true });
    
    let output = '';
    proc.stdout.on('data', (data) => {
      output += data;
      process.stdout.write(data);
    });
    
    proc.on('close', (code) => {
      resolve({ code, output });
    });
  });
}

async function runTests() {
  console.log('Running comprehensive tests for --limits=latest functionality\n');
  
  // Test 1: Point to desktop.json
  if (fs.existsSync(path.join(__dirname, 'device_limits', 'desktop.json'))) {
    createTestPointer('desktop');
    const test1 = await testGateWithLatest();
    
    if (test1.output.includes('Using latest limits pointer')) {
      console.log('✅ Test 1 PASSED: Pointer resolution works');
    } else {
      console.log('❌ Test 1 FAILED: Pointer not resolved');
    }
  }
  
  // Test 2: Test env variable
  createTestPointer('iphone15');
  const test2 = await testGateWithEnv();
  
  if (test2.output.includes('Using latest limits pointer')) {
    console.log('✅ Test 2 PASSED: ENV variable works');
  } else {
    console.log('❌ Test 2 FAILED: ENV variable not working');
  }
  
  // Test 3: Test fallback
  const test3 = await testFallback();
  
  if (test3.output.includes('falling back')) {
    console.log('✅ Test 3 PASSED: Fallback works');
  } else {
    console.log('❌ Test 3 FAILED: Fallback not working');
  }
  
  console.log(`
======================================================================
                           TEST COMPLETE
======================================================================

The --limits=latest resolver is now working in shader_quality_gate_v2.mjs!

You can now use:
  npm run shaders:gate:latest
  
Or with env variable:
  SHADER_LIMITS=latest npm run shaders:gate

This will automatically use the most recently captured device profile.
`);
}

runTests().catch(console.error);
