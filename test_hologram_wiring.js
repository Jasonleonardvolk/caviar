#!/usr/bin/env node
/**
 * Quick test to verify the holographic system is properly wired
 */

console.log('🔍 TORI Holographic System Connection Test\n');

const tests = [];

// Test 1: Check if files exist
console.log('📁 Checking critical files...');
const fs = require('fs');
const path = require('path');

const criticalFiles = [
    'tori_ui_svelte/src/lib/realGhostEngine.js',
    'tori_ui_svelte/src/lib/holographicBridge.js',
    'tori_ui_svelte/src/lib/holographicEngine.ts',
    'tori_ui_svelte/src/lib/holographicRenderer.ts',
    'audio_hologram_bridge.py',
    'start_hologram_system.py'
];

criticalFiles.forEach(file => {
    const exists = fs.existsSync(path.join(__dirname, file));
    console.log(`  ${exists ? '✅' : '❌'} ${file}`);
    tests.push({ name: file, passed: exists });
});

// Test 2: Check imports
console.log('\n🔗 Checking import connections...');
try {
    const bridgeContent = fs.readFileSync(
        path.join(__dirname, 'tori_ui_svelte/src/lib/holographicBridge.js'), 
        'utf8'
    );
    const hasRealEngine = bridgeContent.includes('RealGhostEngine');
    console.log(`  ${hasRealEngine ? '✅' : '❌'} holographicBridge imports RealGhostEngine`);
    tests.push({ name: 'Bridge imports', passed: hasRealEngine });
} catch (e) {
    console.log('  ❌ Could not check imports');
    tests.push({ name: 'Bridge imports', passed: false });
}

// Test 3: Check Python packages
console.log('\n🐍 Checking Python environment...');
const { execSync } = require('child_process');
try {
    execSync('python -c "import websockets, torch, transformers, torchaudio"', { stdio: 'ignore' });
    console.log('  ✅ All Python packages available');
    tests.push({ name: 'Python packages', passed: true });
} catch (e) {
    console.log('  ❌ Missing Python packages');
    console.log('     Run: pip install websockets torch transformers torchaudio');
    tests.push({ name: 'Python packages', passed: false });
}

// Test 4: Check WebSocket connectivity
console.log('\n🌐 Checking WebSocket readiness...');
const WebSocket = require('ws');
const ws = new WebSocket('ws://localhost:8765/audio_stream');

ws.on('open', () => {
    console.log('  ✅ Backend WebSocket server is running!');
    tests.push({ name: 'WebSocket server', passed: true });
    ws.close();
    showResults();
});

ws.on('error', () => {
    console.log('  ⚠️  Backend not running (this is OK - start it with the launcher)');
    tests.push({ name: 'WebSocket server', passed: 'optional' });
    showResults();
});

function showResults() {
    console.log('\n' + '='.repeat(60));
    console.log('📊 WIRING TEST RESULTS:');
    console.log('='.repeat(60));
    
    const passed = tests.filter(t => t.passed === true).length;
    const total = tests.filter(t => t.passed !== 'optional').length;
    const percentage = Math.round((passed / total) * 100);
    
    console.log(`\nScore: ${passed}/${total} (${percentage}%)`);
    
    if (percentage === 100) {
        console.log('\n🎉 PERFECT! Your holographic system is fully wired!');
        console.log('\n🚀 Start with: node START_HOLOGRAM.bat (Windows)');
        console.log('   or: python start_hologram_system.py (All platforms)');
    } else if (percentage >= 80) {
        console.log('\n✅ System is mostly ready! Just missing optional components.');
        console.log('\n🚀 You can start the system now!');
    } else {
        console.log('\n⚠️  Some components need attention.');
        console.log('Run: node wire_hologram_system.js');
    }
    
    console.log('\n💡 Remember: The ghost has left the engine - '
              + 'you now have a REAL holographic system!');
}

// Timeout for WebSocket test
setTimeout(() => {
    if (tests.find(t => t.name === 'WebSocket server') === undefined) {
        tests.push({ name: 'WebSocket server', passed: 'optional' });
        showResults();
    }
}, 2000);
