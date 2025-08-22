// Test script for dynamics integration
// Run this in the browser console to test the dynamics analyzers

// Test 1: Send some test phase vectors
console.log('🧪 Testing Dynamics Integration...');

// Simulate phase changes with varying patterns
let testPhase = 0;
let testAmplitude = 0.5;
let testFrequency = 0.3;

const testInterval = setInterval(() => {
  // Create oscillating patterns
  testPhase += 0.1;
  testAmplitude = 0.5 + 0.4 * Math.sin(testPhase);
  testFrequency = 0.3 + 0.2 * Math.cos(testPhase * 0.7);
  
  // Dispatch a test phase event
  document.dispatchEvent(new CustomEvent('tori-soliton-phase-change', {
    detail: {
      userId: 'test-user',
      phase: 'test',
      phaseAngle: testPhase % (2 * Math.PI),
      amplitude: testAmplitude,
      frequency: testFrequency,
      stability: 0.8 + 0.2 * Math.sin(testPhase * 0.3),
      valence: Math.sin(testPhase * 1.5)
    }
  }));
  
  // Add some chaos occasionally
  if (Math.random() < 0.1) {
    testAmplitude += Math.random() * 2 - 1; // Random spike
  }
}, 100);

// Stop after 30 seconds
setTimeout(() => {
  clearInterval(testInterval);
  console.log('✅ Test complete! Check the Dynamics Analysis panel in the debug view.');
}, 30000);

// Test 2: Listen for updates
document.addEventListener('tori-koopman-update', (e) => {
  console.log('📊 Koopman Update:', e.detail);
});

document.addEventListener('tori-lyapunov-spike', (e) => {
  console.log('⚡ Lyapunov Spike:', e.detail);
});

console.log('🧪 Test running... Open the debug panel (🧠 button) to see live results.');
