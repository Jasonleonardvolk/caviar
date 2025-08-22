// Browser Console Test Script for Phase 2
// Copy and paste this into browser DevTools console

console.log('%c=== iRis Phase 2 Browser Tests ===', 'color: cyan; font-size: 16px; font-weight: bold');

// Test 1: Check localStorage plan
const currentPlan = localStorage.getItem('iris.plan') || 'free';
console.log(`%c[1] Current Plan: ${currentPlan}`, 'color: yellow; font-weight: bold');

// Test 2: Check if recorder component exists
const recorderBar = document.querySelector('.recorder-bar');
if (recorderBar) {
    console.log('%c[2] ✓ Recorder bar found', 'color: green');
    const planPill = recorderBar.querySelector('.pill');
    if (planPill) {
        console.log(`    Plan displayed: ${planPill.textContent}`);
    }
} else {
    console.log('%c[2] ✗ Recorder bar not found', 'color: red');
    console.log('    Make sure you are on /hologram-studio');
}

// Test 3: Check canvas
const canvas = document.querySelector('#holo-canvas');
if (canvas) {
    console.log('%c[3] ✓ Hologram canvas found', 'color: green');
    console.log(`    Dimensions: ${canvas.width}x${canvas.height}`);
} else {
    console.log('%c[3] ✗ Canvas not found', 'color: red');
}

// Test 4: MediaRecorder support
const supportedTypes = [
    'video/webm;codecs=vp9',
    'video/webm;codecs=vp8',
    'video/webm',
    'video/mp4'
];

console.log('%c[4] MediaRecorder Support:', 'color: yellow; font-weight: bold');
supportedTypes.forEach(type => {
    const supported = MediaRecorder.isTypeSupported(type);
    const icon = supported ? '✓' : '✗';
    const color = supported ? 'green' : 'red';
    console.log(`%c    ${icon} ${type}`, `color: ${color}`);
});

// Test 5: Plan utilities
console.log('%c[5] Plan Configuration:', 'color: yellow; font-weight: bold');

// Helper functions to test plan features
const testPlanFeatures = {
    setFreePlan: () => {
        localStorage.setItem('iris.plan', 'free');
        location.reload();
    },
    setPlusPlan: () => {
        localStorage.setItem('iris.plan', 'plus');
        location.reload();
    },
    setProPlan: () => {
        localStorage.setItem('iris.plan', 'pro');
        location.reload();
    },
    clearPlan: () => {
        localStorage.removeItem('iris.plan');
        location.reload();
    }
};

console.log('    Available commands:');
console.log('    - testPlanFeatures.setFreePlan()  // Switch to Free');
console.log('    - testPlanFeatures.setPlusPlan()  // Switch to Plus');
console.log('    - testPlanFeatures.setProPlan()   // Switch to Pro');
console.log('    - testPlanFeatures.clearPlan()    // Reset to Free');

// Make functions available globally
window.testPlanFeatures = testPlanFeatures;

// Test 6: Check Stripe configuration
fetch('/api/billing/checkout', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ planId: 'plus' })
})
.then(res => {
    if (res.ok) {
        console.log('%c[6] ✓ Stripe endpoint responsive', 'color: green');
    } else {
        console.log('%c[6] ⚠ Stripe endpoint returned error', 'color: orange');
        console.log('    Check .env configuration');
    }
})
.catch(err => {
    console.log('%c[6] ✗ Stripe endpoint failed', 'color: red');
    console.error(err);
});

console.log('%c=== Test Complete ===', 'color: cyan; font-size: 14px; font-weight: bold');
console.log('Use testPlanFeatures.* to switch plans for testing');