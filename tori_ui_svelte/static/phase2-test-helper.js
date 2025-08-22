// Phase 2 Test Helper - Paste in Browser Console
// This script helps verify Phase 2 functionality

(function() {
    console.clear();
    console.log('%c🚀 iRis Phase 2 Test Helper', 'font-size: 20px; color: #00ff88; font-weight: bold');
    console.log('%c==========================', 'color: #666');
    
    // Current state check
    const plan = localStorage.getItem('iris.plan') || 'free';
    const planColors = { free: '#888', plus: '#10b981', pro: '#8b5cf6' };
    
    console.log(`%c📋 Current Plan: ${plan.toUpperCase()}`, `font-size: 14px; color: ${planColors[plan]}; font-weight: bold`);
    
    // Test utilities
    window.irisTest = {
        // Quick plan switchers
        setFree: () => {
            localStorage.setItem('iris.plan', 'free');
            console.log('%c✓ Switched to FREE plan', 'color: #888');
            console.log('  → Refresh page to apply');
            return 'Reload page now';
        },
        
        setPlus: () => {
            localStorage.setItem('iris.plan', 'plus');
            console.log('%c✓ Switched to PLUS plan', 'color: #10b981');
            console.log('  → Refresh page to apply');
            return 'Reload page now';
        },
        
        setPro: () => {
            localStorage.setItem('iris.plan', 'pro');
            console.log('%c✓ Switched to PRO plan', 'color: #8b5cf6');
            console.log('  → Refresh page to apply');
            return 'Reload page now';
        },
        
        // Verify current state
        checkSetup: () => {
            console.log('%c📊 Setup Verification:', 'font-size: 14px; color: #ffa500; font-weight: bold');
            
            // Check recorder
            const recorder = document.querySelector('.recorder-bar');
            if (recorder) {
                console.log('  ✅ Recorder component found');
                const button = recorder.querySelector('button');
                if (button) {
                    console.log(`  ✅ Record button: "${button.textContent}"`);
                }
            } else {
                console.log('  ⚠️ Recorder not found (navigate to /hologram)');
            }
            
            // Check canvas
            const canvas = document.querySelector('#holo-canvas');
            if (canvas) {
                console.log('  ✅ Hologram canvas found');
            } else {
                console.log('  ⚠️ Canvas not found');
            }
            
            // Check MediaRecorder support
            if (typeof MediaRecorder !== 'undefined') {
                console.log('  ✅ MediaRecorder supported');
                const webm = MediaRecorder.isTypeSupported('video/webm');
                console.log(`  ${webm ? '✅' : '❌'} WebM recording supported`);
            } else {
                console.log('  ❌ MediaRecorder not supported');
            }
            
            return 'Setup check complete';
        },
        
        // Test recordings
        testRecording: () => {
            const button = document.querySelector('.recorder-bar button');
            if (button && button.textContent.includes('Record')) {
                console.log('%c🎬 Starting test recording...', 'color: #00ff88');
                button.click();
                setTimeout(() => {
                    const stopBtn = document.querySelector('.recorder-bar button.stop');
                    if (stopBtn) {
                        console.log('  → Recording for 3 seconds...');
                        setTimeout(() => {
                            stopBtn.click();
                            console.log('  ✅ Recording stopped');
                            console.log('  → Check downloads for iris_*.webm');
                        }, 3000);
                    }
                }, 100);
            } else {
                console.log('  ❌ Record button not found');
            }
        },
        
        // Verify Stripe
        checkStripe: async () => {
            console.log('%c💳 Testing Stripe endpoint...', 'color: #635bff');
            try {
                const res = await fetch('/api/billing/checkout', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ planId: 'plus' })
                });
                
                if (res.ok) {
                    const data = await res.json();
                    if (data.url) {
                        console.log('  ✅ Stripe configured correctly');
                        console.log('  → Checkout URL generated');
                    } else {
                        console.log('  ⚠️ No checkout URL returned');
                    }
                } else {
                    console.log('  ❌ Stripe endpoint error:', res.status);
                    console.log('  → Check .env configuration');
                }
            } catch (err) {
                console.log('  ❌ Failed to reach Stripe endpoint');
                console.error(err);
            }
        },
        
        // Quick navigation
        nav: {
            hologram: () => location.href = '/hologram',
            studio: () => location.href = '/hologram-studio',
            pricing: () => location.href = '/pricing',
            home: () => location.href = '/'
        },
        
        // Test summary
        runAllTests: async () => {
            console.log('%c🔍 Running All Phase 2 Tests...', 'font-size: 16px; color: #00ff88; font-weight: bold');
            console.log('');
            
            // Check current plan
            const plan = localStorage.getItem('iris.plan') || 'free';
            console.log(`1️⃣ Current Plan: ${plan.toUpperCase()}`);
            
            // Check setup
            irisTest.checkSetup();
            console.log('');
            
            // Check Stripe
            await irisTest.checkStripe();
            console.log('');
            
            console.log('%c✅ Test suite complete!', 'font-size: 14px; color: #00ff88; font-weight: bold');
            console.log('');
            console.log('Quick Commands:');
            console.log('  irisTest.setFree()   - Switch to Free');
            console.log('  irisTest.setPlus()   - Switch to Plus');
            console.log('  irisTest.setPro()    - Switch to Pro');
            console.log('  irisTest.nav.pricing() - Go to pricing');
            console.log('  irisTest.testRecording() - Test record');
        }
    };
    
    // Auto-run tests
    irisTest.runAllTests();
    
    // Add test card reminder
    console.log('');
    console.log('%c💳 Test Card: 4242 4242 4242 4242', 'background: #635bff; color: white; padding: 4px 8px; border-radius: 4px');
    console.log('   Expiry: 12/34, CVC: 123');
})();