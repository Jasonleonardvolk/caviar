// Browser helper for video recording
// Paste in console during recording session

(function() {
    console.clear();
    console.log('%c📹 iRis Video Recording Helper', 'font-size: 20px; color: #ff0066; font-weight: bold');
    console.log('%c==========================', 'color: #666');
    
    // Video recording helper
    window.videoHelper = {
        // Switch to Free for Video A
        setupVideoA: () => {
            localStorage.setItem('iris.plan', 'free');
            console.log('%c✓ Set to FREE plan (watermark visible)', 'color: #888');
            console.log('  → Reload page and record 10s');
            console.log('  → Watermark should be visible');
            location.reload();
        },
        
        // Switch to Plus for Video B
        setupVideoB: () => {
            localStorage.setItem('iris.plan', 'plus');
            console.log('%c✓ Set to PLUS plan (no watermark)', 'color: #10b981');
            console.log('  → Reload page and record 20-30s');
            console.log('  → NO watermark should appear');
            location.reload();
        },
        
        // Create visual effects for recording
        addVisualEffects: () => {
            // Add glow effect to canvas
            const canvas = document.querySelector('#holo-canvas');
            if (canvas) {
                canvas.style.filter = 'contrast(1.2) saturate(1.3)';
                canvas.style.boxShadow = '0 0 40px rgba(0, 255, 255, 0.5)';
                console.log('✓ Added visual effects to canvas');
            }
            
            // Enhance recorder bar
            const recorder = document.querySelector('.recorder-bar');
            if (recorder) {
                recorder.style.background = 'linear-gradient(135deg, rgba(16,185,129,0.9), rgba(139,92,246,0.9))';
                recorder.style.backdropFilter = 'blur(10px)';
                console.log('✓ Enhanced recorder bar');
            }
        },
        
        // Auto-start recording
        startRecording: (seconds = 10) => {
            const button = document.querySelector('.recorder-bar button');
            if (button && button.textContent.includes('Record')) {
                console.log(`%c🎬 Starting ${seconds}s recording...`, 'color: #ff0066');
                button.click();
                
                // Auto-stop after specified time
                setTimeout(() => {
                    const stopBtn = document.querySelector('.recorder-bar button.stop');
                    if (stopBtn) {
                        console.log('⏹ Auto-stopping recording');
                        stopBtn.click();
                    }
                }, seconds * 1000);
            }
        },
        
        // Show recording countdown
        showCountdown: () => {
            let count = 3;
            const interval = setInterval(() => {
                if (count > 0) {
                    console.log(`%c${count}...`, 'font-size: 30px; color: #ff0066');
                    count--;
                } else {
                    console.log('%c🎬 ACTION!', 'font-size: 30px; color: #00ff00');
                    clearInterval(interval);
                    videoHelper.startRecording();
                }
            }, 1000);
        },
        
        // Check current setup
        checkSetup: () => {
            const plan = localStorage.getItem('iris.plan') || 'free';
            const recorder = document.querySelector('.recorder-bar');
            const canvas = document.querySelector('#holo-canvas');
            
            console.log('%c📊 Recording Setup:', 'font-size: 14px; color: #ffa500');
            console.log(`  Plan: ${plan.toUpperCase()}`);
            console.log(`  Recorder: ${recorder ? '✓' : '✗'}`);
            console.log(`  Canvas: ${canvas ? '✓' : '✗'}`);
            console.log(`  Watermark: ${plan === 'free' ? 'VISIBLE' : 'HIDDEN'}`);
            
            return {plan, recorder: !!recorder, canvas: !!canvas};
        },
        
        // Quick video setups
        videos: {
            A: () => {
                console.log('%c🎬 Setting up Video A (Shock Proof)', 'color: #ff0066; font-size: 16px');
                console.log('  - 10s recording');
                console.log('  - FREE plan (watermark)');
                console.log('  - High contrast visuals');
                videoHelper.setupVideoA();
            },
            
            B: () => {
                console.log('%c🎬 Setting up Video B (How-To)', 'color: #10b981; font-size: 16px');
                console.log('  - 20-30s recording');
                console.log('  - PLUS plan (no watermark)');
                console.log('  - Tutorial flow');
                videoHelper.setupVideoB();
            },
            
            C: () => {
                console.log('%c🎬 Video C Instructions', 'color: #8b5cf6; font-size: 16px');
                console.log('  1. Open File Explorer');
                console.log('  2. Navigate to D:\\Dev\\kha\\exports\\video\\');
                console.log('  3. Show MP4 files (5s)');
                console.log('  4. Play one file (5s)');
                console.log('  5. Record 15-20s total');
            }
        }
    };
    
    // Auto-check setup
    const setup = videoHelper.checkSetup();
    
    // Show instructions
    console.log('');
    console.log('%c🎯 Quick Commands:', 'font-size: 14px; color: #00ff88');
    console.log('  videoHelper.videos.A()    - Setup Video A (Free)');
    console.log('  videoHelper.videos.B()    - Setup Video B (Plus)');
    console.log('  videoHelper.videos.C()    - Instructions for Video C');
    console.log('  videoHelper.addVisualEffects() - Enhance visuals');
    console.log('  videoHelper.showCountdown()    - Start with countdown');
    console.log('  videoHelper.startRecording(10) - Record X seconds');
    
    // Add visual hint
    if (setup.plan === 'free') {
        console.log('');
        console.log('%c📌 Currently on FREE plan - Perfect for Video A!', 'background: #dc2626; color: white; padding: 4px 8px; border-radius: 4px');
    } else {
        console.log('');
        console.log('%c📌 Currently on ' + setup.plan.toUpperCase() + ' plan - Ready for Video B/C!', 'background: #10b981; color: white; padding: 4px 8px; border-radius: 4px');
    }
})();