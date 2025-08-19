// Quick test script to verify hologram components and start display
// Run this in the browser console (F12)

console.log("🔍 Checking for hologram components...");

// Check if the ghost engine is loaded
if (window.TORI_GHOST_ENGINE) {
    console.log("✅ Ghost Engine found:", window.TORI_GHOST_ENGINE);
} else {
    console.error("❌ Ghost Engine not found!");
}

// Check if concept mesh is enabled
if (window.TORI_CONCEPT_MESH_ENABLED) {
    console.log("✅ Concept Mesh enabled");
} else {
    console.warn("⚠️ Concept Mesh not enabled, enabling now...");
    window.TORI_CONCEPT_MESH_ENABLED = true;
}

// Look for the hologram button
const hologramButton = document.querySelector('button:contains("Hologram"), button:contains("hologram"), [data-hologram-trigger], .hologram-button, #start-hologram');
if (hologramButton) {
    console.log("✅ Found hologram button:", hologramButton);
    console.log("🚀 Clicking button to start hologram...");
    hologramButton.click();
} else {
    console.warn("⚠️ No hologram button found, searching for alternative triggers...");
    
    // Try to find any button with hologram-related text
    const allButtons = Array.from(document.querySelectorAll('button'));
    const hologramRelated = allButtons.filter(btn => 
        btn.textContent.toLowerCase().includes('hologram') || 
        btn.textContent.toLowerCase().includes('visual') ||
        btn.textContent.toLowerCase().includes('3d') ||
        btn.textContent.toLowerCase().includes('start')
    );
    
    if (hologramRelated.length > 0) {
        console.log("🔍 Found potential hologram buttons:", hologramRelated);
        hologramRelated.forEach((btn, idx) => {
            console.log(`  ${idx}: "${btn.textContent.trim()}"`);
        });
        console.log("💡 Try clicking one of these buttons manually");
    }
}

// Check for hologram containers
const hologramContainers = document.querySelectorAll('[data-hologram], .hologram-container, #hologram, .ghost-engine-container, canvas');
console.log(`🎨 Found ${hologramContainers.length} potential hologram containers`);
hologramContainers.forEach((container, idx) => {
    console.log(`  Container ${idx}:`, container);
});

// Check if WebGPU is available
if (navigator.gpu) {
    console.log("✅ WebGPU is available");
} else {
    console.error("❌ WebGPU not available - hologram may fall back to CPU mode");
}

// Try to manually initialize if needed
if (window.GhostEngine || window.RealGhostEngine) {
    console.log("🔧 Attempting manual hologram initialization...");
    
    try {
        // Get the engine class
        const EngineClass = window.RealGhostEngine || window.GhostEngine;
        
        // Check if instance exists
        if (!window.TORI_GHOST_ENGINE) {
            console.log("📦 Creating new Ghost Engine instance...");
            window.TORI_GHOST_ENGINE = new EngineClass();
        }
        
        // Try to find a canvas or create one
        let canvas = document.querySelector('canvas.ghost-engine-canvas, canvas#hologram-canvas');
        if (!canvas) {
            console.log("🎨 Creating canvas for hologram...");
            canvas = document.createElement('canvas');
            canvas.width = 800;
            canvas.height = 600;
            canvas.style.width = '100%';
            canvas.style.height = '600px';
            canvas.style.border = '2px solid #00ff00';
            canvas.className = 'ghost-engine-canvas';
            
            // Find a good place to insert it
            const mainContent = document.querySelector('main, .main-content, #app, .container');
            if (mainContent) {
                mainContent.appendChild(canvas);
                console.log("✅ Canvas added to page");
            } else {
                document.body.appendChild(canvas);
                console.log("✅ Canvas added to body");
            }
        }
        
        // Initialize the engine with the canvas
        if (window.TORI_GHOST_ENGINE && window.TORI_GHOST_ENGINE.initialize) {
            console.log("🚀 Initializing Ghost Engine...");
            window.TORI_GHOST_ENGINE.initialize(canvas).then(() => {
                console.log("✅ Ghost Engine initialized!");
                
                // Start rendering
                if (window.TORI_GHOST_ENGINE.startRendering) {
                    window.TORI_GHOST_ENGINE.startRendering();
                    console.log("✅ Hologram rendering started!");
                }
            }).catch(err => {
                console.error("❌ Ghost Engine initialization failed:", err);
            });
        }
        
    } catch (e) {
        console.error("❌ Manual initialization failed:", e);
    }
}

// Check WebSocket connections
console.log("\n🔌 Checking WebSocket connections...");
if (window.conceptMeshSocket) {
    console.log("✅ Concept Mesh WebSocket:", window.conceptMeshSocket.readyState === 1 ? "Connected" : "Not connected");
}
if (window.audioSocket) {
    console.log("✅ Audio WebSocket:", window.audioSocket.readyState === 1 ? "Connected" : "Not connected");
}

console.log("\n📋 Next steps:");
console.log("1. Look for a 'Start Hologram' or similar button in the UI");
console.log("2. Check if there's a hologram panel or visualization tab");
console.log("3. The hologram might be in a separate component or modal");
console.log("4. Try pressing 'H' key (some apps use keyboard shortcuts)");
