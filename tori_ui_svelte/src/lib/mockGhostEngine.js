// mockGhostEngine.js - Fallback when WebGPU isn't available or has errors

export class GhostEngine {
    constructor() {
        console.log('🎭 Using Mock Ghost Engine (WebGPU fallback)');
        this.initialized = false;
        this.mockConcepts = [];
    }

    async initialize() {
        console.log('🎭 Mock Ghost Engine initializing...');
        
        // Check if WebGPU is available
        if (!navigator.gpu) {
            console.warn('⚠️ WebGPU not available - using mock mode');
        }
        
        // Simulate some default concepts
        this.mockConcepts = [
            { id: 'mock-1', name: 'Consciousness', x: 0.2, y: 0.3, z: 0.0, energy: 0.8 },
            { id: 'mock-2', name: 'Knowledge', x: -0.3, y: 0.2, z: 0.1, energy: 0.6 },
            { id: 'mock-3', name: 'Memory', x: 0.1, y: -0.2, z: -0.1, energy: 0.7 },
            { id: 'mock-4', name: 'Learning', x: -0.2, y: -0.3, z: 0.2, energy: 0.5 }
        ];
        
        this.initialized = true;
        console.log('✅ Mock Ghost Engine ready');
        return this;
    }

    async updateConcepts(concepts) {
        console.log('🎭 Mock updating concepts:', concepts.length);
        this.mockConcepts = concepts.map((c, i) => ({
            ...c,
            x: Math.sin(i * 0.5) * 0.5,
            y: Math.cos(i * 0.7) * 0.5,
            z: Math.sin(i * 0.3) * 0.2,
            energy: 0.5 + Math.random() * 0.5
        }));
    }

    async render(canvas) {
        if (!canvas || !this.initialized) return;
        
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        
        // Clear canvas
        ctx.fillStyle = '#000033';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Draw mock hologram effect
        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        const time = Date.now() * 0.001;
        
        // Draw concepts as glowing circles
        this.mockConcepts.forEach((concept, i) => {
            const x = centerX + concept.x * 200 + Math.sin(time + i) * 10;
            const y = centerY + concept.y * 200 + Math.cos(time + i) * 10;
            const radius = 20 + concept.energy * 20;
            
            // Glow effect
            const gradient = ctx.createRadialGradient(x, y, 0, x, y, radius * 2);
            gradient.addColorStop(0, `rgba(100, 200, 255, ${concept.energy})`);
            gradient.addColorStop(0.5, `rgba(50, 150, 255, ${concept.energy * 0.5})`);
            gradient.addColorStop(1, 'rgba(0, 50, 100, 0)');
            
            ctx.fillStyle = gradient;
            ctx.beginPath();
            ctx.arc(x, y, radius * 2, 0, Math.PI * 2);
            ctx.fill();
            
            // Core
            ctx.fillStyle = '#ffffff';
            ctx.beginPath();
            ctx.arc(x, y, radius * 0.3, 0, Math.PI * 2);
            ctx.fill();
            
            // Label
            ctx.fillStyle = '#ffffff';
            ctx.font = '12px Arial';
            ctx.textAlign = 'center';
            ctx.fillText(concept.name || `Concept ${i}`, x, y + radius + 20);
        });
        
        // Add some particle effects
        for (let i = 0; i < 50; i++) {
            const px = Math.random() * canvas.width;
            const py = Math.random() * canvas.height;
            const brightness = Math.random();
            
            ctx.fillStyle = `rgba(150, 200, 255, ${brightness * 0.5})`;
            ctx.fillRect(px, py, 1, 1);
        }
    }

    async connectAudioStream(stream) {
        console.log('🎭 Mock audio stream connected');
        // Just log it, don't actually process audio
    }

    destroy() {
        console.log('🎭 Mock Ghost Engine destroyed');
        this.initialized = false;
        this.mockConcepts = [];
    }
}

// Auto-detect and export the right engine
export async function createGhostEngine() {
    try {
        // Try to load the real engine first
        const { GhostEngine: RealEngine } = await import('./realGhostEngine.js');
        const engine = new RealEngine();
        await engine.initialize();
        console.log('✅ Using real Ghost Engine');
        return engine;
    } catch (error) {
        console.warn('⚠️ Real Ghost Engine failed, using mock:', error.message);
        const engine = new GhostEngine();
        await engine.initialize();
        return engine;
    }
}

export default GhostEngine;
