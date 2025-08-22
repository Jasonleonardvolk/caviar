// holographicEngine.js - Core holographic rendering engine for TORI
// Manages psi-states, oscillators, and holographic object visualization

export class SpectralHologramEngine {
    constructor() {
        this.holographicObjects = new Map();
        this.objectIdCounter = 0;
        
        // Core psi-state for holographic rendering
        this.psiState = {
            psi_phase: 0,
            phase_coherence: 0.8,
            oscillator_phases: new Array(12).fill(0),
            oscillator_frequencies: new Array(12).fill(0).map((_, i) => 0.1 + i * 0.05),
            coupling_strength: 0.5,
            dominant_frequency: 0.3
        };
        
        // Animation and rendering state
        this.time = 0;
        this.deltaTime = 0;
        this.lastFrameTime = performance.now();
        
        // Rendering configuration
        this.config = {
            maxOscillators: 12,
            baseFrequency: 0.1,
            frequencySpread: 0.05,
            phaseSpeed: 1.0,
            coherenceDecay: 0.01,
            resonanceThreshold: 0.7
        };
        
        // WebGL/WebGPU context (to be set by renderer)
        this.renderer = null;
        this.canvas = null;
        
        console.log('SpectralHologramEngine initialized');
    }
    
    // Initialize with a canvas and optional renderer
    async initialize(canvas, renderer = null) {
        this.canvas = canvas;
        this.renderer = renderer;
        
        // Start animation loop
        this.startAnimationLoop();
        
        console.log('Holographic engine initialized with canvas');
        return true;
    }
    
    // Add a holographic object to the scene
    addHolographicObject(params) {
        const id = params.id || `holo_${this.objectIdCounter++}`;
        
        const holographicObject = {
            id: id,
            type: params.type || 'generic',
            position: params.position || [0, 0, 0],
            rotation: params.rotation || [0, 0, 0],
            scale: params.scale || [1, 1, 1],
            
            // Holographic properties
            resonance: params.resonance || 0.5,
            frequency: params.frequency || this.psiState.dominant_frequency,
            phase: params.phase || 0,
            intensity: params.intensity || 1.0,
            color: params.color || [0.5, 0.8, 1.0],
            
            // Oscillator coupling
            oscillatorWeights: params.oscillatorWeights || new Array(this.config.maxOscillators).fill(1.0 / this.config.maxOscillators),
            
            // Metadata
            metadata: params.metadata || {},
            
            // State
            active: true,
            visible: true,
            lastUpdate: performance.now()
        };
        
        this.holographicObjects.set(id, holographicObject);
        
        // Trigger resonance with existing objects
        this.calculateResonance(holographicObject);
        
        return id;
    }
    
    // Remove a holographic object
    removeHolographicObject(id) {
        return this.holographicObjects.delete(id);
    }
    
    // Update a holographic object's properties
    updateHolographicObject(id, updates) {
        const obj = this.holographicObjects.get(id);
        if (!obj) return false;
        
        Object.assign(obj, updates);
        obj.lastUpdate = performance.now();
        
        // Recalculate resonance if frequency changed
        if (updates.frequency || updates.resonance) {
            this.calculateResonance(obj);
        }
        
        return true;
    }
    
    // Calculate resonance between objects
    calculateResonance(targetObject) {
        let totalResonance = 0;
        let resonantObjects = 0;
        
        this.holographicObjects.forEach((obj, id) => {
            if (id === targetObject.id || !obj.active) return;
            
            // Calculate frequency difference
            const freqDiff = Math.abs(obj.frequency - targetObject.frequency);
            const maxDiff = 0.1; // Maximum frequency difference for resonance
            
            if (freqDiff < maxDiff) {
                // Calculate distance-based attenuation
                const distance = this.calculateDistance(obj.position, targetObject.position);
                const attenuation = 1.0 / (1.0 + distance * 0.1);
                
                // Calculate resonance strength
                const resonanceStrength = (1.0 - freqDiff / maxDiff) * attenuation;
                
                totalResonance += resonanceStrength * obj.resonance;
                resonantObjects++;
            }
        });
        
        // Update target object's resonance
        if (resonantObjects > 0) {
            const avgResonance = totalResonance / resonantObjects;
            targetObject.resonance = targetObject.resonance * 0.7 + avgResonance * 0.3;
        }
    }
    
    // Calculate distance between two positions
    calculateDistance(pos1, pos2) {
        const dx = pos1[0] - pos2[0];
        const dy = pos1[1] - pos2[1];
        const dz = pos1[2] - pos2[2];
        return Math.sqrt(dx * dx + dy * dy + dz * dz);
    }
    
    // Update the global psi-state
    updatePsiState(deltaTime) {
        // Update main phase
        this.psiState.psi_phase += deltaTime * this.psiState.dominant_frequency * Math.PI * 2;
        
        // Update oscillator phases
        for (let i = 0; i < this.psiState.oscillator_phases.length; i++) {
            this.psiState.oscillator_phases[i] += deltaTime * this.psiState.oscillator_frequencies[i] * Math.PI * 2;
            
            // Keep phases in [0, 2π]
            if (this.psiState.oscillator_phases[i] > Math.PI * 2) {
                this.psiState.oscillator_phases[i] -= Math.PI * 2;
            }
        }
        
        // Decay coherence slightly (entropy)
        this.psiState.phase_coherence *= (1.0 - this.config.coherenceDecay * deltaTime);
        this.psiState.phase_coherence = Math.max(0.1, this.psiState.phase_coherence);
        
        // Update coupling based on active objects
        const activeCount = Array.from(this.holographicObjects.values()).filter(obj => obj.active).length;
        this.psiState.coupling_strength = Math.min(1.0, activeCount * 0.1);
    }
    
    // Update all holographic objects
    updateHolographicObjects(deltaTime) {
        this.holographicObjects.forEach(obj => {
            if (!obj.active) return;
            
            // Update object phase
            obj.phase += deltaTime * obj.frequency * Math.PI * 2;
            
            // Calculate composite oscillation
            let compositeOscillation = 0;
            for (let i = 0; i < this.config.maxOscillators; i++) {
                const oscillatorPhase = this.psiState.oscillator_phases[i];
                const weight = obj.oscillatorWeights[i];
                compositeOscillation += Math.sin(oscillatorPhase) * weight;
            }
            
            // Apply oscillation to intensity
            const baseIntensity = obj.intensity;
            obj.currentIntensity = baseIntensity * (1.0 + compositeOscillation * 0.2);
            
            // Apply resonance boost
            if (obj.resonance > this.config.resonanceThreshold) {
                obj.currentIntensity *= 1.0 + (obj.resonance - this.config.resonanceThreshold);
            }
            
            // Update color based on phase and resonance
            obj.currentColor = [
                obj.color[0] * (0.8 + 0.2 * Math.sin(obj.phase)),
                obj.color[1] * (0.8 + 0.2 * Math.sin(obj.phase + Math.PI / 3)),
                obj.color[2] * (0.8 + 0.2 * Math.sin(obj.phase + 2 * Math.PI / 3))
            ];
            
            // Boost color intensity based on resonance
            const resonanceBoost = Math.min(1.5, 1.0 + obj.resonance * 0.5);
            obj.currentColor = obj.currentColor.map(c => Math.min(1.0, c * resonanceBoost));
        });
    }
    
    // Main update loop
    update() {
        const currentTime = performance.now();
        this.deltaTime = (currentTime - this.lastFrameTime) / 1000.0; // Convert to seconds
        this.lastFrameTime = currentTime;
        this.time += this.deltaTime;
        
        // Update psi-state
        this.updatePsiState(this.deltaTime);
        
        // Update all holographic objects
        this.updateHolographicObjects(this.deltaTime);
        
        // Render if renderer is available
        if (this.renderer && typeof this.renderer.render === 'function') {
            this.renderer.render(this);
        }
    }
    
    // Start the animation loop
    startAnimationLoop() {
        const animate = () => {
            this.update();
            requestAnimationFrame(animate);
        };
        animate();
    }
    
    // Get current state for debugging
    getState() {
        return {
            time: this.time,
            objectCount: this.holographicObjects.size,
            psiState: { ...this.psiState },
            activeObjects: Array.from(this.holographicObjects.values())
                .filter(obj => obj.active)
                .map(obj => ({
                    id: obj.id,
                    type: obj.type,
                    resonance: obj.resonance,
                    intensity: obj.currentIntensity || obj.intensity
                }))
        };
    }
    
    // Set global frequency (affects all objects)
    setGlobalFrequency(frequency) {
        this.psiState.dominant_frequency = frequency;
        
        // Optionally update all objects to match
        this.holographicObjects.forEach(obj => {
            obj.frequency = frequency;
        });
    }
    
    // Boost coherence (used when concepts align)
    boostCoherence(amount = 0.1) {
        this.psiState.phase_coherence = Math.min(1.0, this.psiState.phase_coherence + amount);
    }
    
    // Create a resonance pulse at a position
    createResonancePulse(position, strength = 1.0, frequency = null) {
        const pulseFreq = frequency || this.psiState.dominant_frequency;
        
        // Affect all nearby objects
        this.holographicObjects.forEach(obj => {
            const distance = this.calculateDistance(obj.position, position);
            const influence = strength / (1.0 + distance);
            
            // Temporarily boost resonance
            obj.resonance = Math.min(1.0, obj.resonance + influence * 0.3);
            
            // Align frequency slightly
            obj.frequency = obj.frequency * 0.9 + pulseFreq * 0.1;
        });
        
        // Boost global coherence
        this.boostCoherence(strength * 0.05);
    }
    
    // Clear all objects
    clear() {
        this.holographicObjects.clear();
        this.objectIdCounter = 0;
        
        // Reset psi-state
        this.psiState.phase_coherence = 0.8;
        this.psiState.coupling_strength = 0.5;
    }
    
    // Destroy the engine
    destroy() {
        this.clear();
        this.renderer = null;
        this.canvas = null;
        console.log('SpectralHologramEngine destroyed');
    }
}

// Also export a simpler interface for basic holographic rendering
export class ToriHolographicRenderer {
    constructor(engine) {
        this.engine = engine || new SpectralHologramEngine();
        this.renderMode = 'webgl'; // or 'webgpu', 'canvas2d'
    }
    
    async initialize(canvas) {
        return await this.engine.initialize(canvas, this);
    }
    
    render(engine) {
        // Override this in subclasses for actual rendering
        // This is just a placeholder that logs the state
        if (Math.random() < 0.01) { // Log occasionally
            console.log('Holographic render state:', engine.getState());
        }
    }
    
    addConcept(conceptData) {
        return this.engine.addHolographicObject({
            type: 'concept',
            ...conceptData
        });
    }
    
    addRelation(relationData) {
        return this.engine.addHolographicObject({
            type: 'relation',
            ...relationData
        });
    }
    
    updateConcept(id, updates) {
        return this.engine.updateHolographicObject(id, updates);
    }
    
    removeConcept(id) {
        return this.engine.removeHolographicObject(id);
    }
    
    createPulse(position, strength) {
        this.engine.createResonancePulse(position, strength);
    }
    
    getEngine() {
        return this.engine;
    }
}

export default SpectralHologramEngine;
