/**
 * Enhanced ψ-memory frames for holographic integration
 * Review and enhancement of existing implementation
 */

class PsiFrame {
    constructor(timestamp, psiState, hologramHints, emotionalResonance) {
        this.timestamp = timestamp;
        this.psiState = psiState;
        this.hologramHints = hologramHints;
        this.emotionalResonance = emotionalResonance;
        this.coherenceHistory = [];
        this.phaseEvolution = [];
        // Add new fields for enhanced tracking
        this.oscillatorHistory = [];
        this.hologramicMoments = [];
    }
    
    updateCoherence(newCoherence) {
        this.coherenceHistory.push({
            timestamp: Date.now(),
            coherence: newCoherence,
            delta: this.coherenceHistory.length > 0 
                ? newCoherence - this.coherenceHistory[this.coherenceHistory.length - 1].coherence 
                : 0
        });
        
        // Keep only last 100 measurements for memory efficiency
        if (this.coherenceHistory.length > 100) {
            this.coherenceHistory.shift();
        }
        
        // Check for holographic moment (high coherence spike)
        if (newCoherence > 0.85 && this.coherenceHistory.length > 1) {
            const prevCoherence = this.coherenceHistory[this.coherenceHistory.length - 2].coherence;
            if (newCoherence - prevCoherence > 0.2) {
                this.markHolographicMoment('coherence_spike');
            }
        }
    }
    
    evolvePhase(deltaTime) {
        // Evolve ψ-phase based on Koopman operator dynamics
        const phaseVelocity = this.calculatePhaseVelocity();
        const oldPhase = this.psiState.psi_phase;
        this.psiState.psi_phase += phaseVelocity * deltaTime;
        
        // Wrap phase to [0, 2π]
        this.psiState.psi_phase = this.psiState.psi_phase % (2 * Math.PI);
        if (this.psiState.psi_phase < 0) {
            this.psiState.psi_phase += 2 * Math.PI;
        }
        
        this.phaseEvolution.push({
            timestamp: Date.now(),
            phase: this.psiState.psi_phase,
            velocity: phaseVelocity,
            deltaPhase: this.psiState.psi_phase - oldPhase
        });
        
        // Limit history size
        if (this.phaseEvolution.length > 200) {
            this.phaseEvolution.shift();
        }
    }
    
    calculatePhaseVelocity() {
        // Compute phase velocity from oscillator coupling
        const oscillatorPhases = this.psiState.oscillator_phases || [];
        const coupling = this.psiState.phase_coherence || 0.5;
        
        if (oscillatorPhases.length < 2) {
            return 0;
        }
        
        let totalVelocity = 0;
        let pairCount = 0;
        
        // Kuramoto coupling calculation
        for (let i = 0; i < oscillatorPhases.length; i++) {
            for (let j = i + 1; j < oscillatorPhases.length; j++) {
                const phaseDiff = oscillatorPhases[j] - oscillatorPhases[i];
                totalVelocity += Math.sin(phaseDiff) * coupling;
                pairCount++;
            }
        }
        
        // Add natural frequency contribution
        const naturalFrequency = this.psiState.dominant_frequency || 440;
        const naturalContribution = (naturalFrequency / 440) * 0.1; // Normalized contribution
        
        return pairCount > 0 ? (totalVelocity / pairCount) + naturalContribution : naturalContribution;
    }
    
    updateOscillatorState(newOscillatorPhases) {
        if (!Array.isArray(newOscillatorPhases)) return;
        
        this.oscillatorHistory.push({
            timestamp: Date.now(),
            phases: [...newOscillatorPhases],
            synchronization: this.calculateSynchronization(newOscillatorPhases)
        });
        
        // Keep limited history
        if (this.oscillatorHistory.length > 50) {
            this.oscillatorHistory.shift();
        }
        
        this.psiState.oscillator_phases = newOscillatorPhases;
    }
    
    calculateSynchronization(phases) {
        if (!phases || phases.length < 2) return 0;
        
        // Calculate Kuramoto order parameter (synchronization measure)
        let sumCos = 0, sumSin = 0;
        
        phases.forEach(phase => {
            sumCos += Math.cos(phase);
            sumSin += Math.sin(phase);
        });
        
        const r = Math.sqrt(sumCos * sumCos + sumSin * sumSin) / phases.length;
        return r; // 0 = no sync, 1 = perfect sync
    }
    
    markHolographicMoment(type = 'general', metadata = {}) {
        this.hologramicMoments.push({
            timestamp: Date.now(),
            type: type,
            psiState: { ...this.psiState },
            emotionalResonance: { ...this.emotionalResonance },
            metadata: metadata
        });
        
        // Limit stored moments
        if (this.hologramicMoments.length > 20) {
            this.hologramicMoments.shift();
        }
    }
    
    toHologramData() {
        // Enhanced hologram data generation
        const baseData = {
            position3D: this.hologramHints?.position_3d || { x: 0, y: 0, z: 0 },
            colorHSL: this.hologramHints?.color_hsl || { h: 0, s: 50, l: 50 },
            volumetricDensity: this.hologramHints?.volumetric_density || [],
            oscillatorVisualization: this.hologramHints?.oscillator_visualization || {},
            temporalCoherence: this.hologramHints?.temporal_coherence || {},
            recommendedViews: this.hologramHints?.recommended_views || []
        };
        
        // Add dynamic enhancements based on current state
        if (this.coherenceHistory.length > 0) {
            const recentCoherence = this.coherenceHistory[this.coherenceHistory.length - 1].coherence;
            baseData.intensity = recentCoherence;
            baseData.glowEffect = recentCoherence > 0.7;
        }
        
        // Add phase evolution data for smooth animation
        if (this.phaseEvolution.length > 0) {
            const recentPhase = this.phaseEvolution[this.phaseEvolution.length - 1];
            baseData.animationHints = {
                rotationSpeed: recentPhase.velocity,
                phaseOffset: recentPhase.phase,
                smoothing: true
            };
        }
        
        // Add synchronization visualization hints
        if (this.oscillatorHistory.length > 0) {
            const recentSync = this.oscillatorHistory[this.oscillatorHistory.length - 1].synchronization;
            baseData.synchronizationVisual = {
                particleDensity: recentSync,
                connectionStrength: recentSync,
                pulseFrequency: 1 + recentSync * 2 // 1-3 Hz based on sync
            };
        }
        
        return baseData;
    }
    
    // New method for holographic state interpolation
    interpolateToFrame(targetFrame, alpha) {
        if (!targetFrame || alpha < 0 || alpha > 1) return this.toHologramData();
        
        const currentData = this.toHologramData();
        const targetData = targetFrame.toHologramData();
        
        // Interpolate position
        const interpolatedPosition = {
            x: currentData.position3D.x + (targetData.position3D.x - currentData.position3D.x) * alpha,
            y: currentData.position3D.y + (targetData.position3D.y - currentData.position3D.y) * alpha,
            z: currentData.position3D.z + (targetData.position3D.z - currentData.position3D.z) * alpha
        };
        
        // Interpolate color (in HSL space)
        const interpolatedColor = {
            h: this.interpolateHue(currentData.colorHSL.h, targetData.colorHSL.h, alpha),
            s: currentData.colorHSL.s + (targetData.colorHSL.s - currentData.colorHSL.s) * alpha,
            l: currentData.colorHSL.l + (targetData.colorHSL.l - currentData.colorHSL.l) * alpha
        };
        
        return {
            ...currentData,
            position3D: interpolatedPosition,
            colorHSL: interpolatedColor,
            interpolated: true,
            alpha: alpha
        };
    }
    
    interpolateHue(h1, h2, alpha) {
        // Handle circular interpolation for hue
        let delta = h2 - h1;
        if (delta > 180) delta -= 360;
        if (delta < -180) delta += 360;
        
        let result = h1 + delta * alpha;
        if (result < 0) result += 360;
        if (result > 360) result -= 360;
        
        return result;
    }
}

class PsiMemoryManager {
    constructor() {
        this.frames = [];
        this.maxFrames = 1000;
        this.currentFrame = null;
        this.sessionState = {
            totalCoherence: 0,
            averagePhase: 0,
            emotionalTrajectory: [],
            hologramicMoments: [],
            oscillatorStats: {
                averageSynchronization: 0,
                synchronizationHistory: []
            }
        };
        // New fields for enhanced functionality
        this.frameIndex = new Map(); // Quick lookup by timestamp
        this.emotionalClusters = [];
        this.koopmanBuffer = []; // For Koopman operator estimation
    }
    
    addFrame(psiState, hologramHints, emotionalResonance) {
        const frame = new PsiFrame(
            Date.now(),
            { ...psiState },
            { ...hologramHints },
            { ...emotionalResonance }
        );
        
        this.frames.push(frame);
        this.currentFrame = frame;
        this.frameIndex.set(frame.timestamp, frame);
        
        // Update Koopman buffer for predictive capabilities
        this.updateKoopmanBuffer(frame);
        
        // Maintain frame limit
        if (this.frames.length > this.maxFrames) {
            const removedFrame = this.frames.shift();
            this.frameIndex.delete(removedFrame.timestamp);
        }
        
        this.updateSessionState(frame);
        
        // Check for emotional clustering
        this.updateEmotionalClusters(frame);
        
        return frame;
    }
    
    updateSessionState(frame) {
        const coherence = frame.psiState.phase_coherence || 0;
        const phase = frame.psiState.psi_phase || 0;
        
        this.sessionState.totalCoherence += coherence;
        this.sessionState.averagePhase = this.calculateAveragePhase();
        
        // Track emotional trajectory
        if (frame.emotionalResonance) {
            this.sessionState.emotionalTrajectory.push({
                timestamp: frame.timestamp,
                emotion: frame.emotionalResonance,
                phase: phase,
                coherence: coherence
            });
            
            // Limit trajectory history
            if (this.sessionState.emotionalTrajectory.length > 500) {
                this.sessionState.emotionalTrajectory.shift();
            }
        }
        
        // Update oscillator statistics
        if (frame.psiState.oscillator_phases) {
            const sync = frame.calculateSynchronization(frame.psiState.oscillator_phases);
            this.sessionState.oscillatorStats.synchronizationHistory.push({
                timestamp: frame.timestamp,
                sync: sync
            });
            
            // Calculate running average
            const recentSync = this.sessionState.oscillatorStats.synchronizationHistory.slice(-50);
            this.sessionState.oscillatorStats.averageSynchronization = 
                recentSync.reduce((sum, s) => sum + s.sync, 0) / recentSync.length;
        }
        
        // Identify holographic moments (high coherence + strong emotion)
        if (coherence > 0.8 && this.hasStrongEmotion(frame.emotionalResonance)) {
            this.sessionState.hologramicMoments.push({
                timestamp: frame.timestamp,
                coherence: coherence,
                emotion: frame.emotionalResonance,
                hologramData: frame.toHologramData(),
                type: 'strong_emotion_coherence'
            });
            frame.markHolographicMoment('session_peak');
        }
    }
    
    updateKoopmanBuffer(frame) {
        // Store state vectors for Koopman operator estimation
        const stateVector = [
            frame.psiState.psi_phase || 0,
            frame.psiState.phase_coherence || 0,
            frame.emotionalResonance?.excitement || 0,
            frame.emotionalResonance?.calmness || 0,
            frame.emotionalResonance?.energy || 0,
            frame.emotionalResonance?.clarity || 0
        ];
        
        this.koopmanBuffer.push({
            timestamp: frame.timestamp,
            state: stateVector
        });
        
        // Keep buffer size limited
        if (this.koopmanBuffer.length > 100) {
            this.koopmanBuffer.shift();
        }
    }
    
    updateEmotionalClusters(frame) {
        if (!frame.emotionalResonance) return;
        
        // Simple clustering based on dominant emotion
        const emotions = frame.emotionalResonance;
        const dominantEmotion = Object.entries(emotions)
            .reduce((a, b) => emotions[a] > emotions[b] ? a : b)[0];
        
        let cluster = this.emotionalClusters.find(c => c.emotion === dominantEmotion);
        if (!cluster) {
            cluster = {
                emotion: dominantEmotion,
                frames: [],
                centroid: { ...emotions },
                count: 0
            };
            this.emotionalClusters.push(cluster);
        }
        
        cluster.frames.push(frame.timestamp);
        cluster.count++;
        
        // Update centroid (running average)
        Object.keys(emotions).forEach(key => {
            cluster.centroid[key] = (cluster.centroid[key] * (cluster.count - 1) + emotions[key]) / cluster.count;
        });
    }
    
    hasStrongEmotion(emotionalResonance) {
        if (!emotionalResonance) return false;
        
        const totalIntensity = 
            (emotionalResonance.excitement || 0) +
            (emotionalResonance.energy || 0) +
            (emotionalResonance.clarity || 0);
            
        return totalIntensity > 1.5;
    }
    
    calculateAveragePhase() {
        if (this.frames.length === 0) return 0;
        
        // Use circular mean for phase averaging
        let sumSin = 0, sumCos = 0;
        
        this.frames.forEach(frame => {
            const phase = frame.psiState.psi_phase || 0;
            sumSin += Math.sin(phase);
            sumCos += Math.cos(phase);
        });
        
        return Math.atan2(sumSin / this.frames.length, sumCos / this.frames.length);
    }
    
    getRecentFrames(count = 10) {
        return this.frames.slice(-count);
    }
    
    getHologramicMoments() {
        return this.sessionState.hologramicMoments;
    }
    
    // New method: Get frames by time range
    getFramesByTimeRange(startTime, endTime) {
        return this.frames.filter(frame => 
            frame.timestamp >= startTime && frame.timestamp <= endTime
        );
    }
    
    // New method: Get emotional cluster analysis
    getEmotionalClusters() {
        return this.emotionalClusters.map(cluster => ({
            emotion: cluster.emotion,
            count: cluster.count,
            centroid: { ...cluster.centroid },
            percentage: (cluster.count / this.frames.length) * 100
        }));
    }
    
    // New method: Predict next state using Koopman buffer
    predictNextState(steps = 1) {
        if (this.koopmanBuffer.length < 10) {
            return null; // Not enough data
        }
        
        // Simple linear prediction based on recent trends
        const recentStates = this.koopmanBuffer.slice(-10);
        const lastState = recentStates[recentStates.length - 1].state;
        const trends = [];
        
        // Calculate trends for each dimension
        for (let i = 0; i < lastState.length; i++) {
            const values = recentStates.map(s => s.state[i]);
            const x = Array.from({ length: values.length }, (_, i) => i);
            
            // Simple linear regression
            const n = values.length;
            const sumX = x.reduce((a, b) => a + b, 0);
            const sumY = values.reduce((a, b) => a + b, 0);
            const sumXY = x.reduce((sum, xi, i) => sum + xi * values[i], 0);
            const sumX2 = x.reduce((sum, xi) => sum + xi * xi, 0);
            
            const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
            trends.push(slope);
        }
        
        // Predict next state
        const predictedState = lastState.map((val, i) => val + trends[i] * steps);
        
        return {
            psi_phase: predictedState[0],
            phase_coherence: Math.max(0, Math.min(1, predictedState[1])),
            emotional_resonance: {
                excitement: Math.max(0, Math.min(1, predictedState[2])),
                calmness: Math.max(0, Math.min(1, predictedState[3])),
                energy: Math.max(0, Math.min(1, predictedState[4])),
                clarity: Math.max(0, Math.min(1, predictedState[5]))
            }
        };
    }
    
    // New method: Find similar emotional states
    findSimilarStates(targetEmotion, threshold = 0.2) {
        return this.frames.filter(frame => {
            if (!frame.emotionalResonance) return false;
            
            let distance = 0;
            Object.keys(targetEmotion).forEach(key => {
                const diff = (frame.emotionalResonance[key] || 0) - (targetEmotion[key] || 0);
                distance += diff * diff;
            });
            
            return Math.sqrt(distance) < threshold;
        });
    }
    
    exportHologramSession() {
        return {
            sessionState: this.sessionState,
            frames: this.frames.map(frame => frame.toHologramData()),
            emotionalClusters: this.getEmotionalClusters(),
            metadata: {
                totalFrames: this.frames.length,
                averageCoherence: this.sessionState.totalCoherence / this.frames.length,
                averagePhase: this.sessionState.averagePhase,
                averageSynchronization: this.sessionState.oscillatorStats.averageSynchronization,
                hologramicMoments: this.sessionState.hologramicMoments.length,
                sessionDuration: this.frames.length > 0 
                    ? this.frames[this.frames.length - 1].timestamp - this.frames[0].timestamp 
                    : 0
            }
        };
    }
}

// Global memory manager instance
const psiMemory = new PsiMemoryManager();

export function generateInitialHologramHints(psiState) {
    return {
        position_3d: { x: 0, y: 0, z: 0 },
        color_hsl: { h: 240, s: 70, l: 50 },
        volumetric_density: new Array(8).fill(0).map(() => 
            new Array(8).fill(0).map(() => 
                new Array(8).fill(0.1)
            )
        ),
        oscillator_visualization: {
            phases: psiState.oscillator_phases || [],
            frequencies: [80, 120, 200, 350, 600, 1000, 1800, 3200],
            coupling_strength: 0.1
        },
        temporal_coherence: {
            beat_frequency: 0,
            phase_stability: psiState.phase_coherence || 0.5,
            emotional_flow: psiState.emotional_resonance || {}
        },
        semantic_anchors: [],
        recommended_views: [
            { name: 'primary', azimuth: 0, elevation: 0, distance: 1.0, weight: 1.0 }
        ]
    };
}

export function updateHologramHints(psiState, existingHints) {
    const frame = psiMemory.addFrame(psiState, existingHints, psiState.emotional_resonance);
    
    // Enhanced hints generation with predictive elements
    const prediction = psiMemory.predictNextState();
    const hints = frame.toHologramData();
    
    if (prediction) {
        hints.predictive = {
            nextPhase: prediction.psi_phase,
            nextCoherence: prediction.phase_coherence,
            emotionalTrend: prediction.emotional_resonance
        };
    }
    
    return hints;
}

// New export: Real-time interpolation for smooth animations
export function interpolateHologramStates(alpha) {
    const recentFrames = psiMemory.getRecentFrames(2);
    if (recentFrames.length < 2) {
        return recentFrames[0]?.toHologramData() || generateInitialHologramHints({});
    }
    
    return recentFrames[0].interpolateToFrame(recentFrames[1], alpha);
}

// New export: Get holographic moments for replay
export function getHolographicHighlights(limit = 10) {
    const moments = psiMemory.getHologramicMoments();
    return moments
        .sort((a, b) => b.coherence - a.coherence)
        .slice(0, limit);
}

export { PsiFrame, PsiMemoryManager, psiMemory };