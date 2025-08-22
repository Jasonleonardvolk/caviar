// conceptHologramRenderer.js - Renders Concept Mesh as holographic visualization
// Concepts become glowing entities, relationships become energy flows!

import { SpectralHologramEngine, ToriHolographicRenderer } from './holographicEngine';

export class ConceptHologramRenderer {
    constructor() {
        this.wsConnection = null;
        this.concepts = new Map(); // concept_id -> concept data
        this.relations = new Map(); // relation_id -> relation data
        this.conceptMeshes = new Map(); // concept_id -> THREE.js mesh
        this.relationFlows = new Map(); // relation_id -> particle system
        
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.hologramEngine = null;
        
        this.selectedConcept = null;
        this.hoveredConcept = null;
        
        // Animation state
        this.animationTime = 0;
        this.particleSystems = [];
        
        console.log('🧠 Initializing Concept Hologram Renderer');
    }
    
    async initialize(canvas, hologramEngine) {
        this.hologramEngine = hologramEngine;
        
        // Connect to Concept Mesh WebSocket
        await this.connectToConceptMesh();
        
        // Initialize 3D scene (if using Three.js integration)
        // this.initializeScene(canvas);
        
        console.log('✅ Concept Hologram Renderer initialized');
    }
    
    async connectToConceptMesh() {
        const wsUrl = 'ws://localhost:8766/concepts';
        
        return new Promise((resolve, reject) => {
            this.wsConnection = new WebSocket(wsUrl);
            
            this.wsConnection.onopen = () => {
                console.log('✅ Connected to Concept Mesh');
                resolve();
            };
            
            this.wsConnection.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleConceptMeshMessage(data);
            };
            
            this.wsConnection.onerror = (error) => {
                console.error('WebSocket error:', error);
                reject(error);
            };
            
            this.wsConnection.onclose = () => {
                console.log('Disconnected from Concept Mesh');
                // Attempt reconnection
                setTimeout(() => this.connectToConceptMesh(), 5000);
            };
        });
    }
    
    handleConceptMeshMessage(data) {
        switch (data.type) {
            case 'holographic_scene':
                this.loadScene(data.scene);
                break;
                
            case 'mesh_update':
                this.handleMeshUpdate(data);
                break;
                
            case 'position_update':
                this.updateConceptPosition(data.concept_id, data.position);
                break;
                
            case 'concept_data':
                this.highlightConcept(data.concept);
                break;
                
            case 'similar_concepts':
                this.showSimilarConcepts(data.similar);
                break;
        }
    }
    
    loadScene(sceneData) {
        console.log(`Loading holographic scene with ${sceneData.total_concepts} concepts`);
        
        // Clear existing data
        this.concepts.clear();
        this.relations.clear();
        
        // Load concepts
        sceneData.concepts.forEach(concept => {
            this.concepts.set(concept.id, concept);
            this.createConceptHologram(concept);
        });
        
        // Load relations
        sceneData.relations.forEach(relation => {
            this.relations.set(relation.id, relation);
            this.createRelationFlow(relation);
        });
        
        // Update camera to fit scene
        if (sceneData.bounds) {
            this.fitCameraToScene(sceneData.bounds);
        }
        
        console.log('✅ Scene loaded');
    }
    
    createConceptHologram(conceptData) {
        // Convert concept to holographic parameters
        const hologramParams = {
            id: conceptData.id,
            position: conceptData.position,
            
            // Psi-state from concept
            psi_phase: conceptData.hologram.psi_phase,
            phase_coherence: conceptData.hologram.phase_coherence,
            oscillator_phases: conceptData.hologram.oscillator_phases,
            oscillator_frequencies: conceptData.hologram.oscillator_frequencies,
            dominant_frequency: conceptData.hologram.dominant_frequency,
            
            // Visual properties
            color: conceptData.hologram.color,
            size: conceptData.hologram.size,
            intensity: conceptData.hologram.intensity,
            rotation_speed: conceptData.hologram.rotation_speed,
            
            // Metadata for interaction
            name: conceptData.name,
            description: conceptData.description,
            category: conceptData.category
        };
        
        // Update hologram engine with concept data
        if (this.hologramEngine) {
            // Create a holographic object for this concept
            const hologramId = this.hologramEngine.addHolographicObject({
                id: conceptData.id,
                type: 'concept',
                resonance: conceptData.hologram.phase_coherence,
                position: conceptData.position,
                metadata: hologramParams
            });
            
            // Update oscillator state for this concept
            this.updateConceptOscillators(conceptData);
        }
        
        // Store for later updates
        this.conceptMeshes.set(conceptData.id, hologramParams);
    }
    
    updateConceptOscillators(conceptData) {
        if (!this.hologramEngine) return;
        
        // Create a localized psi-state for this concept
        const conceptPsiState = {
            psi_phase: conceptData.hologram.psi_phase,
            phase_coherence: conceptData.hologram.phase_coherence,
            oscillator_phases: conceptData.hologram.oscillator_phases,
            oscillator_frequencies: conceptData.hologram.oscillator_frequencies,
            coupling_strength: 0.5,
            dominant_frequency: conceptData.hologram.dominant_frequency
        };
        
        // Blend with global state based on concept importance
        const blendFactor = conceptData.hologram.intensity;
        this.blendPsiStates(this.hologramEngine.psiState, conceptPsiState, blendFactor);
    }
    
    blendPsiStates(globalState, conceptState, factor) {
        // Blend oscillator phases
        for (let i = 0; i < globalState.oscillator_phases.length; i++) {
            globalState.oscillator_phases[i] = 
                globalState.oscillator_phases[i] * (1 - factor) +
                conceptState.oscillator_phases[i] * factor;
        }
        
        // Blend frequencies
        for (let i = 0; i < globalState.oscillator_frequencies.length; i++) {
            globalState.oscillator_frequencies[i] = 
                globalState.oscillator_frequencies[i] * (1 - factor) +
                conceptState.oscillator_frequencies[i] * factor;
        }
        
        // Update dominant frequency
        globalState.dominant_frequency = 
            globalState.dominant_frequency * (1 - factor) +
            conceptState.dominant_frequency * factor;
    }
    
    createRelationFlow(relationData) {
        // Create energy flow between concepts
        const flowParams = {
            id: relationData.id,
            source_id: relationData.source_id,
            target_id: relationData.target_id,
            
            // Flow properties
            color: relationData.hologram.color,
            width: relationData.hologram.width,
            energy_flow: relationData.hologram.energy_flow,
            particle_count: relationData.hologram.particle_count,
            pulse_speed: relationData.hologram.pulse_speed,
            
            // Wave properties
            wave_frequency: relationData.hologram.wave_frequency,
            wave_amplitude: relationData.hologram.wave_amplitude,
            phase_offset: relationData.hologram.phase_offset
        };
        
        // Create particle system for this relation
        this.createEnergyFlow(flowParams);
        
        // Store for updates
        this.relationFlows.set(relationData.id, flowParams);
    }
    
    createEnergyFlow(flowParams) {
        // Get source and target positions
        const source = this.concepts.get(flowParams.source_id);
        const target = this.concepts.get(flowParams.target_id);
        
        if (!source || !target) return;
        
        const particles = [];
        const particleCount = flowParams.particle_count;
        
        for (let i = 0; i < particleCount; i++) {
            particles.push({
                position: [...source.position],
                target: [...target.position],
                progress: i / particleCount, // Stagger particles
                speed: flowParams.pulse_speed,
                color: flowParams.color,
                size: flowParams.width * 2,
                energy: flowParams.energy_flow
            });
        }
        
        this.particleSystems.push({
            id: flowParams.id,
            particles: particles,
            flowParams: flowParams
        });
    }
    
    handleMeshUpdate(update) {
        const diff = update.diff;
        
        switch (diff.diff_type) {
            case 'add':
                if (update.concept) {
                    this.concepts.set(update.concept.id, update.concept);
                    this.createConceptHologram(update.concept);
                }
                break;
                
            case 'remove':
                diff.concepts.forEach(conceptId => {
                    this.removeConceptHologram(conceptId);
                });
                break;
                
            case 'relate':
                if (update.relation) {
                    this.relations.set(update.relation.id, update.relation);
                    this.createRelationFlow(update.relation);
                }
                break;
        }
    }
    
    removeConceptHologram(conceptId) {
        this.concepts.delete(conceptId);
        this.conceptMeshes.delete(conceptId);
        
        // Remove associated relations
        this.relations.forEach((relation, relationId) => {
            if (relation.source_id === conceptId || relation.target_id === conceptId) {
                this.relations.delete(relationId);
                this.relationFlows.delete(relationId);
                
                // Remove particle system
                this.particleSystems = this.particleSystems.filter(
                    ps => ps.id !== relationId
                );
            }
        });
    }
    
    updateConceptPosition(conceptId, position) {
        const concept = this.concepts.get(conceptId);
        if (concept) {
            concept.position = position;
            
            // Update visual representation
            const mesh = this.conceptMeshes.get(conceptId);
            if (mesh) {
                mesh.position = position;
            }
        }
    }
    
    update(deltaTime) {
        this.animationTime += deltaTime;
        
        // Update concept rotations and pulsing
        this.concepts.forEach((concept, conceptId) => {
            const mesh = this.conceptMeshes.get(conceptId);
            if (mesh) {
                // Rotate based on importance
                mesh.rotation_angle = (mesh.rotation_angle || 0) + 
                    deltaTime * concept.hologram.rotation_speed;
                
                // Pulse intensity
                const pulseFactor = 1 + 0.2 * Math.sin(
                    this.animationTime * 2 + concept.hologram.psi_phase
                );
                mesh.current_intensity = mesh.intensity * pulseFactor;
            }
        });
        
        // Update particle flows
        this.updateParticleFlows(deltaTime);
        
        // Update hologram engine if concepts are active
        if (this.concepts.size > 0 && this.hologramEngine) {
            this.updateGlobalHologramState();
        }
    }
    
    updateParticleFlows(deltaTime) {
        this.particleSystems.forEach(system => {
            system.particles.forEach(particle => {
                // Move particle along path
                particle.progress += deltaTime * particle.speed;
                
                if (particle.progress >= 1) {
                    // Reset to start
                    particle.progress = 0;
                    particle.position = [...this.concepts.get(
                        system.flowParams.source_id
                    ).position];
                }
                
                // Interpolate position
                const t = particle.progress;
                const source = particle.position;
                const target = particle.target;
                
                // Add wave motion
                const waveOffset = Math.sin(
                    t * Math.PI * system.flowParams.wave_frequency +
                    system.flowParams.phase_offset
                ) * system.flowParams.wave_amplitude;
                
                // Update position with wave
                for (let i = 0; i < 3; i++) {
                    particle.position[i] = source[i] * (1 - t) + target[i] * t;
                }
                particle.position[1] += waveOffset; // Add wave to Y axis
            });
        });
    }
    
    updateGlobalHologramState() {
        // Calculate weighted average of all concept states
        let totalWeight = 0;
        const avgState = {
            phase_coherence: 0,
            dominant_frequency: 0
        };
        
        this.concepts.forEach(concept => {
            const weight = concept.hologram.intensity;
            avgState.phase_coherence += concept.hologram.phase_coherence * weight;
            avgState.dominant_frequency += concept.hologram.dominant_frequency * weight;
            totalWeight += weight;
        });
        
        if (totalWeight > 0) {
            avgState.phase_coherence /= totalWeight;
            avgState.dominant_frequency /= totalWeight;
            
            // Smoothly update global state
            const blendFactor = 0.1; // Smooth transitions
            this.hologramEngine.psiState.phase_coherence = 
                this.hologramEngine.psiState.phase_coherence * (1 - blendFactor) +
                avgState.phase_coherence * blendFactor;
                
            this.hologramEngine.psiState.dominant_frequency = 
                this.hologramEngine.psiState.dominant_frequency * (1 - blendFactor) +
                avgState.dominant_frequency * blendFactor;
        }
    }
    
    // Interaction methods
    selectConcept(conceptId) {
        this.selectedConcept = conceptId;
        
        // Request detailed data
        this.sendMessage({
            type: 'get_concept',
            concept_id: conceptId
        });
        
        // Highlight visually
        this.highlightConcept(this.concepts.get(conceptId));
    }
    
    highlightConcept(concept) {
        if (!concept) return;
        
        // Boost intensity for selected concept
        const mesh = this.conceptMeshes.get(concept.id);
        if (mesh) {
            mesh.highlighted = true;
            mesh.intensity = concept.hologram.intensity * 2;
        }
        
        // Dim other concepts
        this.concepts.forEach((otherConcept, otherId) => {
            if (otherId !== concept.id) {
                const otherMesh = this.conceptMeshes.get(otherId);
                if (otherMesh && !otherMesh.highlighted) {
                    otherMesh.intensity = otherConcept.hologram.intensity * 0.5;
                }
            }
        });
    }
    
    clearHighlights() {
        this.conceptMeshes.forEach((mesh, conceptId) => {
            const concept = this.concepts.get(conceptId);
            if (concept) {
                mesh.highlighted = false;
                mesh.intensity = concept.hologram.intensity;
            }
        });
    }
    
    findSimilarConcepts(conceptId, threshold = 0.7) {
        this.sendMessage({
            type: 'find_similar',
            concept_id: conceptId,
            threshold: threshold
        });
    }
    
    showSimilarConcepts(similarConcepts) {
        // Create temporary connections to similar concepts
        similarConcepts.forEach((item, index) => {
            const tempRelation = {
                id: `similar_${index}`,
                source_id: this.selectedConcept,
                target_id: item.concept.id,
                hologram: {
                    color: [0.7, 0.4, 0.7], // Magenta for similarity
                    width: item.similarity * 0.3,
                    energy_flow: item.similarity,
                    particle_count: Math.floor(item.similarity * 30),
                    pulse_speed: 2 - item.similarity,
                    wave_frequency: 2,
                    wave_amplitude: 0.3,
                    phase_offset: 0
                }
            };
            
            this.createRelationFlow(tempRelation);
        });
    }
    
    addConceptFromText(text) {
        this.sendMessage({
            type: 'extract_concepts',
            text: text,
            min_importance: 0.3
        });
    }
    
    fitCameraToScene(bounds) {
        // Calculate scene center and size
        const center = [
            (bounds.min[0] + bounds.max[0]) / 2,
            (bounds.min[1] + bounds.max[1]) / 2,
            (bounds.min[2] + bounds.max[2]) / 2
        ];
        
        const size = Math.max(
            bounds.max[0] - bounds.min[0],
            bounds.max[1] - bounds.min[1],
            bounds.max[2] - bounds.min[2]
        );
        
        // Update camera position
        if (this.camera) {
            this.camera.position.set(
                center[0],
                center[1] + size * 0.5,
                center[2] + size * 1.5
            );
            this.camera.lookAt(center[0], center[1], center[2]);
        }
    }
    
    sendMessage(message) {
        if (this.wsConnection && this.wsConnection.readyState === WebSocket.OPEN) {
            this.wsConnection.send(JSON.stringify(message));
        }
    }
    
    // Get current state for debugging
    getState() {
        return {
            conceptCount: this.concepts.size,
            relationCount: this.relations.size,
            particleSystemCount: this.particleSystems.length,
            selectedConcept: this.selectedConcept,
            wsConnected: this.wsConnection?.readyState === WebSocket.OPEN
        };
    }
    
    destroy() {
        if (this.wsConnection) {
            this.wsConnection.close();
        }
        
        this.concepts.clear();
        this.relations.clear();
        this.conceptMeshes.clear();
        this.relationFlows.clear();
        this.particleSystems = [];
    }
}

export default ConceptHologramRenderer;
