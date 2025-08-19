/**
 * Real Ghost Engine v2
 * =====================
 * Advanced holographic ghost engine with ZeroMQ integration,
 * mesh context SSE, and persona/intent controls.
 */

import * as THREE from 'three';
import { io } from 'socket.io-client';

// ============================================================================
// CONFIGURATION
// ============================================================================

const CONFIG = {
    ZMQ_ENDPOINT: 'ws://localhost:5555',
    SSE_ENDPOINT: 'http://localhost:8001/api/sse/mesh',
    API_ENDPOINT: 'http://localhost:8001/api/saigon',
    HOLOGRAM_FPS: 60,
    PARTICLE_COUNT: 10000,
    GHOST_OPACITY: 0.7,
    MORPH_DURATION: 2000,
    AUDIO_BUFFER_SIZE: 2048
};

// ============================================================================
// GHOST ENGINE CORE
// ============================================================================

export class RealGhostEngine {
    constructor(container, options = {}) {
        this.container = container;
        this.options = { ...CONFIG, ...options };
        
        // Three.js components
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.particles = null;
        this.ghostMesh = null;
        
        // State management
        this.state = {
            user_id: null,
            session_id: null,
            mesh_context: null,
            lattice_state: null,
            audio_phase: 0,
            coherence: 1.0,
            morphing: false,
            connected: false
        };
        
        // Connections
        this.socket = null;
        this.sseSource = null;
        this.audioContext = null;
        
        // Animation
        this.animationId = null;
        this.morphTween = null;
        
        // Callbacks
        this.callbacks = {
            onMeshUpdate: null,
            onLatticeUpdate: null,
            onAudioSync: null,
            onStateChange: null
        };
        
        this.init();
    }
    
    // ========================================================================
    // INITIALIZATION
    // ========================================================================
    
    init() {
        this.initThreeJS();
        this.initParticles();
        this.initGhost();
        this.initConnections();
        this.initAudio();
        this.startAnimation();
        
        console.log('[Ghost Engine v2] Initialized');
    }
    
    initThreeJS() {
        // Scene setup
        this.scene = new THREE.Scene();
        this.scene.fog = new THREE.FogExp2(0x000000, 0.0008);
        
        // Camera setup
        const aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera = new THREE.PerspectiveCamera(75, aspect, 0.1, 10000);
        this.camera.position.set(0, 0, 500);
        
        // Renderer setup
        this.renderer = new THREE.WebGLRenderer({ 
            antialias: true, 
            alpha: true 
        });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.container.appendChild(this.renderer.domElement);
        
        // Lights
        const ambientLight = new THREE.AmbientLight(0x404040);
        this.scene.add(ambientLight);
        
        const pointLight = new THREE.PointLight(0x00ffff, 1, 1000);
        pointLight.position.set(0, 100, 100);
        this.scene.add(pointLight);
        
        // Handle resize
        window.addEventListener('resize', () => this.handleResize());
    }
    
    initParticles() {
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(this.options.PARTICLE_COUNT * 3);
        const colors = new Float32Array(this.options.PARTICLE_COUNT * 3);
        
        for (let i = 0; i < this.options.PARTICLE_COUNT; i++) {
            const i3 = i * 3;
            
            // Random positions in sphere
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.acos(2 * Math.random() - 1);
            const r = 200 + Math.random() * 100;
            
            positions[i3] = r * Math.sin(phi) * Math.cos(theta);
            positions[i3 + 1] = r * Math.sin(phi) * Math.sin(theta);
            positions[i3 + 2] = r * Math.cos(phi);
            
            // Cyan-purple gradient
            colors[i3] = 0.0;
            colors[i3 + 1] = Math.random() * 0.5 + 0.5;
            colors[i3 + 2] = Math.random() * 0.5 + 0.5;
        }
        
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        
        const material = new THREE.PointsMaterial({
            size: 2,
            vertexColors: true,
            transparent: true,
            opacity: 0.6,
            blending: THREE.AdditiveBlending
        });
        
        this.particles = new THREE.Points(geometry, material);
        this.scene.add(this.particles);
    }
    
    initGhost() {
        // Create ghost mesh (icosahedron for now)
        const geometry = new THREE.IcosahedronGeometry(100, 2);
        const material = new THREE.MeshPhongMaterial({
            color: 0x00ffff,
            transparent: true,
            opacity: this.options.GHOST_OPACITY,
            wireframe: true,
            emissive: 0x00ffff,
            emissiveIntensity: 0.2
        });
        
        this.ghostMesh = new THREE.Mesh(geometry, material);
        this.scene.add(this.ghostMesh);
        
        // Add glow effect
        const glowGeometry = new THREE.IcosahedronGeometry(105, 2);
        const glowMaterial = new THREE.MeshBasicMaterial({
            color: 0x00ffff,
            transparent: true,
            opacity: 0.2,
            side: THREE.BackSide
        });
        
        const glowMesh = new THREE.Mesh(glowGeometry, glowMaterial);
        this.ghostMesh.add(glowMesh);
    }
    
    // ========================================================================
    // CONNECTIONS
    // ========================================================================
    
    initConnections() {
        // WebSocket connection for real-time updates
        this.socket = io(this.options.ZMQ_ENDPOINT, {
            transports: ['websocket'],
            reconnection: true
        });
        
        this.socket.on('connect', () => {
            console.log('[Ghost Engine] Connected to ZeroMQ bridge');
            this.state.connected = true;
            this.onStateChange();
        });
        
        this.socket.on('lattice_update', (data) => {
            this.handleLatticeUpdate(data);
        });
        
        this.socket.on('mesh_update', (data) => {
            this.handleMeshUpdate(data);
        });
        
        this.socket.on('audio_sync', (data) => {
            this.handleAudioSync(data);
        });
        
        // SSE for mesh context streaming
        this.initSSE();
    }
    
    initSSE() {
        if (!this.state.user_id) return;
        
        const url = `${this.options.SSE_ENDPOINT}?user_id=${this.state.user_id}`;
        this.sseSource = new EventSource(url);
        
        this.sseSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleMeshUpdate(data);
        };
        
        this.sseSource.onerror = (error) => {
            console.error('[Ghost Engine] SSE error:', error);
        };
    }
    
    initAudio() {
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        
        // Create analyser for frequency data
        this.analyser = this.audioContext.createAnalyser();
        this.analyser.fftSize = this.options.AUDIO_BUFFER_SIZE;
        
        // Create buffer for frequency data
        this.frequencyData = new Uint8Array(this.analyser.frequencyBinCount);
        
        // Request microphone access (optional)
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia({ audio: true })
                .then((stream) => {
                    const source = this.audioContext.createMediaStreamSource(stream);
                    source.connect(this.analyser);
                    console.log('[Ghost Engine] Audio input connected');
                })
                .catch((err) => {
                    console.warn('[Ghost Engine] No audio input:', err);
                });
        }
    }
    
    // ========================================================================
    // UPDATE HANDLERS
    // ========================================================================
    
    handleLatticeUpdate(data) {
        this.state.lattice_state = data;
        
        // Update ghost mesh based on lattice
        if (data.vertices && this.ghostMesh) {
            this.morphGhostToLattice(data.vertices);
        }
        
        // Update coherence
        this.state.coherence = data.coherence || 1.0;
        
        // Callback
        if (this.callbacks.onLatticeUpdate) {
            this.callbacks.onLatticeUpdate(data);
        }
    }
    
    handleMeshUpdate(data) {
        this.state.mesh_context = data;
        
        // Update particle colors based on mesh
        if (data.nodes) {
            this.updateParticlesFromMesh(data.nodes);
        }
        
        // Callback
        if (this.callbacks.onMeshUpdate) {
            this.callbacks.onMeshUpdate(data);
        }
    }
    
    handleAudioSync(data) {
        this.state.audio_phase = data.phase || 0;
        
        // Modulate ghost opacity with audio
        if (this.ghostMesh) {
            this.ghostMesh.material.opacity = 
                this.options.GHOST_OPACITY * (0.5 + 0.5 * Math.sin(this.state.audio_phase));
        }
        
        // Callback
        if (this.callbacks.onAudioSync) {
            this.callbacks.onAudioSync(data);
        }
    }
    
    // ========================================================================
    // MORPHING
    // ========================================================================
    
    morphGhostToLattice(vertices) {
        if (!this.ghostMesh || this.state.morphing) return;
        
        this.state.morphing = true;
        
        // Create new geometry from lattice vertices
        const newGeometry = new THREE.BufferGeometry();
        const positions = new Float32Array(vertices.flat());
        newGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        
        // Animate morph
        const startPositions = this.ghostMesh.geometry.attributes.position.array;
        const endPositions = positions;
        
        let progress = 0;
        const animate = () => {
            progress += 0.02;
            
            if (progress >= 1) {
                this.ghostMesh.geometry = newGeometry;
                this.state.morphing = false;
                return;
            }
            
            // Interpolate positions
            const currentPositions = this.ghostMesh.geometry.attributes.position.array;
            for (let i = 0; i < currentPositions.length; i++) {
                currentPositions[i] = startPositions[i] + 
                    (endPositions[i] - startPositions[i]) * this.easeInOutCubic(progress);
            }
            
            this.ghostMesh.geometry.attributes.position.needsUpdate = true;
            requestAnimationFrame(animate);
        };
        
        animate();
    }
    
    updateParticlesFromMesh(nodes) {
        if (!this.particles) return;
        
        const colors = this.particles.geometry.attributes.color.array;
        
        // Map node confidence to particle colors
        nodes.forEach((node, index) => {
            if (index < this.options.PARTICLE_COUNT) {
                const i3 = index * 3;
                const confidence = node.confidence || 0.5;
                
                // High confidence = bright cyan
                // Low confidence = dim purple
                colors[i3] = 1 - confidence;  // R
                colors[i3 + 1] = confidence;  // G
                colors[i3 + 2] = 1;           // B
            }
        });
        
        this.particles.geometry.attributes.color.needsUpdate = true;
    }
    
    // ========================================================================
    // ANIMATION
    // ========================================================================
    
    startAnimation() {
        const animate = () => {
            this.animationId = requestAnimationFrame(animate);
            
            // Update audio
            if (this.analyser) {
                this.analyser.getByteFrequencyData(this.frequencyData);
                this.updateAudioVisualization();
            }
            
            // Rotate particles
            if (this.particles) {
                this.particles.rotation.y += 0.001;
                this.particles.rotation.x += 0.0005;
            }
            
            // Pulse ghost with coherence
            if (this.ghostMesh) {
                const scale = 1 + 0.1 * Math.sin(Date.now() * 0.001) * this.state.coherence;
                this.ghostMesh.scale.setScalar(scale);
                this.ghostMesh.rotation.y += 0.005;
            }
            
            // Render
            this.renderer.render(this.scene, this.camera);
        };
        
        animate();
    }
    
    updateAudioVisualization() {
        // Calculate average frequency
        const avgFreq = this.frequencyData.reduce((a, b) => a + b, 0) / this.frequencyData.length;
        
        // Modulate particle size with audio
        if (this.particles) {
            this.particles.material.size = 2 + (avgFreq / 128) * 3;
        }
        
        // Modulate ghost emissive intensity
        if (this.ghostMesh) {
            this.ghostMesh.material.emissiveIntensity = 0.2 + (avgFreq / 256) * 0.5;
        }
    }
    
    // ========================================================================
    // API METHODS
    // ========================================================================
    
    async setUser(user_id, session_id = null) {
        this.state.user_id = user_id;
        this.state.session_id = session_id;
        
        // Reinitialize SSE with new user
        if (this.sseSource) {
            this.sseSource.close();
        }
        this.initSSE();
        
        // Fetch initial mesh context
        await this.fetchMeshContext();
        
        console.log(`[Ghost Engine] User set: ${user_id}`);
    }
    
    async fetchMeshContext() {
        if (!this.state.user_id) return;
        
        try {
            const response = await fetch(
                `${this.options.API_ENDPOINT}/mesh/${this.state.user_id}`
            );
            const data = await response.json();
            this.handleMeshUpdate(data);
        } catch (error) {
            console.error('[Ghost Engine] Failed to fetch mesh:', error);
        }
    }
    
    async morphTo(target, duration = 2000) {
        // Send morph request to backend
        try {
            const response = await fetch(`${this.options.API_ENDPOINT}/morph`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    user_id: this.state.user_id,
                    target: target,
                    duration_ms: duration,
                    audio_sync: true
                })
            });
            
            const result = await response.json();
            console.log('[Ghost Engine] Morph initiated:', result);
        } catch (error) {
            console.error('[Ghost Engine] Morph request failed:', error);
        }
    }
    
    // ========================================================================
    // UTILITIES
    // ========================================================================
    
    easeInOutCubic(t) {
        return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    }
    
    handleResize() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        
        this.renderer.setSize(width, height);
    }
    
    onStateChange() {
        if (this.callbacks.onStateChange) {
            this.callbacks.onStateChange(this.state);
        }
    }
    
    // ========================================================================
    // LIFECYCLE
    // ========================================================================
    
    destroy() {
        // Stop animation
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        
        // Close connections
        if (this.socket) {
            this.socket.disconnect();
        }
        
        if (this.sseSource) {
            this.sseSource.close();
        }
        
        if (this.audioContext) {
            this.audioContext.close();
        }
        
        // Clean up Three.js
        this.renderer.dispose();
        this.scene.clear();
        
        // Remove from DOM
        if (this.renderer.domElement.parentNode) {
            this.renderer.domElement.parentNode.removeChild(this.renderer.domElement);
        }
        
        console.log('[Ghost Engine] Destroyed');
    }
}

// ============================================================================
// EXPORTS
// ============================================================================

export default RealGhostEngine;

// Factory function
export function createGhostEngine(container, options) {
    return new RealGhostEngine(container, options);
}
