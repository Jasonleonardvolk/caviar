/**
 * Concept Hologram Renderer
 * ==========================
 * Multi-view holographic renderer with dynamic mesh/adapter feedback.
 */

import * as THREE from 'three';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { ShaderPass } from 'three/examples/jsm/postprocessing/ShaderPass.js';

// ============================================================================
// CONFIGURATION
// ============================================================================

const CONFIG = {
    VIEWS: 45,  // Looking Glass display views
    QUILT_ROWS: 9,
    QUILT_COLS: 5,
    VIEW_CONE: 40,  // Degrees
    DEPTH_SCALE: 1.0,
    HOLOGRAM_WIDTH: 2560,
    HOLOGRAM_HEIGHT: 1600,
    BLOOM_STRENGTH: 1.5,
    BLOOM_RADIUS: 0.4,
    BLOOM_THRESHOLD: 0.85
};

// ============================================================================
// HOLOGRAM SHADERS
// ============================================================================

const HologramVertexShader = `
    varying vec2 vUv;
    varying vec3 vPosition;
    varying vec3 vNormal;
    
    uniform float time;
    uniform float morphAmount;
    
    void main() {
        vUv = uv;
        vPosition = position;
        vNormal = normal;
        
        // Add holographic distortion
        vec3 pos = position;
        pos.y += sin(position.x * 10.0 + time) * 0.1 * morphAmount;
        pos.z += cos(position.y * 10.0 + time) * 0.1 * morphAmount;
        
        gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
    }
`;

const HologramFragmentShader = `
    varying vec2 vUv;
    varying vec3 vPosition;
    varying vec3 vNormal;
    
    uniform float time;
    uniform float coherence;
    uniform vec3 baseColor;
    uniform sampler2D meshTexture;
    
    void main() {
        // Holographic scanlines
        float scanline = sin(vUv.y * 300.0 + time * 10.0) * 0.04;
        
        // Fresnel effect
        vec3 viewDirection = normalize(cameraPosition - vPosition);
        float fresnel = pow(1.0 - dot(viewDirection, vNormal), 2.0);
        
        // Base color with coherence modulation
        vec3 color = baseColor * (0.5 + 0.5 * coherence);
        
        // Add mesh texture if available
        vec4 meshColor = texture2D(meshTexture, vUv);
        color = mix(color, meshColor.rgb, meshColor.a * 0.5);
        
        // Holographic effect
        color += vec3(0.0, 1.0, 1.0) * fresnel * coherence;
        color += scanline;
        
        // Glitch effect
        float glitch = step(0.98, sin(time * 20.0)) * step(0.9, sin(vUv.y * 20.0 + time * 10.0));
        color = mix(color, vec3(1.0, 0.0, 0.0), glitch);
        
        gl_FragColor = vec4(color, 0.8 + fresnel * 0.2);
    }
`;

// ============================================================================
// HOLOGRAM RENDERER
// ============================================================================

export class ConceptHologramRenderer {
    constructor(container, options = {}) {
        this.container = container;
        this.options = { ...CONFIG, ...options };
        
        // Three.js components
        this.scene = null;
        this.cameras = [];
        this.renderer = null;
        this.composer = null;
        
        // Hologram components
        this.hologramMesh = null;
        this.conceptNodes = [];
        this.connectionLines = null;
        this.quiltRenderTarget = null;
        
        // State
        this.state = {
            mesh_context: null,
            adapter_feedback: null,
            coherence: 1.0,
            morphing: false,
            viewIndex: 0,
            time: 0
        };
        
        // Animation
        this.animationId = null;
        
        this.init();
    }
    
    // ========================================================================
    // INITIALIZATION
    // ========================================================================
    
    init() {
        this.initScene();
        this.initCameras();
        this.initRenderer();
        this.initPostProcessing();
        this.initHologram();
        this.initQuiltTarget();
        this.startAnimation();
        
        console.log('[Hologram Renderer] Initialized');
    }
    
    initScene() {
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x000000);
        
        // Ambient light
        const ambient = new THREE.AmbientLight(0x404040, 0.5);
        this.scene.add(ambient);
        
        // Point lights for holographic effect
        const colors = [0x00ffff, 0xff00ff, 0xffff00];
        colors.forEach((color, i) => {
            const light = new THREE.PointLight(color, 0.5, 500);
            light.position.set(
                Math.cos(i * Math.PI * 2 / 3) * 200,
                0,
                Math.sin(i * Math.PI * 2 / 3) * 200
            );
            this.scene.add(light);
        });
    }
    
    initCameras() {
        const aspect = this.options.HOLOGRAM_WIDTH / this.options.HOLOGRAM_HEIGHT;
        const viewCone = this.options.VIEW_CONE;
        const numViews = this.options.VIEWS;
        
        for (let i = 0; i < numViews; i++) {
            const camera = new THREE.PerspectiveCamera(45, aspect, 0.1, 10000);
            
            // Calculate camera position for each view
            const angle = (i / (numViews - 1) - 0.5) * viewCone * Math.PI / 180;
            const radius = 500;
            
            camera.position.set(
                Math.sin(angle) * radius,
                0,
                Math.cos(angle) * radius
            );
            camera.lookAt(0, 0, 0);
            
            this.cameras.push(camera);
        }
    }
    
    initRenderer() {
        this.renderer = new THREE.WebGLRenderer({
            antialias: true,
            alpha: true,
            preserveDrawingBuffer: true
        });
        
        this.renderer.setSize(
            this.options.HOLOGRAM_WIDTH,
            this.options.HOLOGRAM_HEIGHT
        );
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1;
        
        this.container.appendChild(this.renderer.domElement);
    }
    
    initPostProcessing() {
        // Create composer for bloom effect
        this.composer = new EffectComposer(this.renderer);
        
        // Render pass
        const renderPass = new RenderPass(this.scene, this.cameras[0]);
        this.composer.addPass(renderPass);
        
        // Bloom pass
        const bloomPass = new UnrealBloomPass(
            new THREE.Vector2(
                this.options.HOLOGRAM_WIDTH,
                this.options.HOLOGRAM_HEIGHT
            ),
            this.options.BLOOM_STRENGTH,
            this.options.BLOOM_RADIUS,
            this.options.BLOOM_THRESHOLD
        );
        this.composer.addPass(bloomPass);
    }
    
    initHologram() {
        // Create hologram material
        const material = new THREE.ShaderMaterial({
            uniforms: {
                time: { value: 0 },
                coherence: { value: 1.0 },
                morphAmount: { value: 0 },
                baseColor: { value: new THREE.Color(0x00ffff) },
                meshTexture: { value: null }
            },
            vertexShader: HologramVertexShader,
            fragmentShader: HologramFragmentShader,
            transparent: true,
            side: THREE.DoubleSide
        });
        
        // Create hologram mesh (torus knot for complexity)
        const geometry = new THREE.TorusKnotGeometry(100, 30, 100, 16);
        this.hologramMesh = new THREE.Mesh(geometry, material);
        this.scene.add(this.hologramMesh);
        
        // Add wireframe overlay
        const wireframeMaterial = new THREE.MeshBasicMaterial({
            color: 0x00ffff,
            wireframe: true,
            transparent: true,
            opacity: 0.1
        });
        const wireframeMesh = new THREE.Mesh(geometry, wireframeMaterial);
        this.hologramMesh.add(wireframeMesh);
    }
    
    initQuiltTarget() {
        // Create render target for quilt rendering
        const quiltWidth = this.options.HOLOGRAM_WIDTH;
        const quiltHeight = this.options.HOLOGRAM_HEIGHT;
        
        this.quiltRenderTarget = new THREE.WebGLRenderTarget(
            quiltWidth,
            quiltHeight,
            {
                minFilter: THREE.LinearFilter,
                magFilter: THREE.LinearFilter,
                format: THREE.RGBAFormat,
                type: THREE.UnsignedByteType
            }
        );
    }
    
    // ========================================================================
    // MESH INTEGRATION
    // ========================================================================
    
    updateFromMeshContext(meshContext) {
        this.state.mesh_context = meshContext;
        
        if (!meshContext || !meshContext.nodes) return;
        
        // Clear existing concept nodes
        this.conceptNodes.forEach(node => this.scene.remove(node));
        this.conceptNodes = [];
        
        // Create visual representation of mesh nodes
        meshContext.nodes.forEach((node, index) => {
            const geometry = new THREE.SphereGeometry(5, 16, 16);
            const material = new THREE.MeshPhongMaterial({
                color: new THREE.Color().setHSL(index / meshContext.nodes.length, 1, 0.5),
                emissive: new THREE.Color().setHSL(index / meshContext.nodes.length, 1, 0.5),
                emissiveIntensity: node.confidence || 0.5,
                transparent: true,
                opacity: 0.8
            });
            
            const sphere = new THREE.Mesh(geometry, material);
            
            // Position nodes in 3D space
            const angle = (index / meshContext.nodes.length) * Math.PI * 2;
            const radius = 150 + Math.random() * 50;
            sphere.position.set(
                Math.cos(angle) * radius,
                (Math.random() - 0.5) * 100,
                Math.sin(angle) * radius
            );
            
            // Store node data
            sphere.userData = node;
            
            this.conceptNodes.push(sphere);
            this.scene.add(sphere);
        });
        
        // Create connections between nodes
        this.updateConnections(meshContext.edges);
    }
    
    updateConnections(edges) {
        // Remove existing connections
        if (this.connectionLines) {
            this.scene.remove(this.connectionLines);
        }
        
        if (!edges || edges.length === 0) return;
        
        const geometry = new THREE.BufferGeometry();
        const positions = [];
        const colors = [];
        
        edges.forEach(edge => {
            const sourceNode = this.conceptNodes[edge.source];
            const targetNode = this.conceptNodes[edge.target];
            
            if (sourceNode && targetNode) {
                positions.push(
                    sourceNode.position.x,
                    sourceNode.position.y,
                    sourceNode.position.z,
                    targetNode.position.x,
                    targetNode.position.y,
                    targetNode.position.z
                );
                
                // Color based on edge weight
                const weight = edge.weight || 0.5;
                colors.push(
                    0, weight, 1,
                    0, weight, 1
                );
            }
        });
        
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        
        const material = new THREE.LineBasicMaterial({
            vertexColors: true,
            transparent: true,
            opacity: 0.5,
            blending: THREE.AdditiveBlending
        });
        
        this.connectionLines = new THREE.LineSegments(geometry, material);
        this.scene.add(this.connectionLines);
    }
    
    updateAdapterFeedback(feedback) {
        this.state.adapter_feedback = feedback;
        
        // Update coherence
        if (feedback.coherence !== undefined) {
            this.state.coherence = feedback.coherence;
            
            if (this.hologramMesh) {
                this.hologramMesh.material.uniforms.coherence.value = this.state.coherence;
            }
        }
        
        // Update color based on adapter state
        if (feedback.adapter_active && this.hologramMesh) {
            const color = feedback.adapter_active ? 0x00ff00 : 0xff0000;
            this.hologramMesh.material.uniforms.baseColor.value = new THREE.Color(color);
        }
    }
    
    // ========================================================================
    // QUILT RENDERING
    // ========================================================================
    
    renderQuilt() {
        const rows = this.options.QUILT_ROWS;
        const cols = this.options.QUILT_COLS;
        const viewWidth = this.options.HOLOGRAM_WIDTH / cols;
        const viewHeight = this.options.HOLOGRAM_HEIGHT / rows;
        
        // Set render target
        this.renderer.setRenderTarget(this.quiltRenderTarget);
        this.renderer.clear();
        
        let viewIndex = 0;
        for (let row = 0; row < rows; row++) {
            for (let col = 0; col < cols; col++) {
                if (viewIndex >= this.options.VIEWS) break;
                
                // Set viewport for this view
                this.renderer.setViewport(
                    col * viewWidth,
                    row * viewHeight,
                    viewWidth,
                    viewHeight
                );
                
                // Render with corresponding camera
                this.renderer.render(this.scene, this.cameras[viewIndex]);
                viewIndex++;
            }
        }
        
        // Reset
        this.renderer.setRenderTarget(null);
        this.renderer.setViewport(0, 0, this.options.HOLOGRAM_WIDTH, this.options.HOLOGRAM_HEIGHT);
        
        return this.quiltRenderTarget.texture;
    }
    
    // ========================================================================
    // ANIMATION
    // ========================================================================
    
    startAnimation() {
        const animate = () => {
            this.animationId = requestAnimationFrame(animate);
            
            // Update time
            this.state.time += 0.01;
            
            // Update uniforms
            if (this.hologramMesh) {
                this.hologramMesh.material.uniforms.time.value = this.state.time;
                this.hologramMesh.rotation.y += 0.005;
                this.hologramMesh.rotation.x += 0.002;
            }
            
            // Animate concept nodes
            this.conceptNodes.forEach((node, i) => {
                node.position.y += Math.sin(this.state.time + i) * 0.5;
                node.rotation.y += 0.01;
                
                // Pulse based on confidence
                const scale = 1 + 0.2 * Math.sin(this.state.time * 2 + i) * node.userData.confidence;
                node.scale.setScalar(scale);
            });
            
            // Animate connections
            if (this.connectionLines) {
                this.connectionLines.rotation.y += 0.001;
            }
            
            // Render current view
            const currentCamera = this.cameras[this.state.viewIndex];
            this.composer.render();
            
            // Cycle through views for preview
            this.state.viewIndex = (this.state.viewIndex + 1) % this.options.VIEWS;
        };
        
        animate();
    }
    
    // ========================================================================
    // API METHODS
    // ========================================================================
    
    setMeshContext(meshContext) {
        this.updateFromMeshContext(meshContext);
    }
    
    setAdapterFeedback(feedback) {
        this.updateAdapterFeedback(feedback);
    }
    
    getQuiltTexture() {
        return this.renderQuilt();
    }
    
    morphTo(targetGeometry, duration = 2000) {
        if (this.state.morphing || !this.hologramMesh) return;
        
        this.state.morphing = true;
        
        // Animate morph amount
        const startTime = Date.now();
        const animate = () => {
            const elapsed = Date.now() - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            this.hologramMesh.material.uniforms.morphAmount.value = 
                Math.sin(progress * Math.PI);
            
            if (progress < 1) {
                requestAnimationFrame(animate);
            } else {
                this.state.morphing = false;
                
                // Update geometry
                if (targetGeometry) {
                    this.hologramMesh.geometry.dispose();
                    this.hologramMesh.geometry = targetGeometry;
                }
            }
        };
        
        animate();
    }
    
    // ========================================================================
    // LIFECYCLE
    // ========================================================================
    
    resize() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        
        this.cameras.forEach(camera => {
            camera.aspect = width / height;
            camera.updateProjectionMatrix();
        });
        
        this.renderer.setSize(width, height);
        this.composer.setSize(width, height);
    }
    
    destroy() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        
        // Clean up Three.js
        this.scene.traverse(child => {
            if (child.geometry) child.geometry.dispose();
            if (child.material) {
                if (Array.isArray(child.material)) {
                    child.material.forEach(m => m.dispose());
                } else {
                    child.material.dispose();
                }
            }
        });
        
        this.renderer.dispose();
        this.quiltRenderTarget.dispose();
        
        // Remove from DOM
        if (this.renderer.domElement.parentNode) {
            this.renderer.domElement.parentNode.removeChild(this.renderer.domElement);
        }
        
        console.log('[Hologram Renderer] Destroyed');
    }
}

// ============================================================================
// EXPORTS
// ============================================================================

export default ConceptHologramRenderer;

export function createHologramRenderer(container, options) {
    return new ConceptHologramRenderer(container, options);
}
