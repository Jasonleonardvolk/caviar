<script lang="ts">
    import { onMount, onDestroy } from 'svelte';
    import { ToriHolographicRenderer } from '../lib/holographicRenderer';
    import { psiMemoryStore } from '../stores/psiMemory';
    import { interpolateHologramStates, getHolographicHighlights } from '../../core/psiMemory/psiFrames';
    
    let canvas: HTMLCanvasElement;
    let renderer: ToriHolographicRenderer;
    let animationFrame: number;
    let isInitialized = false;
    let error: string = '';
    let lastFrameTime = 0;
    let fps = 0;
    let interpolationAlpha = 0;
    
    // Reactive state from stores
    $: currentPsiState = $psiMemoryStore.currentState;
    $: hologramHints = $psiMemoryStore.hologramHints;
    
    // Control state
    let isHologramActive = false;
    let renderMode: 'preview' | 'holographic' | 'debug' = 'preview';
    let viewAngle = 0;
    let intensity = 1.0;
    let autoRotate = false;
    let showStats = false;
    
    // Performance monitoring
    let frameCount = 0;
    let lastFpsUpdate = 0;
    
    // Holographic highlights for replay
    let highlights: any[] = [];
    let isPlayingHighlight = false;
    let highlightIndex = 0;
    
    onMount(async () => {
        try {
            await initializeRenderer();
            loadHighlights();
            startRenderLoop();
        } catch (err) {
            error = `WebGPU initialization failed: ${err.message}`;
            console.error('Holographic renderer error:', err);
            
            // Fallback to WebGL if WebGPU not available
            if (!navigator.gpu) {
                error = 'WebGPU not supported. Falling back to WebGL preview mode.';
                await initializeFallbackRenderer();
            }
        }
    });
    
    onDestroy(() => {
        if (animationFrame) {
            cancelAnimationFrame(animationFrame);
        }
        if (renderer) {
            renderer.dispose?.();
        }
    });
    
    async function initializeRenderer() {
        if (!canvas) throw new Error('Canvas not available');
        
        // Check WebGPU support
        if (!navigator.gpu) {
            throw new Error('WebGPU not supported in this browser');
        }
        
        renderer = new ToriHolographicRenderer(canvas);
        await renderer.initialize();
        isInitialized = true;
    }
    
    async function initializeFallbackRenderer() {
        // Implement WebGL fallback for broader compatibility
        const { WebGLFallbackRenderer } = await import('../lib/webglFallbackRenderer');
        renderer = new WebGLFallbackRenderer(canvas);
        await renderer.initialize();
        isInitialized = true;
        renderMode = 'preview'; // Force preview mode for fallback
    }
    
    function startRenderLoop() {
        let lastTime = performance.now();
        
        function render(currentTime: number) {
            const deltaTime = currentTime - lastTime;
            lastTime = currentTime;
            
            // Update FPS counter
            updateFPS(currentTime);
            
            // Update interpolation for smooth animation
            interpolationAlpha += deltaTime * 0.001; // 1 second interpolation
            if (interpolationAlpha > 1) interpolationAlpha = 0;
            
            if (renderer && isHologramActive && currentPsiState) {
                try {
                    // Auto-rotate if enabled
                    if (autoRotate) {
                        viewAngle = (viewAngle + deltaTime * 0.05) % 360;
                    }
                    
                    // Get interpolated state for smooth animation
                    const interpolatedHints = interpolateHologramStates(interpolationAlpha);
                    
                    // Create holographic scene from current ψ-state
                    const scene = createHolographicScene(
                        currentPsiState, 
                        interpolatedHints || hologramHints,
                        { viewAngle, intensity, deltaTime }
                    );
                    
                    // Update renderer settings based on mode
                    renderer.setRenderMode?.(renderMode);
                    renderer.setIntensity?.(intensity);
                    
                    // Render frame
                    renderer.renderFrame(scene, currentPsiState);
                } catch (err) {
                    console.error('Render error:', err);
                    // Don't stop the render loop on errors
                }
            }
            
            animationFrame = requestAnimationFrame(render);
        }
        
        render(performance.now());
    }
    
    function updateFPS(currentTime: number) {
        frameCount++;
        if (currentTime - lastFpsUpdate > 1000) {
            fps = Math.round(frameCount * 1000 / (currentTime - lastFpsUpdate));
            frameCount = 0;
            lastFpsUpdate = currentTime;
        }
    }
    
    function createHolographicScene(psiState: any, hints: any, params: any) {
        const { viewAngle, intensity, deltaTime } = params;
        
        return {
            render(renderPass: GPURenderPassEncoder) {
                // Render ψ-oscillator visualizations
                renderOscillatorField(renderPass, psiState, deltaTime);
                
                // Render semantic anchors with proper depth
                if (hints?.semantic_anchors) {
                    renderSemanticAnchors(renderPass, hints.semantic_anchors, viewAngle);
                }
                
                // Render volumetric density field
                if (hints?.volumetric_density) {
                    renderVolumetricField(renderPass, hints.volumetric_density, intensity);
                }
                
                // Render particle effects based on coherence
                if (psiState.phase_coherence > 0.6) {
                    renderCoherenceParticles(renderPass, psiState.phase_coherence);
                }
                
                // Debug mode overlays
                if (renderMode === 'debug') {
                    renderDebugInfo(renderPass, psiState, hints);
                }
            }
        };
    }
    
    function renderOscillatorField(renderPass: GPURenderPassEncoder, psiState: any, deltaTime: number) {
        // Render oscillator phases as 3D field
        const oscillatorPhases = psiState.oscillator_phases || [];
        const centerX = 0, centerY = 0, centerZ = 0;
        const radius = 2;
        
        oscillatorPhases.forEach((phase, index) => {
            const angle = (index / oscillatorPhases.length) * Math.PI * 2;
            const x = centerX + Math.cos(angle) * radius;
            const y = centerY + Math.sin(phase) * 0.5; // Phase affects height
            const z = centerZ + Math.sin(angle) * radius;
            
            // Render oscillator sphere at position
            // Implementation would create geometry based on oscillator states
        });
    }
    
    function renderSemanticAnchors(renderPass: GPURenderPassEncoder, anchors: any[], viewAngle: number) {
        // Render semantic concept anchors in 3D space
        anchors.forEach((anchor, index) => {
            const angle = (anchor.temporal_position * Math.PI * 2) + (viewAngle * Math.PI / 180);
            const radius = 1.5;
            
            const x = Math.cos(angle) * radius;
            const y = (index / anchors.length) - 0.5; // Distribute vertically
            const z = Math.sin(angle) * radius;
            
            // Render anchor with concept-specific visualization
            // Color and size based on semantic weight
        });
    }
    
    function renderVolumetricField(renderPass: GPURenderPassEncoder, density: number[][][], intensity: number) {
        // Render volumetric density using ray marching
        // Apply intensity scaling to opacity
        const scaledIntensity = intensity * 0.8; // Prevent oversaturation
        
        // Implementation would use compute shaders for volume rendering
    }
    
    function renderCoherenceParticles(renderPass: GPURenderPassEncoder, coherence: number) {
        // Render particle effects based on phase coherence
        const particleCount = Math.floor(coherence * 1000);
        
        // GPU particle system implementation
    }
    
    function renderDebugInfo(renderPass: GPURenderPassEncoder, psiState: any, hints: any) {
        // Render debug overlays
        // Show oscillator connections, phase values, etc.
    }
    
    function toggleHologram() {
        isHologramActive = !isHologramActive;
        if (isHologramActive && !isInitialized) {
            initializeRenderer().catch(err => {
                error = `Failed to start hologram: ${err.message}`;
                isHologramActive = false;
            });
        }
    }
    
    function switchRenderMode(mode: typeof renderMode) {
        renderMode = mode;
        // Update renderer configuration
        if (renderer) {
            renderer.setRenderMode?.(mode);
        }
    }
    
    // Load holographic highlights for replay
    function loadHighlights() {
        highlights = getHolographicHighlights(10);
    }
    
    function playHighlight(index: number) {
        if (index >= 0 && index < highlights.length) {
            highlightIndex = index;
            isPlayingHighlight = true;
            
            const highlight = highlights[index];
            // Update stores with highlight data
            psiMemoryStore.update(store => ({
                ...store,
                currentState: {
                    psi_phase: highlight.hologramData.animationHints?.phaseOffset || 0,
                    phase_coherence: highlight.coherence,
                    ...highlight.hologramData
                },
                hologramHints: highlight.hologramData
            }));
            
            // Auto-stop after 5 seconds
            setTimeout(() => {
                isPlayingHighlight = false;
            }, 5000);
        }
    }
    
    // Export functions for parent components
    export function captureHologram() {
        if (renderer && canvas) {
            return canvas.toDataURL('image/png');
        }
        return null;
    }
    
    export function getHologramState() {
        return {
            isActive: isHologramActive,
            mode: renderMode,
            psiState: currentPsiState,
            hints: hologramHints,
            fps: fps,
            viewAngle: viewAngle,
            intensity: intensity
        };
    }
    
    export function recordHologramicMoment() {
        if (currentPsiState && hologramHints) {
            // Trigger recording in psiMemory
            psiMemoryStore.markHolographicMoment('user_marked');
            loadHighlights(); // Refresh highlights
        }
    }
    
    // Keyboard shortcuts
    function handleKeydown(event: KeyboardEvent) {
        if (!isHologramActive) return;
        
        switch(event.key) {
            case 'r':
                autoRotate = !autoRotate;
                break;
            case 's':
                showStats = !showStats;
                break;
            case 'd':
                switchRenderMode('debug');
                break;
            case 'h':
                switchRenderMode('holographic');
                break;
            case 'p':
                switchRenderMode('preview');
                break;
            case ' ':
                recordHologramicMoment();
                break;
            case 'ArrowLeft':
                viewAngle = (viewAngle - 5) % 360;
                break;
            case 'ArrowRight':
                viewAngle = (viewAngle + 5) % 360;
                break;
            case 'ArrowUp':
                intensity = Math.min(2, intensity + 0.1);
                break;
            case 'ArrowDown':
                intensity = Math.max(0, intensity - 0.1);
                break;
        }
    }
</script>

<svelte:window on:keydown={handleKeydown} />

<div class="holographic-container">
    <!-- Control Panel -->
    <div class="controls">
        <button 
            class="holo-toggle {isHologramActive ? 'active' : ''}"
            on:click={toggleHologram}
            disabled={!isInitialized && !error}
        >
            {#if isHologramActive}
                🔮 Hologram Active
            {:else}
                ✨ Start Hologram
            {/if}
        </button>
        
        <div class="mode-selector">
            <button 
                class="mode-btn {renderMode === 'preview' ? 'active' : ''}"
                on:click={() => switchRenderMode('preview')}
                title="Preview Mode (P)"
            >
                Preview
            </button>
            <button 
                class="mode-btn {renderMode === 'holographic' ? 'active' : ''}"
                on:click={() => switchRenderMode('holographic')}
                title="Holographic Mode (H)"
            >
                Holographic
            </button>
            <button 
                class="mode-btn {renderMode === 'debug' ? 'active' : ''}"
                on:click={() => switchRenderMode('debug')}
                title="Debug Mode (D)"
            >
                Debug
            </button>
        </div>
        
        {#if isHologramActive}
            <div class="controls-group">
                <div class="intensity-control">
                    <label>
                        Intensity: {intensity.toFixed(2)}
                        <input 
                            type="range" 
                            min="0" 
                            max="2" 
                            step="0.1" 
                            bind:value={intensity}
                        />
                    </label>
                </div>
                
                <div class="angle-control">
                    <label>
                        View Angle: {viewAngle.toFixed(0)}°
                        <input 
                            type="range" 
                            min="0" 
                            max="360" 
                            step="5" 
                            bind:value={viewAngle}
                        />
                    </label>
                </div>
                
                <label class="checkbox-control">
                    <input type="checkbox" bind:checked={autoRotate} />
                    Auto-rotate (R)
                </label>
                
                <label class="checkbox-control">
                    <input type="checkbox" bind:checked={showStats} />
                    Show Stats (S)
                </label>
                
                <button 
                    class="action-btn"
                    on:click={recordHologramicMoment}
                    title="Record Holographic Moment (Space)"
                >
                    📸 Capture Moment
                </button>
            </div>
        {/if}
    </div>
    
    <!-- Holographic Highlights -->
    {#if highlights.length > 0}
        <div class="highlights-panel">
            <h3>🌟 Holographic Highlights</h3>
            <div class="highlights-list">
                {#each highlights as highlight, index}
                    <button 
                        class="highlight-btn {isPlayingHighlight && highlightIndex === index ? 'playing' : ''}"
                        on:click={() => playHighlight(index)}
                    >
                        <span class="coherence-badge">{(highlight.coherence * 100).toFixed(0)}%</span>
                        <span class="emotion-type">{highlight.type}</span>
                    </button>
                {/each}
            </div>
        </div>
    {/if}
    
    <!-- Error Display -->
    {#if error}
        <div class="error-panel">
            <h3>⚠️ Holographic System Notice</h3>
            <p>{error}</p>
            <button on:click={() => error = ''}>Dismiss</button>
        </div>
    {/if}
    
    <!-- Holographic Canvas -->
    <div class="canvas-container">
        <canvas 
            bind:this={canvas}
            width="1920"
            height="1080"
            class="holographic-canvas {renderMode}"
        ></canvas>
        
        {#if !isInitialized && !error}
            <div class="loading-overlay">
                <div class="spinner"></div>
                <p>Initializing WebGPU holographic renderer...</p>
            </div>
        {/if}
        
        {#if isHologramActive && currentPsiState}
            <div class="psi-overlay">
                <div class="psi-info">
                    <h4>ψ-State</h4>
                    <div class="psi-phase">
                        Phase: {currentPsiState.psi_phase?.toFixed(3) ?? 'N/A'}
                    </div>
                    <div class="psi-coherence">
                        Coherence: {((currentPsiState.phase_coherence ?? 0) * 100).toFixed(1)}%
                    </div>
                    {#if showStats}
                        <div class="fps-counter">
                            FPS: {fps}
                        </div>
                        <div class="oscillator-count">
                            Oscillators: {currentPsiState.oscillator_phases?.length ?? 0}
                        </div>
                    {/if}
                </div>
                
                {#if hologramHints?.emotional_flow || currentPsiState.emotional_resonance}
                    {@const emotions = hologramHints?.emotional_flow || currentPsiState.emotional_resonance}
                    <div class="emotion-info">
                        <h4>Emotional Resonance</h4>
                        <div class="emotion-bar">
                            <span>Excitement</span>
                            <div class="bar">
                                <div 
                                    class="fill excitement" 
                                    style="width: {(emotions.excitement * 100)}%"
                                ></div>
                            </div>
                        </div>
                        <div class="emotion-bar">
                            <span>Calmness</span>
                            <div class="bar">
                                <div 
                                    class="fill calmness" 
                                    style="width: {(emotions.calmness * 100)}%"
                                ></div>
                            </div>
                        </div>
                        <div class="emotion-bar">
                            <span>Energy</span>
                            <div class="bar">
                                <div 
                                    class="fill energy" 
                                    style="width: {(emotions.energy * 100)}%"
                                ></div>
                            </div>
                        </div>
                        <div class="emotion-bar">
                            <span>Clarity</span>
                            <div class="bar">
                                <div 
                                    class="fill clarity" 
                                    style="width: {(emotions.clarity * 100)}%"
                                ></div>
                            </div>
                        </div>
                    </div>
                {/if}
                
                {#if hologramHints?.temporal_coherence}
                    <div class="temporal-info">
                        <h4>Temporal Coherence</h4>
                        <div class="beat-frequency">
                            Beat: {hologramHints.temporal_coherence.beat_frequency?.toFixed(2) ?? 0} Hz
                        </div>
                        <div class="phase-stability">
                            Stability: {((hologramHints.temporal_coherence.phase_stability ?? 0) * 100).toFixed(1)}%
                        </div>
                    </div>
                {/if}
            </div>
            
            <!-- Keyboard shortcuts help -->
            <div class="keyboard-help">
                <span>R: Rotate</span>
                <span>S: Stats</span>
                <span>D/H/P: Modes</span>
                <span>Space: Capture</span>
                <span>↑↓: Intensity</span>
                <span>←→: Angle</span>
            </div>
        {/if}
    </div>
</div>

<style>
    .holographic-container {
        position: relative;
        width: 100%;
        height: 100vh;
        background: radial-gradient(circle at center, #0a0a2e 0%, #000000 100%);
        display: flex;
        flex-direction: column;
        overflow: hidden;
    }
    
    .controls {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 1rem;
        background: rgba(0, 0, 0, 0.8);
        border-bottom: 1px solid #333;
        backdrop-filter: blur(10px);
        z-index: 10;
    }
    
    .controls-group {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-left: auto;
    }
    
    .holo-toggle {
        padding: 0.5rem 1rem;
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 6px;
        color: white;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 2px 10px rgba(102, 126, 234, 0.3);
    }
    
    .holo-toggle:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.5);
    }
    
    .holo-toggle.active {
        background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%);
        box-shadow: 0 0 30px rgba(240, 147, 251, 0.6);
        animation: pulse 2s infinite;
    }
    
    .mode-selector {
        display: flex;
        gap: 0.5rem;
        background: rgba(255, 255, 255, 0.05);
        padding: 0.25rem;
        border-radius: 6px;
    }
    
    .mode-btn {
        padding: 0.3rem 0.8rem;
        background: transparent;
        border: 1px solid transparent;
        color: #999;
        border-radius: 4px;
        cursor: pointer;
        transition: all 0.2s;
        font-size: 0.9rem;
    }
    
    .mode-btn:hover {
        color: #ccc;
        background: rgba(255, 255, 255, 0.1);
    }
    
    .mode-btn.active {
        background: rgba(102, 126, 234, 0.3);
        color: white;
        border-color: #667eea;
    }
    
    .intensity-control, .angle-control {
        display: flex;
        flex-direction: column;
        gap: 0.2rem;
    }
    
    .intensity-control label, .angle-control label {
        color: #ccc;
        font-size: 0.85rem;
        display: flex;
        flex-direction: column;
        gap: 0.3rem;
    }
    
    .checkbox-control {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: #ccc;
        font-size: 0.85rem;
        cursor: pointer;
    }
    
    .checkbox-control input {
        cursor: pointer;
    }
    
    .action-btn {
        padding: 0.4rem 0.8rem;
        background: rgba(102, 126, 234, 0.2);
        border: 1px solid #667eea;
        color: #ccc;
        border-radius: 4px;
        cursor: pointer;
        transition: all 0.2s;
        font-size: 0.85rem;
    }
    
    .action-btn:hover {
        background: rgba(102, 126, 234, 0.4);
        color: white;
    }
    
    .highlights-panel {
        position: absolute;
        top: 5rem;
        left: 1rem;
        background: rgba(0, 0, 0, 0.9);
        border: 1px solid #333;
        border-radius: 8px;
        padding: 1rem;
        max-width: 200px;
        z-index: 5;
        backdrop-filter: blur(10px);
    }
    
    .highlights-panel h3 {
        margin: 0 0 0.5rem 0;
        color: #667eea;
        font-size: 1rem;
    }
    
    .highlights-list {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }
    
    .highlight-btn {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid #333;
        color: #ccc;
        border-radius: 4px;
        cursor: pointer;
        transition: all 0.2s;
        font-size: 0.8rem;
        text-align: left;
    }
    
    .highlight-btn:hover {
        background: rgba(102, 126, 234, 0.2);
        border-color: #667eea;
    }
    
    .highlight-btn.playing {
        background: rgba(240, 147, 251, 0.3);
        border-color: #f093fb;
        animation: pulse 1s infinite;
    }
    
    .coherence-badge {
        background: #667eea;
        color: white;
        padding: 0.2rem 0.4rem;
        border-radius: 3px;
        font-weight: bold;
        font-size: 0.7rem;
    }
    
    .emotion-type {
        flex: 1;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    
    .canvas-container {
        flex: 1;
        position: relative;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    
    .holographic-canvas {
        max-width: 100%;
        max-height: 100%;
        border-radius: 8px;
        box-shadow: 0 0 50px rgba(102, 126, 234, 0.3);
        transition: box-shadow 0.3s ease;
    }
    
    .holographic-canvas.holographic {
        box-shadow: 0 0 100px rgba(240, 147, 251, 0.6);
    }
    
    .holographic-canvas.debug {
        box-shadow: 0 0 50px rgba(255, 200, 0, 0.3);
    }
    
    .loading-overlay {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        text-align: center;
        color: #ccc;
    }
    
    .spinner {
        width: 50px;
        height: 50px;
        border: 3px solid rgba(102, 126, 234, 0.3);
        border-top: 3px solid #667eea;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin: 0 auto 1rem;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    @keyframes pulse {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.8; transform: scale(1.05); }
        100% { opacity: 1; transform: scale(1); }
    }
    
    .error-panel {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: rgba(20, 20, 20, 0.95);
        border: 1px solid #666;
        border-radius: 8px;
        padding: 2rem;
        text-align: center;
        color: #ff9900;
        max-width: 400px;
        backdrop-filter: blur(10px);
        z-index: 100;
    }
    
    .error-panel h3 {
        margin: 0 0 1rem 0;
        color: #ff9900;
    }
    
    .error-panel button {
        margin-top: 1rem;
        padding: 0.5rem 1rem;
        background: #ff9900;
        color: black;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-weight: bold;
    }
    
    .psi-overlay {
        position: absolute;
        top: 1rem;
        right: 1rem;
        background: rgba(0, 0, 0, 0.9);
        border: 1px solid #333;
        border-radius: 8px;
        padding: 1rem;
        min-width: 220px;
        color: #ccc;
        backdrop-filter: blur(10px);
        max-height: 80vh;
        overflow-y: auto;
    }
    
    .psi-info h4, .emotion-info h4, .temporal-info h4 {
        margin: 0 0 0.5rem 0;
        color: #667eea;
        font-size: 0.9rem;
        font-weight: 600;
    }
    
    .psi-phase, .psi-coherence, .fps-counter, .oscillator-count, 
    .beat-frequency, .phase-stability {
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        margin: 0.3rem 0;
        color: #ddd;
    }
    
    .emotion-info {
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid #333;
    }
    
    .temporal-info {
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid #333;
    }
    
    .emotion-bar {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin: 0.4rem 0;
    }
    
    .emotion-bar span {
        font-size: 0.75rem;
        min-width: 65px;
        color: #999;
    }
    
    .bar {
        flex: 1;
        height: 6px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 3px;
        overflow: hidden;
    }
    
    .fill {
        height: 100%;
        transition: width 0.3s ease;
        border-radius: 3px;
    }
    
    .fill.excitement {
        background: linear-gradient(90deg, #ff6b6b, #ff8e53);
    }
    
    .fill.calmness {
        background: linear-gradient(90deg, #4ecdc4, #44a08d);
    }
    
    .fill.energy {
        background: linear-gradient(90deg, #feca57, #ff9ff3);
    }
    
    .fill.clarity {
        background: linear-gradient(90deg, #54a0ff, #667eea);
    }
    
    .keyboard-help {
        position: absolute;
        bottom: 1rem;
        left: 50%;
        transform: translateX(-50%);
        display: flex;
        gap: 1rem;
        padding: 0.5rem 1rem;
        background: rgba(0, 0, 0, 0.8);
        border-radius: 4px;
        font-size: 0.75rem;
        color: #666;
        opacity: 0.7;
        transition: opacity 0.2s;
    }
    
    .keyboard-help:hover {
        opacity: 1;
    }
    
    .keyboard-help span {
        white-space: nowrap;
    }
    
    /* Custom scrollbar for overlays */
    .psi-overlay::-webkit-scrollbar {
        width: 6px;
    }
    
    .psi-overlay::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 3px;
    }
    
    .psi-overlay::-webkit-scrollbar-thumb {
        background: #667eea;
        border-radius: 3px;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .controls {
            flex-wrap: wrap;
        }
        
        .controls-group {
            width: 100%;
            justify-content: space-between;
        }
        
        .highlights-panel {
            display: none;
        }
        
        .keyboard-help {
            display: none;
        }
        
        .psi-overlay {
            right: 0.5rem;
            top: 0.5rem;
            min-width: 180px;
            font-size: 0.8rem;
        }
    }
</style>