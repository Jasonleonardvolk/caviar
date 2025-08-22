<script lang="ts">
    import { onMount, onDestroy } from 'svelte';
    // THE REAL CONNECTION! 🎉
    import { RealGhostEngine } from '../realGhostEngine.js';
    import { psiMemoryStore } from '../stores/psiMemory';
    import { interpolateHologramStates, getHolographicHighlights } from '../../core/psiMemory/psiFrames';
    
    let canvas: HTMLCanvasElement;
    let renderer: RealGhostEngine; // Using the REAL engine now!
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
    
    // Audio connection status
    let audioConnected = false;
    let wsStatus = 'disconnected';
    
    onMount(async () => {
        try {
            console.log('🚀 Initializing CONNECTED Holographic System!');
            await initializeRenderer();
            loadHighlights();
            // Don't start render loop here - RealGhostEngine handles it!
        } catch (err) {
  const msg = err instanceof Error ? err.message : String(err);
            error = `Initialization failed: ${msg
}`;
            console.error('Holographic renderer error:', err);
        }
    });
    
    onDestroy(() => {
        if (renderer) {
            renderer.destroy();
        }
    });
    
    async function initializeRenderer() {
        if (!canvas) throw new Error('Canvas not available');
        
        // Check WebGPU support
        if (!navigator.gpu) {
            throw new Error('WebGPU not supported in this browser');
        }
        
        // Initialize the REAL Ghost Engine!
        renderer = new RealGhostEngine();
        
        // Get display type from environment or default
        const displayType = window.TORI_DISPLAY_TYPE || 'webgpu_only';
        
        const result = await renderer.initialize(canvas, {
            displayType,
            quality: 'high',
            enableAudio: true,
            enableHoTT: true
        });
        
        if (result.success) {
            console.log('✅ Real Ghost Engine initialized!', result.capabilities);
            isInitialized = true;
            
            // Check WebSocket status
            if (renderer.wsConnection) {
                wsStatus = 'connected';
                audioConnected = true;
            }
        }
    }
    
    function createHolographicScene(psiState: any, hints: any, params: any) {
        // The Real Ghost Engine handles scene creation internally
        // We just need to update the state
        if (renderer && psiState) {
            renderer.updateFromOscillator(psiState);
            
            if (hints) {
                renderer.updateFromWavefieldParams({
                    phase_modulation: psiState.psi_phase || 0,
                    coherence: psiState.phase_coherence || 0.8,
                    oscillator_phases: psiState.oscillator_phases || [],
                    dominant_freq: psiState.dominant_frequency || 440,
                    spatial_frequencies: hints.spatial_frequencies || [],
                    amplitudes: hints.amplitudes || []
                });
            }
        }
    }
    
    function toggleHologram() {
        isHologramActive = !isHologramActive;
        if (isHologramActive && renderer) {
            // Update render mode
            renderer.setRenderMode?.(renderMode);
            
            // Start audio if available
            if (audioConnected) {
                startAudioCapture();
            }
        }
    }
    
    async function startAudioCapture() {
        try {
            // Get microphone access
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const source = audioContext.createMediaStreamSource(stream);
            const processor = audioContext.createScriptProcessor(4096, 1, 1);
            
            processor.onaudioprocess = (e) => {
                const inputData = e.inputBuffer.getChannelData(0);
                
                // Send to backend via Ghost Engine
                if (renderer && renderer.wsConnection) {
                    renderer.sendMessage({
                        type: 'audio_data',
                        samples: Array.from(inputData)
                    });
                }
            };
            
            source.connect(processor);
            processor.connect(audioContext.destination);
            
            console.log('🎤 Audio capture started!');
        } catch (err) {
  const msg = err instanceof Error ? err.message : String(err);
            console.error('Audio capture error:', err);
        
}
    }
    
    function switchRenderMode(mode: typeof renderMode) {
        renderMode = mode;
        if (renderer) {
            renderer.setRenderMode?.(mode);
        }
    }
    
    // Watch for psi state changes
    $: if (renderer && isHologramActive && currentPsiState) {
        createHolographicScene(currentPsiState, hologramHints, {
            viewAngle,
            intensity,
            deltaTime: 16
        });
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
        if (renderer) {
            return renderer.captureHologram();
        }
        return null;
    }
    
    export function getHologramState() {
        const stats = renderer?.render() || { fps: 0 };
        return {
            isActive: isHologramActive,
            mode: renderMode,
            psiState: currentPsiState,
            hints: hologramHints,
            fps: stats.fps,
            viewAngle: viewAngle,
            intensity: intensity,
            audioConnected,
            wsStatus
        };
    }
    
    export function recordHologramicMoment() {
        if (currentPsiState && hologramHints) {
            psiMemoryStore.markHolographicMoment('user_marked');
            loadHighlights();
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
                if (renderer) renderer.setIntensity?.(intensity);
                break;
            case 'ArrowDown':
                intensity = Math.max(0, intensity - 0.1);
                if (renderer) renderer.setIntensity?.(intensity);
                break;
        }
    }
    
    // Update FPS from engine
    setInterval(() => {
        if (renderer && isHologramActive) {
            const stats = renderer.render();
            fps = stats.fps;
        }
    }, 100);
</script>

<svelte:window on:keydown={handleKeydown} />

<div class="holographic-container">
    <!-- Status Bar showing connection status -->
    <div class="status-bar">
        <div class="status-item">
            <span class="status-dot {isInitialized ? 'active' : ''}"></span>
            WebGPU
        </div>
        <div class="status-item">
            <span class="status-dot {audioConnected ? 'active' : ''}"></span>
            Audio
        </div>
        <div class="status-item">
            <span class="status-dot {wsStatus === 'connected' ? 'active' : ''}"></span>
            Backend
        </div>
        <div class="status-item">
            FPS: {fps}
        </div>
    </div>
    
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
                            on:input={() => renderer?.setIntensity?.(intensity)}
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
                
                {#if audioConnected}
                    <button 
                        class="action-btn audio-active"
                        on:click={startAudioCapture}
                        title="Audio is connected"
                    >
                        🎤 Audio Active
                    </button>
                {/if}
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
            class="holographic-canvas {renderMode} {isInitialized ? 'initialized' : ''}"
        ></canvas>
        
        {#if !isInitialized && !error}
            <div class="loading-overlay">
                <div class="spinner"></div>
                <p>Initializing CONNECTED WebGPU holographic system...</p>
            </div>
        {/if}
        
        {#if isHologramActive && currentPsiState && showStats}
            <div class="psi-overlay">
                <div class="psi-info">
                    <h4>ψ-State (LIVE)</h4>
                    <div class="psi-phase">
                        Phase: {currentPsiState.psi_phase?.toFixed(3) ?? 'N/A'}
                    </div>
                    <div class="psi-coherence">
                        Coherence: {((currentPsiState.phase_coherence ?? 0) * 100).toFixed(1)}%
                    </div>
                    <div class="oscillator-count">
                        Oscillators: {currentPsiState.oscillator_phases?.length ?? 0}
                    </div>
                    <div class="ws-status">
                        WebSocket: {wsStatus}
                    </div>
                </div>
            </div>
        {/if}
    </div>
</div>

<style>
    /* All the existing styles from the original component */
    /* Plus these additions: */
    
    .status-bar {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 30px;
        background: rgba(0, 0, 0, 0.9);
        display: flex;
        align-items: center;
        padding: 0 1rem;
        gap: 2rem;
        font-size: 0.85rem;
        z-index: 100;
        border-bottom: 1px solid #333;
    }
    
    .status-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: #999;
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #333;
        transition: all 0.3s;
    }
    
    .status-dot.active {
        background: #51cf66;
        box-shadow: 0 0 10px #51cf66;
    }
    
    .holographic-canvas.initialized {
        box-shadow: 0 0 100px rgba(102, 126, 234, 0.6);
    }
    
    .action-btn.audio-active {
        background: rgba(81, 207, 102, 0.3);
        border-color: #51cf66;
    }
    
    .ws-status {
        color: #51cf66;
        font-family: 'Courier New', monospace;
        font-size: 0.8rem;
    }
    
    /* Keep all other styles from original component... */
</style>