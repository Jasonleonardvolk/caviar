<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { Capacitor } from '@capacitor/core';
  import { App } from '@capacitor/app';
  import { StatusBar } from '@capacitor/status-bar';
  import { Haptics, ImpactStyle } from '@capacitor/haptics';
  import MobileHolographicEngine, { MobileQualityPreset } from './holographicEngine';
  import { getTelemetryService } from './telemetry';
  
  let canvas: HTMLCanvasElement;
  let engine: MobileHolographicEngine;
  let telemetry = getTelemetryService();
  
  let isConnected = false;
  let currentQuality: MobileQualityPreset = MobileQualityPreset.BALANCED;
  let showSettings = false;
  let showPairing = false;
  let jwt: string = '';
  let desktopUrl: string = '';
  
  // Performance stats
  let fps = 0;
  let frameTime = 0;
  let gpuMemory = 0;
  
  // Error state
  let error: string | null = null;
  
  onMount(async () => {
    try {
      // Hide status bar for immersive experience
      if (Capacitor.getPlatform() !== 'web') {
        await StatusBar.hide();
      }
      
      // Handle deep links for pairing
      App.addListener('appUrlOpen', (event) => {
        handleDeepLink(event.url);
      });
      
      // Check if launched with pairing URL
      const launchUrl = await App.getLaunchUrl();
      if (launchUrl?.url) {
        handleDeepLink(launchUrl.url);
      }
      
      // Initialize engine
      await initializeEngine();
      
      // Start render loop
      requestAnimationFrame(render);
      
    } catch (err) {
      error = err.message;
      telemetry.recordError(err, { context: 'initialization' });
    }
  });
  
  onDestroy(() => {
    if (engine) {
      engine.destroy();
    }
    telemetry.destroy();
  });
  
  async function initializeEngine() {
    engine = new MobileHolographicEngine();
    
    try {
      await engine.initialize(canvas);
      
      // Log successful initialization
      telemetry.logEvent('engine_initialized', {
        quality: engine.getCurrentQuality(),
        platform: Capacitor.getPlatform()
      });
      
    } catch (err) {
      if (err.message.includes('WebGPU not supported')) {
        error = 'Your device does not support WebGPU. Please update your browser or OS.';
      } else {
        error = 'Failed to initialize holographic engine';
      }
      throw err;
    }
  }
  
  function handleDeepLink(url: string) {
    // Parse tori-holo://pair?jwt=XXX&session=YYY
    const urlObj = new URL(url);
    
    if (urlObj.pathname === 'pair') {
      jwt = urlObj.searchParams.get('jwt') || '';
      const sessionId = urlObj.searchParams.get('session') || '';
      
      if (jwt) {
        showPairing = true;
        attemptPairing();
      }
    }
  }
  
  async function attemptPairing() {
    try {
      // Extract desktop URL from JWT claims (simplified)
      const payload = JSON.parse(atob(jwt.split('.')[1]));
      desktopUrl = payload.desktop_url || 'http://localhost:7690';
      
      // Attempt to connect
      await engine.enableStreaming(jwt);
      isConnected = true;
      showPairing = false;
      
      // Haptic feedback on successful connection
      await Haptics.impact({ style: ImpactStyle.Light });
      
      telemetry.logEvent('pairing_success', {
        capabilities: payload.capabilities
      });
      
    } catch (err) {
      telemetry.recordError(err, { context: 'pairing' });
      alert('Failed to connect to desktop. Please try again.');
    }
  }
  
  function render(timestamp: number) {
    if (!engine || error) return;
    
    // Render frame
    engine.render();
    
    // Update stats every second
    if (timestamp % 1000 < 16) {
      updateStats();
    }
    
    requestAnimationFrame(render);
  }
  
  async function updateStats() {
    const stats = await engine.getStats();
    fps = stats.fps || 0;
    frameTime = stats.frameTime || 0;
    gpuMemory = stats.gpuMemory || 0;
    
    // Log performance metrics
    if (fps > 0) {
      telemetry.recordFrameMetrics({
        fps,
        frameTime,
        quality: currentQuality,
        gpuTime: stats.gpuTime
      });
    }
  }
  
  async function changeQuality(preset: MobileQualityPreset) {
    const oldQuality = currentQuality;
    currentQuality = preset;
    
    await engine.setQuality(preset);
    
    // Haptic feedback
    await Haptics.impact({ style: ImpactStyle.Light });
    
    telemetry.recordQualityChange(
      oldQuality, 
      preset, 
      'user_selection'
    );
  }
  
  function toggleSettings() {
    showSettings = !showSettings;
  }
  
  async function disconnect() {
    await engine.disableStreaming();
    isConnected = false;
    jwt = '';
    
    telemetry.logEvent('disconnected', {
      reason: 'user_action'
    });
  }
  
  function formatMemory(bytes: number): string {
    return (bytes / 1024 / 1024).toFixed(1) + ' MB';
  }
</script>

<div class="app-container">
  {#if error}
    <div class="error-screen">
      <div class="error-icon">⚠️</div>
      <h2>Unable to Start</h2>
      <p>{error}</p>
      <button on:click={() => window.location.reload()}>
        Retry
      </button>
    </div>
  {:else}
    <canvas 
      bind:this={canvas} 
      class="hologram-canvas"
      class:streaming={isConnected}
    ></canvas>
    
    <!-- Performance overlay -->
    <div class="stats-overlay">
      <div class="stat">
        <span class="label">FPS</span>
        <span class="value">{fps.toFixed(0)}</span>
      </div>
      <div class="stat">
        <span class="label">Frame</span>
        <span class="value">{frameTime.toFixed(1)}ms</span>
      </div>
      {#if gpuMemory > 0}
        <div class="stat">
          <span class="label">GPU</span>
          <span class="value">{formatMemory(gpuMemory)}</span>
        </div>
      {/if}
    </div>
    
    <!-- Connection status -->
    <div class="connection-status" class:connected={isConnected}>
      {#if isConnected}
        <span class="status-icon">📡</span>
        <span>Streaming from Desktop</span>
      {:else}
        <span class="status-icon">📱</span>
        <span>Local Rendering</span>
      {/if}
    </div>
    
    <!-- Settings button -->
    <button class="settings-btn" on:click={toggleSettings}>
      ⚙️
    </button>
    
    <!-- Settings panel -->
    {#if showSettings}
      <div class="settings-panel">
        <h3>Settings</h3>
        
        <div class="setting-group">
          <label>Quality</label>
          <div class="quality-buttons">
            <button 
              class:active={currentQuality === MobileQualityPreset.BATTERY_SAVER}
              on:click={() => changeQuality(MobileQualityPreset.BATTERY_SAVER)}
            >
              🔋 Battery
            </button>
            <button 
              class:active={currentQuality === MobileQualityPreset.BALANCED}
              on:click={() => changeQuality(MobileQualityPreset.BALANCED)}
            >
              ⚖️ Balanced
            </button>
            <button 
              class:active={currentQuality === MobileQualityPreset.PERFORMANCE}
              on:click={() => changeQuality(MobileQualityPreset.PERFORMANCE)}
            >
              🚀 Performance
            </button>
          </div>
        </div>
        
        {#if isConnected}
          <div class="setting-group">
            <button class="disconnect-btn" on:click={disconnect}>
              Disconnect from Desktop
            </button>
          </div>
        {:else}
          <div class="setting-group">
            <button class="pair-btn" on:click={() => showPairing = true}>
              Pair with Desktop
            </button>
          </div>
        {/if}
        
        <button class="close-btn" on:click={() => showSettings = false}>
          Close
        </button>
      </div>
    {/if}
    
    <!-- Pairing dialog -->
    {#if showPairing && !isConnected}
      <div class="pairing-dialog">
        <h3>Pair with Desktop</h3>
        <p>Open the TORI Hologram desktop app and scan the QR code shown there.</p>
        <p>Or enter the pairing code manually:</p>
        <input 
          type="text" 
          placeholder="Enter pairing code"
          on:keyup={(e) => {
            if (e.key === 'Enter' && e.target.value.length === 8) {
              // Manual pairing with code
              manualPair(e.target.value);
            }
          }}
        />
        <button on:click={() => showPairing = false}>Cancel</button>
      </div>
    {/if}
  {/if}
</div>

<style>
  .app-container {
    position: relative;
    width: 100vw;
    height: 100vh;
    background: #000;
    overflow: hidden;
  }
  
  .hologram-canvas {
    width: 100%;
    height: 100%;
    touch-action: none;
  }
  
  .hologram-canvas.streaming {
    /* Add visual indicator for streaming mode */
    box-shadow: inset 0 0 20px rgba(0, 255, 255, 0.3);
  }
  
  .error-screen {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100vh;
    padding: 2rem;
    text-align: center;
    color: white;
  }
  
  .error-icon {
    font-size: 4rem;
    margin-bottom: 1rem;
  }
  
  .error-screen h2 {
    margin-bottom: 1rem;
  }
  
  .error-screen button {
    margin-top: 2rem;
    padding: 0.75rem 1.5rem;
    background: #0066ff;
    color: white;
    border: none;
    border-radius: 25px;
    font-size: 1rem;
  }
  
  .stats-overlay {
    position: absolute;
    top: env(safe-area-inset-top, 20px);
    left: env(safe-area-inset-left, 20px);
    display: flex;
    gap: 1rem;
    font-family: monospace;
    font-size: 0.8rem;
    color: rgba(255, 255, 255, 0.8);
    background: rgba(0, 0, 0, 0.5);
    padding: 0.5rem;
    border-radius: 8px;
    backdrop-filter: blur(10px);
  }
  
  .stat {
    display: flex;
    flex-direction: column;
    align-items: center;
  }
  
  .stat .label {
    font-size: 0.7rem;
    opacity: 0.7;
  }
  
  .stat .value {
    font-weight: bold;
  }
  
  .connection-status {
    position: absolute;
    top: env(safe-area-inset-top, 20px);
    right: env(safe-area-inset-right, 20px);
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    background: rgba(0, 0, 0, 0.5);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    color: white;
    font-size: 0.85rem;
  }
  
  .connection-status.connected {
    background: rgba(0, 255, 100, 0.2);
  }
  
  .settings-btn {
    position: absolute;
    bottom: env(safe-area-inset-bottom, 20px);
    right: env(safe-area-inset-right, 20px);
    width: 50px;
    height: 50px;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    font-size: 1.5rem;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  
  .settings-panel {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    background: rgba(20, 20, 30, 0.95);
    backdrop-filter: blur(20px);
    padding: 2rem;
    padding-bottom: env(safe-area-inset-bottom, 2rem);
    border-radius: 20px 20px 0 0;
    color: white;
  }
  
  .settings-panel h3 {
    margin-bottom: 1.5rem;
    text-align: center;
  }
  
  .setting-group {
    margin-bottom: 1.5rem;
  }
  
  .setting-group label {
    display: block;
    margin-bottom: 0.5rem;
    opacity: 0.8;
  }
  
  .quality-buttons {
    display: flex;
    gap: 0.5rem;
  }
  
  .quality-buttons button {
    flex: 1;
    padding: 0.75rem;
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 10px;
    color: white;
    transition: all 0.2s;
  }
  
  .quality-buttons button.active {
    background: rgba(0, 150, 255, 0.3);
    border-color: #0096ff;
  }
  
  .disconnect-btn, .pair-btn {
    width: 100%;
    padding: 1rem;
    background: rgba(255, 50, 50, 0.2);
    border: 1px solid rgba(255, 50, 50, 0.5);
    border-radius: 10px;
    color: white;
  }
  
  .pair-btn {
    background: rgba(0, 150, 255, 0.2);
    border-color: rgba(0, 150, 255, 0.5);
  }
  
  .close-btn {
    width: 100%;
    padding: 1rem;
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 10px;
    color: white;
  }
  
  .pairing-dialog {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background: rgba(20, 20, 30, 0.95);
    backdrop-filter: blur(20px);
    padding: 2rem;
    border-radius: 20px;
    color: white;
    width: 90%;
    max-width: 400px;
  }
  
  .pairing-dialog input {
    width: 100%;
    padding: 0.75rem;
    margin: 1rem 0;
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 10px;
    color: white;
    font-size: 1.2rem;
    text-align: center;
    letter-spacing: 0.1em;
  }
</style>
