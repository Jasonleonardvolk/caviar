<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import ParameterPanel from './components/ParameterPanel.svelte';
  import MeshSummaryPanel from './components/MeshSummaryPanel.svelte';
  import AVStatus from './components/AVStatus.svelte';
  import LogPanel from './components/LogPanel.svelte';
  import { sseClient } from './lib/sseClient';
  import { handleRenderError } from './lib/errorHandler';
  import { isWebGPUAvailable, getDeviceTier } from './lib/deviceDetect';
  
  // Plugin system
  export let plugins = [];
  let pluginComponents = new Map();
  
  // State
  let renderMode = 'webgpu';
  let deviceTier = 'medium';
  let meshSummary = {};
  let avStatus = { audio: false, video: false };
  let logs = [];
  let showDiagnostics = false;
  let controlState = {
    phaseMode: 'Kerr',
    personaState: 'neutral',
    viewMode: 'quilt',
    blendRatio: 0.5
  };
  
  // SSE subscription
  let sseUnsubscribe = null;
  
  onMount(async () => {
    // Detect device capabilities
    deviceTier = getDeviceTier();
    renderMode = isWebGPUAvailable() ? 'webgpu' : 'wasm';
    
    // Initialize SSE connection
    try {
      sseUnsubscribe = sseClient.subscribe((event) => {
        handleSSEEvent(event);
      });
      await sseClient.connect();
    } catch (err) {
      console.error('SSE connection failed:', err);
    }
    
    // Load plugins dynamically
    for (const plugin of plugins) {
      try {
        const module = await import(`./plugins/${plugin.name}.svelte`);
        pluginComponents.set(plugin.name, module.default);
      } catch (err) {
        console.warn(`Failed to load plugin ${plugin.name}:`, err);
      }
    }
    
    // Handle render errors globally
    window.addEventListener('error', handleGlobalError);
    window.addEventListener('switchToBaseRender', switchToFallback);
    
    // Load saved state
    loadPersistedState();
  });
  
  onDestroy(() => {
    if (sseUnsubscribe) sseUnsubscribe();
    sseClient.disconnect();
    window.removeEventListener('error', handleGlobalError);
    window.removeEventListener('switchToBaseRender', switchToFallback);
  });
  
  function handleSSEEvent(event) {
    // Update appropriate panel based on event type
    switch(event.type) {
      case 'mesh_updated':
        meshSummary = event.data;
        break;
      case 'av_status':
        avStatus = event.data;
        break;
      case 'log':
        logs = [...logs, event.data].slice(-100); // Keep last 100 logs
        break;
      case 'adapter_swap':
        logs = [...logs, { 
          level: 'info', 
          message: `Adapter swapped to ${event.data.adapter}`,
          timestamp: new Date().toISOString()
        }].slice(-100);
        break;
    }
  }
  
  function handleGlobalError(event) {
    handleRenderError(event.error || new Error(event.message));
  }
  
  function switchToFallback() {
    renderMode = 'fallback';
    logs = [...logs, {
      level: 'warning',
      message: 'Switched to fallback render mode',
      timestamp: new Date().toISOString()
    }].slice(-100);
  }
  
  function handleControlUpdate(event) {
    controlState = { ...controlState, ...event.detail };
    persistState();
    
    // Dispatch to renderer
    window.dispatchEvent(new CustomEvent('controlUpdate', { 
      detail: controlState 
    }));
  }
  
  function loadPersistedState() {
    try {
      const saved = localStorage.getItem('tori_control_state');
      if (saved) {
        controlState = { ...controlState, ...JSON.parse(saved) };
      }
    } catch (err) {
      console.warn('Failed to load persisted state:', err);
    }
  }
  
  function persistState() {
    try {
      localStorage.setItem('tori_control_state', JSON.stringify(controlState));
    } catch (err) {
      console.warn('Failed to persist state:', err);
    }
  }
</script>

<main class="tori-app">
  <header class="app-header">
    <h1>TORI Hybrid Holography</h1>
    <div class="status-bar">
      <span class="device-tier tier-{deviceTier}">{deviceTier.toUpperCase()}</span>
      <span class="render-mode mode-{renderMode}">{renderMode.toUpperCase()}</span>
      {#if renderMode === 'wasm' || renderMode === 'fallback'}
        <span class="reduced-quality">⚠️ Reduced Quality Mode</span>
      {/if}
    </div>
  </header>
  
  <div class="app-container">
    <aside class="control-sidebar">
      <ParameterPanel 
        bind:state={controlState}
        on:update={handleControlUpdate}
      />
      
      <MeshSummaryPanel 
        summary={meshSummary}
      />
      
      <AVStatus 
        status={avStatus}
      />
      
      <details class="diagnostics-panel" bind:open={showDiagnostics}>
        <summary>Diagnostics</summary>
        <ul>
          <li>Device Tier: {deviceTier}</li>
          <li>Render Mode: {renderMode}</li>
          <li>WebGPU: {isWebGPUAvailable() ? 'Available' : 'Not Available'}</li>
          <li>View Indices: {window.currentViewIndices?.h || 0}, {window.currentViewIndices?.v || 0}</li>
        </ul>
      </details>
      
      <!-- Plugin slots -->
      {#each plugins as plugin}
        {#if pluginComponents.has(plugin.name)}
          <div class="plugin-slot" data-plugin={plugin.name}>
            <svelte:component 
              this={pluginComponents.get(plugin.name)} 
              {...plugin.props}
            />
          </div>
        {/if}
      {/each}
    </aside>
    
    <div class="render-container" id="render-target">
      <canvas id="hologram-canvas"></canvas>
      {#if renderMode === 'fallback'}
        <div class="fallback-overlay">
          <p>Running in fallback mode. Some features may be limited.</p>
        </div>
      {/if}
    </div>
    
    <aside class="log-sidebar">
      <LogPanel logs={logs} />
    </aside>
  </div>
</main>

<style>
  :global(body) {
    margin: 0;
    padding: 0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #000;
    color: #fff;
  }
  
  .tori-app {
    display: flex;
    flex-direction: column;
    height: 100vh;
    overflow: hidden;
  }
  
  .app-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem;
    background: linear-gradient(90deg, #1a1a2e, #16213e);
    border-bottom: 1px solid #333;
  }
  
  .app-header h1 {
    margin: 0;
    font-size: 1.5rem;
    background: linear-gradient(90deg, #00d2ff, #3a7bd5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  
  .status-bar {
    display: flex;
    gap: 1rem;
    align-items: center;
  }
  
  .status-bar span {
    padding: 0.25rem 0.75rem;
    border-radius: 4px;
    font-size: 0.875rem;
    font-weight: 600;
  }
  
  .device-tier {
    background: #2a2a3e;
  }
  
  .tier-high { color: #4ade80; }
  .tier-medium { color: #fbbf24; }
  .tier-low { color: #f87171; }
  
  .render-mode {
    background: #1e293b;
  }
  
  .mode-webgpu { color: #4ade80; }
  .mode-wasm { color: #fbbf24; }
  .mode-fallback { color: #f87171; }
  
  .reduced-quality {
    color: #fbbf24;
    animation: pulse 2s infinite;
  }
  
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }
  
  .app-container {
    display: grid;
    grid-template-columns: 280px 1fr 280px;
    flex: 1;
    overflow: hidden;
  }
  
  .control-sidebar,
  .log-sidebar {
    background: #0f172a;
    border: 1px solid #1e293b;
    overflow-y: auto;
    padding: 1rem;
  }
  
  .render-container {
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #000;
  }
  
  #hologram-canvas {
    max-width: 100%;
    max-height: 100%;
    display: block;
  }
  
  .fallback-overlay {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    padding: 1rem;
    background: rgba(239, 68, 68, 0.1);
    border-bottom: 2px solid #ef4444;
    text-align: center;
  }
  
  .diagnostics-panel {
    margin-top: 1rem;
    padding: 0.5rem;
    background: #1e293b;
    border-radius: 4px;
  }
  
  .diagnostics-panel summary {
    cursor: pointer;
    font-weight: 600;
  }
  
  .diagnostics-panel ul {
    margin: 0.5rem 0 0;
    padding-left: 1.5rem;
    font-size: 0.875rem;
    color: #94a3b8;
  }
  
  .plugin-slot {
    margin-top: 1rem;
    padding: 1rem;
    background: #1e293b;
    border-radius: 4px;
    border: 1px solid #334155;
  }
  
  /* Mobile responsiveness */
  @media (max-width: 768px) {
    .app-container {
      grid-template-columns: 1fr;
    }
    
    .control-sidebar,
    .log-sidebar {
      position: fixed;
      top: 0;
      bottom: 0;
      width: 280px;
      z-index: 1000;
      transform: translateX(-100%);
      transition: transform 0.3s;
    }
    
    .control-sidebar.open,
    .log-sidebar.open {
      transform: translateX(0);
    }
    
    .log-sidebar {
      right: 0;
      left: auto;
      transform: translateX(100%);
    }
  }
</style>
