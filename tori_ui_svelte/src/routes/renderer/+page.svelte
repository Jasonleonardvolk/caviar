<script lang="ts">
  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  import HolographicDisplay from '$lib/components/HolographicDisplay.svelte';
  import { darkMode } from '$lib/stores/darkMode';
  
  // Feature flag from environment
  const useCanonical = Boolean(import.meta.env.VITE_TORI_HOLOGRAM_CANONICAL ?? 1);
  
  let displayReady = false;
  let displayConfig = {
    width: 800,
    height: 600,
    usePenrose: false,
    showStats: true,
    enableVideo: false,
    videoSource: 'webcam' as 'webcam' | 'file' | 'stream'
  };
  
  onMount(() => {
    // Check for WebGPU support
    if (!navigator.gpu) {
      console.warn('WebGPU not supported, will use fallback renderer');
    }
    
    displayReady = true;
  });
</script>

<svelte:head>
  <title>TORI Holographic Renderer</title>
  <meta name="description" content="Advanced holographic display renderer for TORI system" />
</svelte:head>

<div class="renderer-container" class:dark={$darkMode}>
  <header class="renderer-header">
    <h1>Holographic Renderer</h1>
    <div class="status-bar">
      {#if displayReady}
        <span class="status-indicator active">Ready</span>
      {:else}
        <span class="status-indicator loading">Initializing...</span>
      {/if}
      {#if useCanonical}
        <span class="version-tag">Canonical</span>
      {/if}
    </div>
  </header>
  
  <main class="renderer-main">
    {#if displayReady && useCanonical}
      <HolographicDisplay 
        width={displayConfig.width}
        height={displayConfig.height}
        usePenrose={displayConfig.usePenrose}
        showStats={displayConfig.showStats}
        enableVideo={displayConfig.enableVideo}
        videoSource={displayConfig.videoSource}
      />
    {:else if displayReady}
      <div class="fallback-message">
        <p>Holographic renderer is disabled. Set VITE_TORI_HOLOGRAM_CANONICAL=1 to enable.</p>
      </div>
    {:else}
      <div class="loading-spinner">
        <div class="spinner"></div>
        <p>Initializing holographic renderer...</p>
      </div>
    {/if}
  </main>
  
  <footer class="renderer-footer">
    <div class="controls">
      <button 
        on:click={() => displayConfig.showStats = !displayConfig.showStats}
        aria-label="Toggle statistics display"
      >
        Stats: {displayConfig.showStats ? 'ON' : 'OFF'}
      </button>
      <button 
        on:click={() => displayConfig.usePenrose = !displayConfig.usePenrose}
        aria-label="Toggle Penrose mode"
      >
        Penrose: {displayConfig.usePenrose ? 'ON' : 'OFF'}
      </button>
    </div>
  </footer>
</div>

<style>
  .renderer-container {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    background: var(--bg-primary, #ffffff);
    color: var(--text-primary, #000000);
    font-family: system-ui, -apple-system, sans-serif;
  }
  
  .renderer-container.dark {
    --bg-primary: #0a0a0a;
    --text-primary: #ffffff;
    --border-color: rgba(255, 255, 255, 0.1);
  }
  
  .renderer-header {
    padding: 1rem 2rem;
    border-bottom: 1px solid var(--border-color, rgba(0, 0, 0, 0.1));
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  
  .renderer-header h1 {
    margin: 0;
    font-size: 1.5rem;
    font-weight: 600;
  }
  
  .status-bar {
    display: flex;
    gap: 1rem;
    align-items: center;
  }
  
  .status-indicator {
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.875rem;
    font-weight: 500;
  }
  
  .status-indicator.active {
    background: #10b981;
    color: white;
  }
  
  .status-indicator.loading {
    background: #f59e0b;
    color: white;
  }
  
  .version-tag {
    padding: 0.25rem 0.5rem;
    background: rgba(59, 130, 246, 0.1);
    color: #3b82f6;
    border-radius: 0.25rem;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
  }
  
  .renderer-main {
    flex: 1;
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 2rem;
  }
  
  .loading-spinner {
    text-align: center;
  }
  
  .spinner {
    width: 40px;
    height: 40px;
    margin: 0 auto 1rem;
    border: 3px solid rgba(59, 130, 246, 0.3);
    border-top-color: #3b82f6;
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }
  
  @keyframes spin {
    to { transform: rotate(360deg); }
  }
  
  .fallback-message {
    padding: 2rem;
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 0.5rem;
    color: #ef4444;
  }
  
  .renderer-footer {
    padding: 1rem 2rem;
    border-top: 1px solid var(--border-color, rgba(0, 0, 0, 0.1));
  }
  
  .controls {
    display: flex;
    gap: 1rem;
  }
  
  .controls button {
    padding: 0.5rem 1rem;
    background: rgba(59, 130, 246, 0.1);
    color: #3b82f6;
    border: 1px solid #3b82f6;
    border-radius: 0.25rem;
    font-size: 0.875rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
  }
  
  .controls button:hover {
    background: #3b82f6;
    color: white;
  }
  
  .controls button:focus {
    outline: 2px solid #3b82f6;
    outline-offset: 2px;
  }
</style>
