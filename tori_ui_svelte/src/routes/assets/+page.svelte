<script lang="ts">
  import { onMount } from 'svelte';
  import manifestData from '../../../assets/3d/luxury/ASSET_MANIFEST.json';
  
  interface Asset {
    id: string;
    category: string;
    brandless: boolean;
    license: string;
    source: string;
    source_url: string;
    raw: string;
    opt: string;
    polycount_hint: number;
    units: string;
    scale_m: number;
    centered: boolean;
    pivot: string;
  }
  
  const assets: Asset[] = manifestData;
  const base = '/'; // Adjust if behind a proxy
  
  let selectedAsset: Asset | null = null;
  let viewerLoaded = false;
  let arSupported = false;
  
  onMount(() => {
    // Check for AR support
    if ('xr' in navigator) {
      // @ts-ignore
      navigator.xr?.isSessionSupported('immersive-ar').then((supported: boolean) => {
        arSupported = supported;
      });
    }
    
    // Mark viewer as loaded when model-viewer is ready
    setTimeout(() => {
      viewerLoaded = true;
    }, 1000);
  });
  
  function selectAsset(asset: Asset) {
    selectedAsset = asset;
  }
  
  function formatFileSize(bytes: number): string {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / 1048576).toFixed(2) + ' MB';
  }
</script>

<svelte:head>
  <title>iRis AR Assets - Free 3D Models</title>
  <meta name="description" content="Free 3D models for AR experiences in iRis" />
  <!-- model-viewer web component for 3D/AR viewing -->
  <script type="module" src="https://unpkg.com/@google/model-viewer@3.3.0/dist/model-viewer.min.js"></script>
</svelte:head>

<div class="assets-container">
  <header class="assets-header">
    <h1>iRis AR Assets</h1>
    <p class="subtitle">Free 3D Models for Augmented Reality</p>
    {#if arSupported}
      <span class="ar-badge">AR Ready</span>
    {/if}
  </header>
  
  <div class="assets-grid">
    <aside class="assets-sidebar">
      <h2>Available Models</h2>
      <div class="model-list">
        {#each assets as asset}
          <button 
            class="model-card"
            class:selected={selectedAsset?.id === asset.id}
            on:click={() => selectAsset(asset)}
          >
            <div class="model-info">
              <h3>{asset.id.replace(/_/g, ' ')}</h3>
              <span class="category">{asset.category}</span>
              <span class="license">{asset.license}</span>
            </div>
            {#if asset.polycount_hint}
              <div class="polycount">~{asset.polycount_hint.toLocaleString()} polys</div>
            {/if}
          </button>
        {/each}
      </div>
    </aside>
    
    <main class="viewer-container">
      {#if selectedAsset}
        <div class="viewer-wrapper">
          <model-viewer
            src={`${base}${selectedAsset.opt}`}
            alt={selectedAsset.id}
            ar
            ar-modes="webxr scene-viewer quick-look"
            camera-controls
            touch-action="pan-y"
            autoplay
            auto-rotate
            auto-rotate-delay="2000"
            rotation-per-second="30deg"
            exposure="0.9"
            shadow-intensity="1"
            shadow-softness="0.5"
            style="width: 100%; height: 100%;"
          >
            <button slot="ar-button" class="ar-button">
              View in AR
            </button>
            
            <div class="progress-bar" slot="progress-bar">
              <div class="update-bar"></div>
            </div>
          </model-viewer>
          
          <div class="asset-details">
            <h2>{selectedAsset.id.replace(/_/g, ' ')}</h2>
            <dl>
              <dt>Category:</dt>
              <dd>{selectedAsset.category}</dd>
              
              <dt>License:</dt>
              <dd>{selectedAsset.license}</dd>
              
              <dt>Scale:</dt>
              <dd>{selectedAsset.scale_m}m</dd>
              
              <dt>Source:</dt>
              <dd>
                <a href={selectedAsset.source_url} target="_blank" rel="noreferrer">
                  {selectedAsset.source}
                </a>
              </dd>
              
              {#if selectedAsset.license.includes('CC')}
                <dt>Attribution:</dt>
                <dd class="attribution">
                  Model from {selectedAsset.source}, licensed under {selectedAsset.license}
                </dd>
              {/if}
            </dl>
          </div>
        </div>
      {:else}
        <div class="no-selection">
          <h2>Select a Model</h2>
          <p>Choose a 3D model from the sidebar to preview it here.</p>
          {#if arSupported}
            <p class="ar-hint">Your device supports AR - try viewing models in your space!</p>
          {/if}
        </div>
      {/if}
    </main>
  </div>
</div>

<style>
  .assets-container {
    min-height: 100vh;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    font-family: system-ui, -apple-system, sans-serif;
  }
  
  .assets-header {
    padding: 2rem;
    text-align: center;
    color: white;
    position: relative;
  }
  
  .assets-header h1 {
    font-size: 2.5rem;
    font-weight: 700;
    margin: 0;
    text-shadow: 0 2px 4px rgba(0,0,0,0.2);
  }
  
  .subtitle {
    margin: 0.5rem 0;
    opacity: 0.9;
    font-size: 1.1rem;
  }
  
  .ar-badge {
    position: absolute;
    top: 2rem;
    right: 2rem;
    background: #10b981;
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 9999px;
    font-size: 0.875rem;
    font-weight: 600;
    animation: pulse 2s infinite;
  }
  
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
  }
  
  .assets-grid {
    display: grid;
    grid-template-columns: 320px 1fr;
    gap: 2rem;
    padding: 0 2rem 2rem;
    max-width: 1400px;
    margin: 0 auto;
  }
  
  .assets-sidebar {
    background: white;
    border-radius: 1rem;
    padding: 1.5rem;
    box-shadow: 0 20px 25px -5px rgba(0,0,0,0.1);
    max-height: calc(100vh - 200px);
    overflow-y: auto;
  }
  
  .assets-sidebar h2 {
    margin: 0 0 1rem;
    font-size: 1.25rem;
    color: #1f2937;
  }
  
  .model-list {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }
  
  .model-card {
    background: #f9fafb;
    border: 2px solid transparent;
    border-radius: 0.5rem;
    padding: 1rem;
    text-align: left;
    cursor: pointer;
    transition: all 0.2s;
  }
  
  .model-card:hover {
    background: #f3f4f6;
    transform: translateX(4px);
  }
  
  .model-card.selected {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-color: #667eea;
  }
  
  .model-info h3 {
    margin: 0 0 0.25rem;
    font-size: 1rem;
    font-weight: 600;
    text-transform: capitalize;
  }
  
  .category, .license {
    display: inline-block;
    font-size: 0.75rem;
    padding: 0.125rem 0.5rem;
    border-radius: 9999px;
    margin-right: 0.5rem;
    margin-top: 0.25rem;
  }
  
  .model-card:not(.selected) .category {
    background: #ddd6fe;
    color: #6b21a8;
  }
  
  .model-card:not(.selected) .license {
    background: #fef3c7;
    color: #92400e;
  }
  
  .model-card.selected .category,
  .model-card.selected .license {
    background: rgba(255,255,255,0.2);
  }
  
  .polycount {
    margin-top: 0.5rem;
    font-size: 0.75rem;
    opacity: 0.7;
  }
  
  .viewer-container {
    background: white;
    border-radius: 1rem;
    overflow: hidden;
    box-shadow: 0 20px 25px -5px rgba(0,0,0,0.1);
    min-height: 600px;
  }
  
  .viewer-wrapper {
    height: 100%;
    display: flex;
    flex-direction: column;
  }
  
  model-viewer {
    flex: 1;
    min-height: 500px;
  }
  
  .ar-button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 0.5rem;
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    cursor: pointer;
    position: absolute;
    bottom: 1rem;
    right: 1rem;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
  }
  
  .ar-button:hover {
    transform: scale(1.05);
  }
  
  .progress-bar {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: rgba(0,0,0,0.1);
  }
  
  .update-bar {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    height: 100%;
    width: 0;
    animation: progress 2s;
  }
  
  @keyframes progress {
    to { width: 100%; }
  }
  
  .asset-details {
    padding: 1.5rem;
    border-top: 1px solid #e5e7eb;
  }
  
  .asset-details h2 {
    margin: 0 0 1rem;
    text-transform: capitalize;
    color: #1f2937;
  }
  
  .asset-details dl {
    display: grid;
    grid-template-columns: 120px 1fr;
    gap: 0.5rem 1rem;
    margin: 0;
  }
  
  .asset-details dt {
    font-weight: 600;
    color: #6b7280;
  }
  
  .asset-details dd {
    margin: 0;
    color: #1f2937;
  }
  
  .asset-details a {
    color: #667eea;
    text-decoration: none;
  }
  
  .asset-details a:hover {
    text-decoration: underline;
  }
  
  .attribution {
    font-size: 0.875rem;
    font-style: italic;
    color: #6b7280;
  }
  
  .no-selection {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    padding: 3rem;
    text-align: center;
    color: #6b7280;
  }
  
  .no-selection h2 {
    margin: 0 0 1rem;
    color: #1f2937;
  }
  
  .ar-hint {
    margin-top: 1rem;
    padding: 1rem;
    background: #f0fdf4;
    color: #166534;
    border-radius: 0.5rem;
    font-weight: 500;
  }
  
  /* Mobile responsive */
  @media (max-width: 768px) {
    .assets-grid {
      grid-template-columns: 1fr;
    }
    
    .assets-sidebar {
      max-height: 200px;
    }
    
    .ar-badge {
      position: static;
      display: inline-block;
      margin-top: 1rem;
    }
  }
</style>