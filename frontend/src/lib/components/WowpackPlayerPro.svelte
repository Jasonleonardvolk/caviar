<script lang="ts">
  import { onMount, onDestroy } from 'svelte';

  type FileItem = { name: string; size: number; mtime: string; contentType: string };
  type Clip = { id: string; title: string; files: FileItem[]; mastersPresent: boolean };
  type Catalog = { items: Clip[] };

  let catalog: Catalog | null = null;
  let chosen: Clip | null = null;
  let chosenFile: FileItem | null = null;
  let error = '';
  let collapsed = false;
  let showcaseMode = false;
  let showcaseIndex = 0;
  let showcaseInterval: number | null = null;

  let videoEl: HTMLVideoElement | null = null;
  let progress = 0;
  
  // Performance metrics
  let fps = 0;
  let memoryUsed = 0;
  let frameTime = 0;
  let lastTime = performance.now();
  let frameCount = 0;

  const prefer = ['video/mp4','video/webm','video/mp4','video/quicktime','video/x-matroska'];

  function bestPlayable(files: FileItem[]): FileItem | null {
    if (!files?.length) return null;
    const v = document.createElement('video');
    for (const mime of prefer) {
      const candidate = files.find(f => f.contentType === mime);
      if (!candidate) continue;
      const ok = v.canPlayType(mime);
      if (ok === 'probably' || ok === 'maybe') return candidate;
    }
    return files[0] || null;
  }

  async function load() {
    try {
      const r = await fetch('/api/wowpack/list', { cache: 'no-store' });
      catalog = await r.json();
      const withFiles = catalog.items.find(i => i.files?.length);
      if (withFiles) {
        chosen = withFiles;
        chosenFile = bestPlayable(withFiles.files);
      } else {
        chosen = catalog.items[0] ?? null;
      }
    } catch (e:any) {
      error = String(e?.message || e);
    }
  }

  function pick(id: string) {
    if (!catalog) return;
    const found = catalog.items.find(i => i.id === id);
    chosen = found || null;
    chosenFile = found ? bestPlayable(found.files) : null;
  }

  function toggle() { collapsed = !collapsed; }

  function toggleShowcase() {
    showcaseMode = !showcaseMode;
    if (showcaseMode) {
      startShowcase();
    } else {
      stopShowcase();
    }
  }

  function startShowcase() {
    if (!catalog?.items?.length) return;
    
    showcaseInterval = window.setInterval(() => {
      showcaseIndex = (showcaseIndex + 1) % catalog.items.length;
      const item = catalog.items[showcaseIndex];
      if (item) pick(item.id);
    }, 8000); // Switch every 8 seconds
  }

  function stopShowcase() {
    if (showcaseInterval) {
      clearInterval(showcaseInterval);
      showcaseInterval = null;
    }
  }

  function updateProgress() {
    if (videoEl && videoEl.duration > 0) {
      progress = (videoEl.currentTime / videoEl.duration) * 100;
    } else {
      progress = 0;
    }
  }

  function updatePerformance() {
    frameCount++;
    const now = performance.now();
    const delta = now - lastTime;
    
    if (delta >= 1000) {
      fps = Math.round((frameCount * 1000) / delta);
      frameTime = Math.round(delta / frameCount);
      frameCount = 0;
      lastTime = now;
      
      // Memory usage if available
      if ('memory' in performance) {
        const mem = (performance as any).memory;
        memoryUsed = Math.round(mem.usedJSHeapSize / 1048576); // MB
      }
    }
  }

  let perfInterval: number;
  let progressInterval: number;

  onMount(() => {
    load();
    progressInterval = window.setInterval(updateProgress, 200);
    perfInterval = window.setInterval(updatePerformance, 100);
    
    return () => {
      if (progressInterval) clearInterval(progressInterval);
      if (perfInterval) clearInterval(perfInterval);
      stopShowcase();
    };
  });

  onDestroy(() => {
    stopShowcase();
  });
</script>

<style>
  .hud { position:absolute; bottom:1rem; right:1rem; width:360px; z-index:50; font-size:0.85rem; }
  .wrap { border:1px solid #2b2b2b; border-radius:16px; padding:.5rem .75rem;
          background:rgba(13,13,13,0.92); backdrop-filter: blur(10px); color:#fff; 
          box-shadow: 0 8px 32px rgba(0,0,0,0.5); }
  .header { display:flex; justify-content:space-between; align-items:center; gap:0.5rem; }
  .chip { border:1px solid #444; border-radius:999px; padding:.15rem .5rem; background:#111;
          cursor:pointer; font-size:0.75rem; transition: all 0.2s; }
  .chip:hover { transform: scale(1.05); border-color: #666; }
  .chip.active { border-color:#8ec5ff; color:#8ec5ff; background: rgba(142,197,255,0.1); }
  .chip.showcase { background: linear-gradient(135deg, #667eea, #764ba2); color: white; border: none; }
  video { width:100%; border-radius:12px; background:#000; display:block; margin-top:0.5rem; }
  small.mono { font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace; opacity:.75; font-size: 0.7rem; }
  .progress-bar { height:3px; border-radius:2px; background:#1a1a1a; margin-top:.25rem; overflow:hidden; position: relative; }
  .progress { height:100%; background:linear-gradient(90deg,#4ac7ff,#00ffa3); width:0%; transition:width 0.2s; 
              box-shadow: 0 0 10px rgba(74,199,255,0.5); }
  .metrics { display:flex; gap:1rem; margin-top:0.5rem; font-size:0.7rem; opacity:0.8; }
  .metric { display:flex; align-items:center; gap:0.25rem; }
  .metric-label { opacity:0.6; }
  .metric-value { color:#00ffa3; font-family: 'SF Mono', monospace; }
  .pills { display:flex; gap:0.35rem; flex-wrap:wrap; margin:0.5rem 0; }
  
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }
  
  .showcase-indicator {
    animation: pulse 2s infinite;
    font-size: 0.65rem;
    color: #00ffa3;
    margin-left: 0.5rem;
  }
</style>

<div class="hud">
  <div class="wrap">
    <div class="header">
      <div style="display:flex; align-items:center;">
        <b>🎬 WOW Pack</b>
        {#if showcaseMode}
          <span class="showcase-indicator">● AUTO</span>
        {/if}
      </div>
      <div style="display:flex; gap:0.35rem;">
        <button class="chip showcase" on:click={toggleShowcase} title="Auto-cycle videos">
          {showcaseMode ? '⏸' : '▶'}
        </button>
        <button class="chip" on:click={toggle}>{collapsed ? '▲' : '▼'}</button>
      </div>
    </div>

    {#if !collapsed}
      <div class="pills">
        {#if catalog}
          {#each catalog.items as c}
            <span class={"chip " + (chosen?.id===c.id ? 'active' : '')} on:click={() => pick(c.id)}>
              {c.title}
              {#if c.files.length > 0}
                <span style="opacity:0.5; font-size:0.65rem;">({c.files.length})</span>
              {/if}
            </span>
          {/each}
        {/if}
      </div>

      {#if error}
        <div style="color:#ff6b6b; padding:0.5rem; background:rgba(255,0,0,0.1); border-radius:8px;">
          Failed to load: {error}
        </div>
      {/if}

      {#if chosenFile}
        <video bind:this={videoEl} src={"/api/wowpack/file/" + encodeURIComponent(chosenFile.name)} 
               muted autoplay loop playsinline></video>
        <div class="progress-bar">
          <div class="progress" style="width:{progress}%"></div>
        </div>
        
        <div class="metrics">
          <div class="metric">
            <span class="metric-label">FPS:</span>
            <span class="metric-value">{fps}</span>
          </div>
          <div class="metric">
            <span class="metric-label">Frame:</span>
            <span class="metric-value">{frameTime}ms</span>
          </div>
          {#if memoryUsed > 0}
            <div class="metric">
              <span class="metric-label">Memory:</span>
              <span class="metric-value">{memoryUsed}MB</span>
            </div>
          {/if}
        </div>
        
        <small class="mono">
          📹 {chosenFile.name} • {(chosenFile.size / 1048576).toFixed(1)}MB • {chosenFile.contentType}
        </small>
      {:else if chosen}
        <div style="padding:1rem; background:rgba(255,255,255,0.05); border-radius:8px; margin-top:0.5rem;">
          <div style="color:#ff6b6b; margin-bottom:0.5rem;">⚠️ No playable encodes</div>
          {#if chosen.mastersPresent}
            <div style="color:#00ffa3;">✅ ProRes masters present</div>
            <small>Run encoder to generate web versions</small>
          {:else}
            <small>Add ProRes to content/wowpack/input/</small>
          {/if}
        </div>
      {/if}
    {/if}
  </div>
</div>