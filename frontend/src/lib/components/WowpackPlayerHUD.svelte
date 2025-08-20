<script lang="ts">
  import { onMount } from 'svelte';

  type FileItem = { name: string; size: number; mtime: string; contentType: string };
  type Clip = { id: string; title: string; files: FileItem[]; mastersPresent: boolean };
  type Catalog = { items: Clip[] };

  let catalog: Catalog | null = null;
  let chosen: Clip | null = null;
  let chosenFile: FileItem | null = null;
  let videoEl: HTMLVideoElement | null = null;
  let error = '';
  let minimized = false;
  let hudMode = false; // Set to true for overlay mode

  const prefer = ['video/mp4', 'video/webm', 'video/mp4', 'video/quicktime', 'video/x-matroska'];

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
    error = '';
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
    if (minimized && chosenFile) minimized = false; // Auto-expand when selecting
  }

  onMount(load);
</script>

<style>
  /* Standard mode - below canvas */
  .wrap { 
    border: 1px solid #2b2b2b; 
    border-radius: 16px; 
    padding: .75rem; 
    background: rgba(13, 13, 13, 0.95);
    backdrop-filter: blur(10px);
    color: #fff; 
    margin-top: 1rem;
    transition: all 0.3s ease;
  }
  
  /* HUD overlay mode - floating */
  .wrap.hud {
    position: fixed;
    bottom: 20px;
    right: 20px;
    max-width: 420px;
    margin-top: 0;
    z-index: 1000;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
    background: rgba(13, 13, 13, 0.85);
  }
  
  .wrap.minimized {
    padding: 0.5rem;
  }
  
  .wrap.minimized .content {
    display: none;
  }
  
  .header {
    display: flex;
    gap: .5rem;
    justify-content: space-between;
    align-items: center;
  }
  
  .title-section {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  
  .controls {
    display: flex;
    gap: 0.3rem;
  }
  
  .control-btn {
    width: 24px;
    height: 24px;
    border: 1px solid #444;
    border-radius: 6px;
    background: rgba(30, 30, 30, 0.8);
    color: #888;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.2s;
    font-size: 12px;
  }
  
  .control-btn:hover {
    background: rgba(50, 50, 50, 0.9);
    color: #fff;
    border-color: #666;
  }
  
  .control-btn.active {
    background: rgba(142, 197, 255, 0.2);
    border-color: #8ec5ff;
    color: #8ec5ff;
  }
  
  .row { 
    display: flex; 
    gap: .5rem; 
    flex-wrap: wrap; 
    align-items: center; 
    margin-top: 0.5rem;
  }
  
  .chip { 
    border: 1px solid #444; 
    border-radius: 999px; 
    padding: .25rem .6rem; 
    background: rgba(17, 17, 17, 0.8);
    cursor: pointer; 
    transition: all 0.2s;
    font-size: 0.85rem;
  }
  
  .chip:hover { 
    border-color: #6ea5ff;
    background: rgba(30, 30, 30, 0.9);
  }
  
  .chip.active { 
    border-color: #8ec5ff; 
    color: #8ec5ff; 
    background: rgba(26, 42, 58, 0.8);
  }
  
  video { 
    width: 100%; 
    max-width: 540px; 
    border-radius: 12px; 
    background: #000; 
    display: block;
    margin-top: 0.75rem;
  }
  
  .wrap.hud video {
    max-width: 400px;
  }
  
  small.mono { 
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; 
    opacity: .75;
    font-size: 0.75rem;
  }
  
  .open-btn { 
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
    color: white; 
    text-decoration: none;
    font-size: 0.8rem;
  }
  
  .open-btn:hover { 
    transform: scale(1.05); 
  }
  
  .status-bar {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-top: 0.5rem;
    padding: 0.5rem;
    background: rgba(0, 0, 0, 0.3);
    border-radius: 8px;
    font-size: 0.8rem;
  }
  
  .pulse {
    display: inline-block;
    width: 8px;
    height: 8px;
    background: #4ade80;
    border-radius: 50%;
    animation: pulse 2s infinite;
  }
  
  @keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(1.2); }
  }
</style>

<div class="wrap" class:hud={hudMode} class:minimized={minimized}>
  <div class="header">
    <div class="title-section">
      <b>🎬 {minimized ? 'WOW' : 'WOW Pack Demo Loops'}</b>
      {#if !minimized}
        <small class="mono">output/</small>
      {/if}
    </div>
    
    <div class="controls">
      {#if chosenFile && !minimized}
        <a class="chip open-btn" href={"/api/wowpack/file/" + encodeURIComponent(chosenFile.name)} target="_blank">📥</a>
      {/if}
      <button class="control-btn" class:active={hudMode} on:click={() => hudMode = !hudMode} title="Toggle HUD mode">
        ⬚
      </button>
      <button class="control-btn" on:click={() => minimized = !minimized} title="{minimized ? 'Expand' : 'Minimize'}">
        {minimized ? '▲' : '▼'}
      </button>
    </div>
  </div>

  <div class="content">
    <div class="row">
      {#if catalog}
        {#each catalog.items as c}
          <div class={"chip " + (chosen?.id===c.id ? 'active' : '')} on:click={() => pick(c.id)}>
            {c.title}
            {#if c.files.length > 0}
              <span style="opacity:0.5">({c.files.length})</span>
            {/if}
          </div>
        {/each}
      {/if}
    </div>

    {#if error}
      <div style="margin-top:.75rem; color:#ff6b6b">Failed to load: {error}</div>
    {/if}

    {#if chosen}
      {#if chosen.files.length === 0}
        <div style="margin-top:.75rem; padding: 1rem; background: rgba(26, 26, 26, 0.5); border-radius: 8px;">
          No web-playable encodes for <b>{chosen.title}</b>
          {#if chosen.mastersPresent}
            <div style="margin-top: 0.5rem; color: #8ec5ff;">
              ✅ Master ready in input/{chosen.id}.mov
            </div>
          {:else}
            <div style="margin-top: 0.5rem; color: #ff6b6b;">
              ❌ Add master to input/{chosen.id}.mov
            </div>
          {/if}
        </div>
      {:else if chosenFile}
        <video 
          bind:this={videoEl} 
          src={"/api/wowpack/file/" + encodeURIComponent(chosenFile.name)} 
          muted 
          autoplay 
          loop 
          playsinline 
          controls
        ></video>
        
        <div class="status-bar">
          <span class="pulse"></span>
          <small class="mono">
            {chosenFile.name} • {(chosenFile.size / 1024 / 1024).toFixed(1)} MB
          </small>
        </div>
      {/if}
    {/if}
  </div>
</div>