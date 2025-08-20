<script lang="ts">
  import { onMount } from 'svelte';
  
  export let videoEl: HTMLVideoElement | undefined = undefined;

  type FileItem = { name: string; size: number; mtime: string; contentType: string };
  type Clip = { id: string; title: string; files: FileItem[]; mastersPresent: boolean };
  type Catalog = { items: Clip[] };

  let catalog: Catalog | null = null;
  let chosen: Clip | null = null;
  let chosenFile: FileItem | null = null;
  let error = '';
  let collapsed = false;

  const prefer = ['video/mp4', 'video/webm', 'video/quicktime'];

  function bestPlayable(files: FileItem[]): FileItem | null {
    if (!files?.length) return null;
    for (const mime of prefer) {
      const candidate = files.find(f => f.contentType === mime);
      if (candidate) return candidate;
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
        if (chosenFile && videoEl) {
          setVideo(chosenFile.name);
        }
      } else {
        chosen = catalog.items[0] ?? null;
      }
    } catch (e: any) {
      error = String(e?.message || e);
    }
  }

  function pick(id: string) {
    if (!catalog) return;
    const found = catalog.items.find(i => i.id === id);
    chosen = found || null;
    chosenFile = found ? bestPlayable(found.files) : null;
    if (chosenFile && videoEl) {
      setVideo(chosenFile.name);
    }
  }

  function setVideo(filename: string) {
    if (videoEl) {
      videoEl.src = "/api/wowpack/file/" + encodeURIComponent(filename);
      videoEl.play().catch(() => {});
    }
  }

  function toggle() {
    collapsed = !collapsed;
  }

  onMount(() => {
    load();
  });
</script>

<style>
  .hud {
    position: absolute;
    top: 1rem;
    right: 1rem;
    width: 320px;
    z-index: 50;
    font-size: 0.85rem;
  }
  .wrap {
    border: 1px solid rgba(142, 197, 255, 0.3);
    border-radius: 16px;
    padding: 0.5rem 0.75rem;
    background: rgba(13, 13, 13, 0.85);
    backdrop-filter: blur(10px);
    color: #fff;
    box-shadow: 0 0 20px rgba(142, 197, 255, 0.2);
  }
  .header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .chip {
    border: 1px solid #444;
    border-radius: 999px;
    padding: 0.15rem 0.5rem;
    background: #111;
    cursor: pointer;
    font-size: 0.75rem;
    color: #fff;
    font-family: inherit;
    transition: all 0.2s;
  }
  .chip:hover {
    transform: scale(1.05);
    border-color: #666;
  }
  .chip.active {
    border-color: #8ec5ff;
    color: #8ec5ff;
    background: rgba(142, 197, 255, 0.1);
  }
  .chips {
    display: flex;
    gap: 0.35rem;
    flex-wrap: wrap;
    margin-top: 0.5rem;
  }
  .status {
    margin-top: 0.5rem;
    font-size: 0.7rem;
    color: #8ec5ff;
    font-family: 'SF Mono', monospace;
  }
</style>

<div class="hud">
  <div class="wrap">
    <div class="header">
      <div><b>HOLOGRAM SOURCE</b></div>
      <button class="chip" on:click={toggle}>{collapsed ? 'SHOW' : 'HIDE'}</button>
    </div>

    {#if !collapsed}
      <div class="chips">
        {#if catalog}
          {#each catalog.items as c}
            <button
              type="button"
              class={"chip " + (chosen?.id === c.id ? 'active' : '')}
              on:click={() => pick(c.id)}>
              {c.title}
            </button>
          {/each}
        {/if}
      </div>

      {#if chosenFile}
        <div class="status">
          RENDERING: {chosenFile.name}
        </div>
      {:else if error}
        <div style="color: #ff6b6b; margin-top: 0.5rem;">{error}</div>
      {:else if chosen && !chosenFile}
        <div style="margin-top: 0.5rem; color: #ff6b6b;">
          No playable files for {chosen.title}
        </div>
      {/if}
    {/if}
  </div>
</div>