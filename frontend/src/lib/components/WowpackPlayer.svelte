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

  const prefer = ['video/mp4', 'video/webm', 'video/mp4', 'video/quicktime', 'video/x-matroska'];

  function bestPlayable(files: FileItem[]): FileItem | null {
    if (!files?.length) return null;
    const v = document.createElement('video');
    // test by contentType preference
    for (const mime of prefer) {
      const candidate = files.find(f => f.contentType === mime);
      if (!candidate) continue;
      const ok = v.canPlayType(mime);
      if (ok === 'probably' || ok === 'maybe') return candidate;
    }
    // fallback to first
    return files[0] || null;
  }

  async function load() {
    error = '';
    try {
      const r = await fetch('/api/wowpack/list', { cache: 'no-store' });
      catalog = await r.json();
      // auto-pick first clip with a playable file
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

  onMount(load);
</script>

<style>
  .wrap { border:1px solid #2b2b2b; border-radius:16px; padding:.75rem; background:#0d0d0d; color:#fff; margin-top: 1rem; }
  .row { display:flex; gap:.5rem; flex-wrap:wrap; align-items:center; }
  .chip { border:1px solid #444; border-radius:999px; padding:.25rem .6rem; background:#111; cursor:pointer; transition: all 0.2s; }
  .chip:hover { border-color:#6ea5ff; }
  .chip.active { border-color:#8ec5ff; color:#8ec5ff; background:#1a2a3a; }
  video { width:100%; max-width:540px; border-radius:16px; background:#000; display:block; }
  small.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; opacity:.75; }
  .open-btn { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; text-decoration: none; }
  .open-btn:hover { transform: scale(1.05); }
</style>

<div class="wrap">
  <div class="row" style="justify-content:space-between">
    <div><b>🎬 WOW Pack Demo Loops</b> <small class="mono">D:\Dev\kha\content\wowpack\output</small></div>
    {#if chosenFile}
      <a class="chip open-btn" href={"/api/wowpack/file/" + encodeURIComponent(chosenFile.name)} target="_blank">📥 Open file</a>
    {/if}
  </div>

  <div class="row" style="margin-top:.5rem">
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
      <div style="margin-top:.75rem; padding: 1rem; background: #1a1a1a; border-radius: 8px;">
        No web-playable encodes found for <b>{chosen.title}</b> yet.
        {#if chosen.mastersPresent}
          <div style="margin-top: 0.5rem; color: #8ec5ff;">
            ✅ Masters present in <code style="background: #2a2a2a; padding: 2px 4px; border-radius: 3px;">D:\Dev\kha\content\wowpack\input\{chosen.id}.mov</code>
          </div>
          <div style="margin-top: 0.5rem;">Run encoder to generate web-playable versions.</div>
        {:else}
          <div style="margin-top: 0.5rem; color: #ff6b6b;">
            ❌ Add ProRes master at <code style="background: #2a2a2a; padding: 2px 4px; border-radius: 3px;">D:\Dev\kha\content\wowpack\input\{chosen.id}.mov</code> and encode.
          </div>
        {/if}
      </div>
    {:else if chosenFile}
      <div style="margin-top:.75rem">
        <video bind:this={videoEl} src={"/api/wowpack/file/" + encodeURIComponent(chosenFile.name)} muted autoplay loop playsinline controls></video>
        <div class="row" style="margin-top:.25rem">
          <small class="mono">📹 Playing: {chosenFile.name} • {chosenFile.contentType} • {(chosenFile.size / 1024 / 1024).toFixed(1)} MB</small>
        </div>
      </div>
    {/if}
  {/if}
</div>