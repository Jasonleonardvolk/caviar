<script lang="ts">
  let file: File | null = null;
  let mode: 'auto' | 'glb' | 'concept' = 'auto';
  let name = '';
  let description = '';
  let tags = 'hologram, ar, mesh';
  let layout = 'grid';
  let scale = '0.12';
  let result: any = null;
  let err = '';

  async function submit(e: Event) {
    e.preventDefault();
    err = ''; result = null;
    if (!file) { err = 'Choose a GLB or Concept-Mesh JSON.'; return; }
    const fd = new FormData();
    fd.set('file', file);
    if (name) fd.set('name', name);
    if (mode !== 'auto') fd.set('mode', mode);
    if (description) fd.set('description', description);
    if (tags) fd.set('tags', tags);
    if (layout) fd.set('layout', layout);
    if (scale) fd.set('scale', scale);
    const r = await fetch('/api/templates/upload', { method: 'POST', body: fd });
    if (!r.ok) { err = await r.text(); return; }
    result = await r.json();
  }
</script>

<style>
  .wrap { max-width: 860px; margin: 1rem auto; padding: 0 1rem; color:#fff;}
  .card { border:1px solid #2b2b2b; border-radius:16px; padding:1rem; background:#0d0d0d; }
  input, textarea, select, button { border:1px solid #444; background:#111; color:#fff; border-radius:10px; padding:.5rem .75rem; width:100%; }
  .row { display:grid; gap:.75rem; grid-template-columns: 1fr 1fr; }
  .row3 { display:grid; gap:.75rem; grid-template-columns: 1fr 1fr 1fr; }
  button { width:auto; }
  a { color:#9ec7ff; }
  a.btn { border:1px solid #444; border-radius:10px; padding:.5rem .9rem; background:#111; color:#fff; text-decoration:none; }
</style>

<div class="wrap">
  <h1>Upload Custom Template</h1>
  <form class="card" on:submit|preventDefault={submit} enctype="multipart/form-data">
    <label>File (.glb or Concept-Mesh .json)
      <input type="file" accept=".glb,.json" on:change={(e:any)=>file=e.currentTarget.files?.[0] ?? null} />
    </label>

    <div class="row">
      <label>Mode
        <select bind:value={mode}>
          <option value="auto">Auto (by extension)</option>
          <option value="glb">GLB (store as-is)</option>
          <option value="concept">Concept JSON → GLB</option>
        </select>
      </label>

      <label>Output name (optional, no ext)
        <input placeholder="concept_grid_v1" bind:value={name} />
      </label>
    </div>

    <div class="row3" style="margin-top:.5rem">
      <label>Layout (JSON → GLB)
        <select bind:value={layout}>
          <option value="grid">grid</option>
          <option value="xyz">xyz (use pos fields)</option>
        </select>
      </label>

      <label>Scale (JSON → GLB)
        <input bind:value={scale} />
      </label>

      <div></div>
    </div>

    <label style="margin-top:.5rem">Description
      <textarea rows="3" bind:value={description} placeholder="Short description for catalog"></textarea>
    </label>
    <label>Tags (comma-separated)
      <input bind:value={tags} />
    </label>

    <div style="display:flex; gap:.75rem; align-items:center; margin-top:.75rem">
      <button type="submit">Upload</button>
      <a href="/templates">Back to Catalog</a>
    </div>

    {#if err}<p style="color:#ff6b6b; margin-top:.5rem">{err}</p>{/if}
    {#if result}
      <div style="margin-top:.75rem">
        <div>Saved GLB: {result.glb}</div>
        <div>Metadata: {result.meta}</div>
        <div style="margin-top:.25rem"><a href={"/api/templates/file/"+encodeURIComponent((result.glb as string).split('\\').pop())}>Stream / download GLB</a></div>
      </div>
    {/if}
  </form>
</div>