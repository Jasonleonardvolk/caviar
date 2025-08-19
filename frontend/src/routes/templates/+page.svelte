<script lang="ts">
  export let data: { items: any[] };
  let q = '';
  let tag = '';

  function fmtBytes(n: number) {
    const u = ['B','KB','MB','GB']; let i = 0; while (n >= 1024 && i < u.length-1) { n/=1024; i++; }
    return `${n.toFixed(1)} ${u[i]}`;
  }

  $: filtered = data.items.filter((x) => {
    const nameMatch = x.name.toLowerCase().includes(q.toLowerCase());
    const metaTags = (x.meta?.tags ?? []) as string[];
    const tagMatch = tag ? metaTags.map(t => String(t).toLowerCase()).includes(tag.toLowerCase()) : true;
    return nameMatch && tagMatch;
  });
</script>

<style>
  .toolbar { display:flex; gap: .75rem; margin-bottom: 1rem; align-items: center; }
  .grid { display:grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 1rem; }
  .card { border:1px solid #2b2b2b; border-radius: 16px; padding: 1rem; background:#0d0d0d; color:#fff; }
  input, select, button { border:1px solid #444; background:#111; color:#fff; border-radius:10px; padding:.5rem .75rem; }
  a.btn { border:1px solid #444; border-radius:10px; padding:.5rem .9rem; background:#111; color:#fff; text-decoration:none; }
</style>

<h1>Template Catalog</h1>
<div class="toolbar">
  <input placeholder="Search by name…" bind:value={q} />
  <select bind:value={tag}>
    <option value=''>All tags</option>
    <option value='hologram'>hologram</option>
    <option value='ar'>ar</option>
    <option value='mesh'>mesh</option>
  </select>

  <!-- One-click export: posts to API to generate a new bundle from default concept mesh -->
  <form method="POST" action="/api/templates/export?zip=1">
    <input type="hidden" name="input" value="D:\Dev\kha\data\concept_graph.json" />
    <input type="hidden" name="layout" value="grid" />
    <input type="hidden" name="scale" value="0.12" />
    <button type="submit">↻ Export New Bundle (ZIP)</button>
  </form>
  
  <a class="btn" href="/templates/upload" style="margin-left:.5rem">Upload</a>
  <a class="btn" href="/publish" style="margin-left:.5rem">Publish Checklist</a>
</div>

<div class="grid">
  {#each filtered as x}
    <div class="card">
      <div style="font-weight:700">{x.name}</div>
      <div style="opacity:.8">Size: {fmtBytes(x.size)} • Updated: {x.mtime}</div>
      {#if x.meta?.description}<div style="margin-top:.5rem">{x.meta.description}</div>{/if}
      {#if x.meta?.tags?.length}
        <div style="margin-top:.5rem; opacity:.8">Tags: {(x.meta.tags as string[]).join(', ')}</div>
      {/if}
      <!-- note: GLB lives under exports; we serve bundles via the export API -->
    </div>
  {/each}
</div>