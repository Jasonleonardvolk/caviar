<script lang="ts">
  export let data: any;
  let { env, artifacts, guides, readiness } = data;
  function fmtBytes(n: number) { const u=['B','KB','MB','GB']; let i=0; while(n>=1024&&i<u.length-1){n/=1024;i++;} return `${n.toFixed(1)} ${u[i]}`; }
</script>

<style>
  .wrap { max-width: 1080px; margin: 1rem auto; padding: 0 1rem; color:#fff; }
  .grid { display:grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
  .card { border:1px solid #2b2b2b; border-radius:16px; padding:1rem; background:#0d0d0d; }
  .ok { color:#7CFFA1 }
  .bad { color:#ff6b6b }
  code { word-break: break-all; }
  .list { max-height: 300px; overflow:auto; border:1px solid #222; border-radius:12px; padding:.5rem; }
  a.btn { border:1px solid #444; border-radius:10px; padding:.5rem .9rem; background:#111; color:#fff; text-decoration:none; }
</style>

<div class="wrap">
  <h1>Publish Checklist (Snap / TikTok)</h1>

  <div class="grid">
    <div class="card">
      <h2>Environment</h2>
      <div>
        <div>Stripe key: {env.stripeOk ? '<span class="ok">present</span>' : '<span class="bad">missing</span>'} <code>{env.envFile}</code></div>
        <div>Plans synced: {env.plansOk ? '<span class="ok">yes</span>' : '<span class="bad">no</span>'} <code>{env.plansSrc}</code> ⇄ <code>{env.plansDst}</code></div>
      </div>
      <div style="margin-top:.5rem">
        <a class="btn" href="/templates">Templates</a>
        <a class="btn" href="/templates/upload">Upload</a>
      </div>
    </div>

    <div class="card">
      <h2>Artifacts</h2>
      <div>Exports dir: <code>{artifacts.exportsDir}</code></div>
      <h3 style="margin-bottom:.25rem">GLB Templates</h3>
      <div class="list">
        {#if artifacts.glbs.length === 0}<div>None found</div>{/if}
        {#each artifacts.glbs as g}
          <div>• <a href={"/api/templates/file/"+encodeURIComponent(g.name)} target="_blank">{g.name}</a> — {fmtBytes(g.size)} — {g.mtime}</div>
        {/each}
      </div>

      <h3 style="margin:.75rem 0 .25rem">Textures (.ktx2)</h3>
      <div class="list">
        {#if artifacts.textures.length === 0}<div>None found</div>{/if}
        {#each artifacts.textures as t}
          <div>• {t.name} — {fmtBytes(t.size)} — {t.mtime}</div>
        {/each}
      </div>
    </div>

    <div class="card">
      <h2>Snap Guides</h2>
      {#if data.guides.snap.length === 0}
        <div>No guides in D:\Dev\kha\integrations\snap\guides</div>
      {:else}
        {#each data.guides.snap as g}
          <details style="margin-bottom:.5rem">
            <summary>{g.name} ({g.path})</summary>
            <pre style="white-space:pre-wrap">{g.content}</pre>
          </details>
        {/each}
      {/if}
    </div>

    <div class="card">
      <h2>TikTok Guides</h2>
      {#if data.guides.tiktok.length === 0}
        <div>No guides in D:\Dev\kha\integrations\tiktok\guides</div>
      {:else}
        {#each data.guides.tiktok as g}
          <details style="margin-bottom:.5rem">
            <summary>{g.name} ({g.path})</summary>
            <pre style="white-space:pre-wrap">{g.content}</pre>
          </details>
        {/each}
      {/if}
    </div>
  </div>

  <div class="card" style="margin-top:1rem">
    <h2>Go/No-Go</h2>
    <div>
      <div>Snap readiness: {data.readiness.snap ? '<span class="ok">READY</span>' : '<span class="bad">NOT READY</span>'}</div>
      <div>TikTok readiness: {data.readiness.tiktok ? '<span class="ok">READY</span>' : '<span class="bad">NOT READY</span>'}</div>
    </div>
    <p style="opacity:.8">READY == GLB present, KTX2 textures present, and at least one guide found.</p>
  </div>
</div>