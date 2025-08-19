<script lang="ts">
  export let data: any;
  const r = data.report;
  function badge(ok: boolean) { return ok ? 'badge ok' : 'badge bad'; }
  function fmt(n: number) { return n.toLocaleString(); }
</script>

<style>
  .wrap { max-width: 1080px; margin: 1rem auto; padding: 0 1rem; color:#fff; }
  .grid { display:grid; grid-template-columns: 1fr 1fr; gap:1rem; }
  .card { border:1px solid #2b2b2b; border-radius:16px; padding:1rem; background:#0d0d0d; }
  .headline { display:flex; align-items:center; gap:.5rem; }
  .badge { padding:.25rem .6rem; border-radius:999px; border:1px solid #333; }
  .ok { background:#102b17; border-color:#2a9d4b; color:#7CFFA1; }
  .bad { background:#2b1010; border-color:#9d2a2a; color:#ff6b6b; }
  pre { white-space:pre-wrap; word-break:break-word; }
  code { word-break:break-all; }
</style>

<div class="wrap">
  <div class="headline">
    <h1 style="margin:0">System Health</h1>
    <span class={badge(r.ok)}>{r.ok ? 'OK' : 'CHECK'}</span>
    <span class="badge">{r.timestamp}</span>
  </div>

  <div class="grid">
    <div class="card">
      <h2>Environment</h2>
      <div>Node: <code>{r.env.node}</code> • {r.env.platform}/{r.env.arch}</div>
      {#if r.env.frontendPackageVersion}<div>frontend\package.json version: <code>{r.env.frontendPackageVersion}</code></div>{/if}
      {#if r.env.projectPackageVersion}<div>project\package.json version: <code>{r.env.projectPackageVersion}</code></div>{/if}
    </div>

    <div class="card">
      <h2>Plans</h2>
      <div>Synced: <span class={badge(r.plans.synced)}>{r.plans.synced ? 'yes' : 'no'}</span></div>
      <div><code>{r.plans.src}</code> ⇄ <code>{r.plans.dst}</code></div>
    </div>

    <div class="card">
      <h2>Monetization</h2>
      <div>Stripe key: <span class={badge(r.monetization.stripeKeyPresent)}>{r.monetization.stripeKeyPresent ? 'present' : 'missing'}</span></div>
      <div>Pricing route: <span class={badge(r.monetization.pricingRouteExists)}>{r.monetization.pricingRouteExists ? 'exists' : 'missing'}</span></div>
      <div>Billing endpoints: <span class={badge(r.monetization.billingEndpointsExist)}>{r.monetization.billingEndpointsExist ? 'present' : 'missing'}</span></div>
    </div>

    <div class="card">
      <h2>Hologram & Templates</h2>
      <div>/hologram: <span class={badge(r.hologram.routeExists)}>{r.hologram.routeExists ? 'present' : 'missing'}</span></div>
      <div>Recorder: <span class={badge(r.hologram.recorderExists)}>{r.hologram.recorderExists ? 'present' : 'missing'}</span></div>
      <div>Exporters: <span class={badge(r.exporters.glbExporterExists && r.exporters.ktx2ScriptExists)}>{(r.exporters.glbExporterExists && r.exporters.ktx2ScriptExists) ? 'ready' : 'missing'}</span></div>
      <div>Catalog route: <span class={badge(r.templates.routes.catalog)}>{r.templates.routes.catalog ? 'present' : 'missing'}</span></div>
      <div>Upload route: <span class={badge(r.templates.routes.upload)}>{r.templates.routes.upload ? 'present' : 'missing'}</span></div>
    </div>

    <div class="card">
      <h2>Artifacts</h2>
      <div>Templates dir: <code>{r.artifacts.templatesDir}</code></div>
      <div>Textures dir: <code>{r.artifacts.texturesDir}</code></div>
      <div>GLB count: <b>{fmt(r.templates.counts.glb)}</b> • KTX2 count: <b>{fmt(r.templates.counts.ktx2)}</b></div>
      {#if r.templates.counts.glb === 0 || r.templates.counts.ktx2 === 0}
        <div class="bad badge" style="margin-top:.5rem">Publish not ready – missing GLB or KTX2</div>
      {/if}
    </div>

    <div class="card">
      <h2>Guides</h2>
      <div>Snap guides: <b>{fmt(r.guides.snap)}</b> • TikTok guides: <b>{fmt(r.guides.tiktok)}</b></div>
      <div>Ready for Snap: <span class={badge(r.readiness.publishSnap)}>{r.readiness.publishSnap ? 'READY' : 'NO'}</span></div>
      <div>Ready for TikTok: <span class={badge(r.readiness.publishTikTok)}>{r.readiness.publishTikTok ? 'READY' : 'NO'}</span></div>
    </div>

    <div class="card">
      <h2>Files</h2>
      <div>Required files present: {fmt(r.files.presentCount)} / {fmt(r.files.requiredTotal)}</div>
      {#if r.files.missing.length}
        <details style="margin-top:.5rem">
          <summary>Show missing</summary>
          <pre>{r.files.missing.join('\n')}</pre>
        </details>
      {/if}
    </div>

    <div class="card">
      <h2>Advice</h2>
      {#if r.advice.length === 0}
        <div class="ok badge">All green. Good to go.</div>
      {:else}
        <ul>{#each r.advice as a}<li>{a}</li>{/each}</ul>
      {/if}
    </div>
  </div>
</div>