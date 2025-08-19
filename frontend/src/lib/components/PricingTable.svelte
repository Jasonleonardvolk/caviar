<script lang="ts">
  let plans: any[] = [];
  let err = '';

  async function load() {
    try {
      const r = await fetch('/config/plans.json', { cache: 'no-store' });
      const j = await r.json();
      plans = j.plans;
    } catch (e) { err = String(e); }
  }
  load();

  async function checkout(planId: string) {
    const r = await fetch('/api/billing/checkout', {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ planId })
    });
    const j = await r.json();
    if (j.url) location.href = j.url;
  }
</script>

<style>
  .grid { display:grid; gap:1rem; grid-template-columns: repeat(auto-fit, minmax(240px,1fr)); }
  .card { border:1px solid #2a2a2a; border-radius:16px; padding:1rem; background:#0d0d0d; color:#fff; }
  .title { font-size:1.1rem; font-weight:700; }
  .price { font-family: ui-monospace,monospace; }
  button { margin-top:.5rem; border:1px solid #444; border-radius:10px; padding:.5rem .9rem; background:#111; color:#fff; }
</style>

{#if err}<div>Failed to load plans: {err}</div>{/if}
<div class="grid">
  {#each plans as p}
    <div class="card">
      <div class="title">{p.name}</div>
      <div class="price">${p.priceMonthly}/mo</div>
      <ul>{#each p.features as f}<li>• {f}</li>{/each}</ul>
      {#if p.stripePriceId}
        <button on:click={() => checkout(p.id)}>Upgrade</button>
      {:else}
        <button disabled>Current</button>
      {/if}
    </div>
  {/each}
</div>