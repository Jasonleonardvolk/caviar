<script lang="ts">
  import plans from '$lib/../../static/plans.json';
  let busy = false;
  async function upgrade(planId: 'plus'|'pro') {
    busy = true;
    const res = await fetch('/api/billing/checkout', {
      method:'POST',
      headers: { 'content-type':'application/json' },
      body: JSON.stringify({ planId })
    });
    busy = false;
    if (!res.ok) { alert('Upgrade failed. Check configuration.'); return; }
    const { url } = await res.json();
    location.href = url;
  }
</script>

<h1>Plans & Pricing</h1>
<div class="grid">
  <div class="card">
    <h2>{plans.free.name}</h2>
    <ul>
      <li>{plans.free.maxDurationSec}s recording</li>
      <li>Watermark</li>
      <li>Export: {plans.free.export.join(', ')}</li>
    </ul>
  </div>
  <div class="card">
    <h2>{plans.plus.name}</h2>
    <ul>
      <li>{plans.plus.maxDurationSec}s recording</li>
      <li>No watermark</li>
      <li>Export: {plans.plus.export.join(', ')}</li>
    </ul>
    <button disabled={busy} on:click={() => upgrade('plus')}>Get Plus</button>
  </div>
  <div class="card">
    <h2>{plans.pro.name}</h2>
    <ul>
      <li>{plans.pro.maxDurationSec/60} min recording</li>
      <li>No watermark</li>
      <li>Export: {plans.pro.export.join(', ')}</li>
    </ul>
    <button disabled={busy} on:click={() => upgrade('pro')}>Get Pro</button>
  </div>
</div>

<style>
  .grid { display:grid; grid-template-columns: repeat(3, 1fr); gap:16px; }
  .card { border:1px solid #222; border-radius:12px; padding:16px; background:#0b0b0b; color:#fff; }
  button { margin-top:10px; padding:8px 14px; border-radius:10px; background:#10b981; color:#001; border:0; font-weight:700; cursor:pointer;}
</style>