<script lang="ts">
  let customerId = (typeof localStorage !== 'undefined' && localStorage.getItem('stripe.customerId')) || '';
  let err = ''; let busy = false;

  function save() {
    try { localStorage.setItem('stripe.customerId', customerId) } catch {}
  }

  async function openPortal() {
    err = ''; busy = true;
    try {
      const r = await fetch('/api/billing/portal', {
        method: 'POST', headers: {'content-type': 'application/json'},
        body: JSON.stringify({ customerId })
      });
      if (!r.ok) throw new Error(await r.text());
      const j = await r.json();
      if (j.url) location.href = j.url;
    } catch (e:any) {
      err = String(e?.message || e);
    } finally {
      busy = false;
    }
  }
</script>

<style>
  .panel { max-width: 560px; margin: 2rem auto; border:1px solid #2b2b2b; border-radius:16px; padding:1rem; background:#0d0d0d; color:#fff; }
  input { width:100%; padding:.6rem .8rem; border-radius:10px; border:1px solid #444; background:#111; color:#fff; }
  button { margin-top:.75rem; border-radius:10px; border:1px solid #444; padding:.6rem .9rem; background:#111; color:#fff; }
</style>

<div class="panel">
  <h2>Manage Subscription</h2>
  <p>Enter your Stripe <em>Customer ID</em> (for now) to open the self-serve portal.</p>
  <input bind:value={customerId} on:change={save} placeholder="cus_123..." />
  <button on:click={openPortal} disabled={!customerId || busy}>Open Billing Portal</button>
  {#if err}<p style="color:#ff6b6b">{err}</p>{/if}
</div>