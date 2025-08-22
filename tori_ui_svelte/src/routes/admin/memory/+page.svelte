<script lang="ts">
  import { onMount } from "svelte";
  let userId = "default";
  let loading = true;
  let error: string | null = null;

  let energy: number | null = null;
  let coherence: number | null = null;
  let laplacianVersion = "";
  let size = 0;

  async function loadState() {
    loading = true; error = null;
    try {
      const res = await fetch(`/api/memory/state/${encodeURIComponent(userId)}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      energy = data.energy;
      coherence = data.coherence;
      laplacianVersion = data.laplacian_version;
      size = data.size;
    } catch (e:any) {
      error = e.message || String(e);
    } finally {
      loading = false;
    }
  }

  onMount(loadState);
</script>

<div class="p-6 max-w-3xl mx-auto space-y-6">
  <h1 class="text-2xl font-bold">Memory Invariants</h1>

  <div class="flex items-end gap-3">
    <label class="text-sm">User ID</label>
    <input bind:value={userId} class="border rounded-lg px-3 py-1" placeholder="default" />
    <button on:click={loadState} class="px-3 py-1 rounded-xl shadow bg-black text-white">Refresh</button>
  </div>

  {#if loading}
    <div class="animate-pulse text-sm opacity-70">Loading invariants...</div>
  {:else if error}
    <div class="text-red-600 text-sm">Error: {error}</div>
  {:else}
    <div class="grid grid-cols-2 gap-4">
      <div class="rounded-2xl shadow p-4">
        <div class="text-xs uppercase opacity-60">Coherence</div>
        <div class="text-3xl font-semibold">{coherence?.toFixed(4)}</div>
        <div class={"mt-2 text-xs " + (coherence! >= 0.95 ? "text-green-700" : "text-amber-600")}>
          {coherence! >= 0.95 ? "Synchronized" : "Partial lock"}
        </div>
      </div>
      <div class="rounded-2xl shadow p-4">
        <div class="text-xs uppercase opacity-60">Energy</div>
        <div class="text-3xl font-semibold">{energy?.toExponential(4)}</div>
        <div class="mt-2 text-xs opacity-70">Lower is better (bounded)</div>
      </div>
    </div>

    <div class="rounded-2xl shadow p-4">
      <div class="text-xs uppercase opacity-60">Laplacian</div>
      <div class="font-mono text-sm">{laplacianVersion}</div>
      <div class="text-xs opacity-70 mt-1">Nodes: {size}</div>
    </div>
  {/if}
</div>
