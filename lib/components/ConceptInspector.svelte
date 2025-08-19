<script lang="ts">
  import { conceptMesh } from '$lib/stores/conceptMesh';
  import type { ConceptDiff } from '$lib/stores/types';
  import { derived } from 'svelte/store';
  
  let minScore = 0;
  let selectedCluster = '';
  let selectedSource = '';
  let showLineage = true;
  
  const concepts = derived(conceptMesh, ($conceptMesh) =>
    $conceptMesh.flatMap((diff: ConceptDiff) => {
      return diff.concepts.map((concept) => ({
        ...concept,
        source: diff.title || 'Unknown',
        clusterId: concept.clusterId ?? 'unclustered',
      }));
    })
  );
  
  // 🧠 Reactive logging for concept store changes
  $: {
    console.log("🧠 Total Concepts in Store:", $concepts.length);
    console.log("🧠 Concept Names:", $concepts.map(c => c.name));
  }
  
  $: uniqueClusters = Array.from(new Set($concepts.map((c) => c.clusterId)));
  $: uniqueSources = Array.from(new Set($concepts.map((c) => c.source)));
  $: filtered = $concepts.filter((c) => {
    return (
      (!selectedCluster || c.clusterId === selectedCluster) &&
      (!selectedSource || c.source === selectedSource) &&
      (c.score ?? 1) >= minScore
    );
  });
</script>

<div class="p-4 space-y-4">
  <div class="flex flex-wrap gap-4">
    <div>
      <label class="block text-sm font-medium">Min Score: {minScore.toFixed(2)}</label>
      <input type="range" min="0" max="1" step="0.01" bind:value={minScore} class="w-48" />
    </div>
    <div>
      <label class="block text-sm font-medium">Cluster</label>
      <select bind:value={selectedCluster} class="border rounded px-2 py-1">
        <option value="">All</option>
        {#each uniqueClusters as cid}
          <option value={cid}>{cid}</option>
        {/each}
      </select>
    </div>
    <div>
      <label class="block text-sm font-medium">Source</label>
      <select bind:value={selectedSource} class="border rounded px-2 py-1">
        <option value="">All</option>
        {#each uniqueSources as src}
          <option value={src}>{src}</option>
        {/each}
      </select>
    </div>
    <div>
      <label class="inline-flex items-center gap-2">
        <input type="checkbox" bind:checked={showLineage} />
        Show Lineage
      </label>
    </div>
  </div>
  
  <div class="border rounded p-3 max-h-[600px] overflow-auto">
    {#each filtered as c}
      <div class="mb-4">
        <div class="font-semibold text-lg">{c.name}</div>
        <div class="text-sm text-gray-600">
          Score: {(c.score ?? 1).toFixed(2)} | 
          Cluster: {c.clusterId} | 
          Source: {c.source}
        </div>
        {#if showLineage && c.mergedFrom?.length > 0}
          <div class="text-xs mt-1 text-gray-500">
            Merged from: {c.mergedFrom.join(', ')}
          </div>
        {/if}
        {#if showLineage && c.originDocs?.length > 0}
          <div class="text-xs text-gray-500 mt-1">
            Docs: {c.originDocs.map((o) => `${o.docId} (${o.occurrences})`).join(', ')}
          </div>
        {/if}
      </div>
    {/each}
    {#if filtered.length === 0}
      <p class="text-gray-500 italic">No concepts match current filters.</p>
    {/if}
  </div>
  
  <!-- Stats Panel -->
  <div class="text-sm text-gray-600 border-t pt-2">
    <div class="flex justify-between">
      <span>Total Concepts: {$concepts.length}</span>
      <span>Filtered: {filtered.length}</span>
      <span>Sources: {uniqueSources.length}</span>
      <span>Clusters: {uniqueClusters.length}</span>
    </div>
  </div>
</div>

<style>
  select, input[type="range"] {
    outline: none;
  }
  
  select:focus, input[type="range"]:focus {
    ring: 2px;
    ring-color: rgb(59 130 246);
  }
  
  .border {
    border-color: rgb(209 213 219);
  }
  
  .border:hover {
    border-color: rgb(156 163 175);
  }
</style>
