<script lang="ts">
  import { conceptMesh } from '$lib/stores/conceptMesh';
  import { onMount } from 'svelte';

  let showUnused = false;
  let fullConcepts = [];
  let grouped = new Map();

  function groupByCluster(concepts) {
    const map = new Map();
    for (const concept of concepts) {
      const id = concept.clusterId ?? concept.cluster_id ?? 'unclustered';
      if (!map.has(id)) map.set(id, []);
      map.get(id).push(concept);
    }
    return map;
  }

  function exportAsJSON() {
    const blob = new Blob([JSON.stringify(fullConcepts, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `concepts_dump_${new Date().toISOString()}.json`;
    link.click();
    URL.revokeObjectURL(url);
  }

  $: {
    const all = $conceptMesh.flatMap(diff => diff.concepts || []);
    fullConcepts = all.map(c => {
      const name = c.name ?? c.concept ?? "undefined";
      const score = c.score ?? c.final_score ?? c.confidence ?? 0;
      const method = c.method ?? c.extraction_method ?? "unknown";
      const cluster = c.clusterId ?? c.cluster_id ?? '–';
      const mergedFrom = (Array.isArray(c.mergedFrom) ? c.mergedFrom.join(', ') : undefined) ?? undefined;
      const from = Array.isArray(c.originDocs)
        ? c.originDocs.map(o => o.docId).join(", ")
        : c.source?.docId ?? "unknown";
      const context = c.context ?? "";

      return { ...c, name, score, method, cluster, mergedFrom, from, context };
    });

    grouped = groupByCluster(fullConcepts);
  }
</script>

<style>
  .cluster {
    border: 2px solid #666;
    padding: 1em;
    margin-bottom: 1.2em;
    border-radius: 6px;
    background: #0f0f0f;
  }

  .concept {
    margin-bottom: 0.75em;
    padding-left: 0.5em;
    border-left: 3px solid #4af;
  }

  .low-score {
    opacity: 0.5;
    font-style: italic;
  }

  .meta {
    font-size: 0.85em;
    color: #bbb;
  }

  .tag {
    font-size: 0.75em;
    color: #ccc;
    background: #222;
    border-radius: 4px;
    padding: 2px 6px;
    margin-left: 4px;
  }

  .controls {
    margin-bottom: 1em;
    display: flex;
    gap: 1em;
    align-items: center;
  }

  button {
    background: #111;
    color: #f0f0f0;
    border: 1px solid #555;
    padding: 4px 12px;
    cursor: pointer;
    border-radius: 4px;
  }

  button:hover {
    background: #222;
  }

  label {
    font-size: 0.9em;
    color: #ccc;
  }

  .cluster-heading {
    font-weight: bold;
    font-size: 1.1em;
    margin-bottom: 0.5em;
    color: #9fd;
  }
</style>

<div class="controls">
  <label><input type="checkbox" bind:checked={showUnused} /> Show unused concepts (score &lt; 0.5)</label>
  <button on:click={exportAsJSON}>📥 Download as JSON</button>
</div>

{#each grouped as concepts, clusterId}
  <div class="cluster">
    <div class="cluster-heading">Cluster {clusterId}</div>
    {#each concepts.filter(c => showUnused || c.score >= 0.5) as concept}
      <div class="concept {concept.score < 0.5 ? 'low-score' : ''}">
        <strong>{concept.name}</strong>
        <div class="meta">
          Score: {concept.score.toFixed(3)} | Source: {concept.from}
          {#if concept.mergedFrom}
            <span class="tag">Synonyms: {concept.mergedFrom}</span>
          {/if}
          {#if concept.method}
            <span class="tag">{concept.method}</span>
          {/if}
        </div>
        {#if concept.context}
          <div class="meta">Context: {concept.context.slice(0, 140)}...</div>
        {/if}
      </div>
    {/each}
  </div>
{/each}
