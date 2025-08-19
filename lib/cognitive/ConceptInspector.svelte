<script lang="ts">
  // src/lib/components/ConceptInspector.svelte
  import { conceptList } from '$lib/stores/conceptData';  // store that holds ConceptTuple[] from concept extraction
  import type { ConceptTuple } from '$lib/cognitive/conceptScoring';
  let concepts: ConceptTuple[] = [];
  $: concepts = $conceptList;
  
  // Filter controls
  let confidenceThreshold: number = 0;
  let selectedDoc: string = "";
  let selectedCluster: string = "";
  let showGraph: boolean = false;
  
  // Unique document and cluster IDs for filter dropdowns
  $: docIds = Array.from(new Set(concepts.flatMap(c => c.originDocs.map(o => o.docId))));
  $: clusterIds = Array.from(new Set(concepts.map(c => c.clusterId)));
  
  // Filtered concept list based on current controls
  $: filteredConcepts = concepts.filter(c =>
    c.score >= confidenceThreshold &&
    (selectedDoc ? c.originDocs.some(o => o.docId === selectedDoc) : true) &&
    (selectedCluster ? c.clusterId === parseInt(selectedCluster) : true)
  );
  
  // If a cluster is selected, prepare its synonyms for tag cloud display
  $: currentClusterSynonyms = [];
  $: {
    if (selectedCluster) {
      const clusterIdNum = parseInt(selectedCluster);
      const clusterConcept = concepts.find(c => c.clusterId === clusterIdNum);
      currentClusterSynonyms = clusterConcept ? clusterConcept.mergedFrom : [];
    }
  }
</script>

<div class="controls">
  <div>
    <label>Confidence ≥ {confidenceThreshold.toFixed(2)}</label>
    <input type="range" min="0" max="1" step="0.01" bind:value={confidenceThreshold} />
  </div>
  <div>
    <label>Document:</label>
    <select bind:value={selectedDoc}>
      <option value="">(All)</option>
      {#each docIds as doc}
        <option value={doc}>{doc}</option>
      {/each}
    </select>
  </div>
  <div>
    <label>Cluster:</label>
    <select bind:value={selectedCluster}>
      <option value="">(All)</option>
      {#each clusterIds as cid}
        <option value={cid}>{cid}</option>
      {/each}
    </select>
  </div>
  <button on:click={() => showGraph = !showGraph}>
    {#if showGraph}Show List{/if}{#if !showGraph}Show Graph{/if}
  </button>
</div>

{#if !showGraph}
  <!-- List view -->
  <ul class="concept-list">
    {#each filteredConcepts as concept}
      <li>
        <strong>{concept.name}</strong>
        (Score: {concept.score.toFixed(2)}, Cluster {concept.clusterId})
        {#if concept.originDocs.length > 0}
          – Appears in: {concept.originDocs.map(o => o.docId).join(', ')}
        {/if}
      </li>
    {/each}
    {#if filteredConcepts.length === 0}
      <li><em>No concepts match the current filters.</em></li>
    {/if}
  </ul>
  {#if selectedCluster && currentClusterSynonyms.length > 0}
    <!-- Tag cloud of cluster synonyms -->
    <div class="tag-cloud">
      {#each currentClusterSynonyms as term}
        <span style="font-size: {1 + ((concepts.find(c => c.name === term)?.frequency || 0) / 10)}em;">
          {term}
        </span>
      {/each}
    </div>
  {/if}
{:else}
  <!-- Graph view (for concept connectivity) -->
  <div id="conceptGraph">
    <!-- Use a force-directed graph (e.g., D3.js) to visualize concepts and their relations -->
    <p><em>Semantic graph visualization would be rendered here (nodes = concepts, edges = co-occurrence links).</em></p>
  </div>
{/if}
