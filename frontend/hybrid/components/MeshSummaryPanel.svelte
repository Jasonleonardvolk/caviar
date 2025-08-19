<script lang="ts">
  export let summary = {};
  
  $: nodeCount = summary.node_count || summary.nodes?.length || 0;
  $: edgeCount = summary.edge_count || summary.edges?.length || 0;
  $: version = summary.version || 'unknown';
  $: lastUpdated = summary.last_updated ? 
    new Date(summary.last_updated * 1000).toLocaleTimeString() : 
    'never';
</script>

<div class="mesh-summary-panel">
  <h3>Mesh Context</h3>
  
  <div class="mesh-stats">
    <div class="stat">
      <span class="stat-label">Nodes</span>
      <span class="stat-value">{nodeCount}</span>
    </div>
    
    <div class="stat">
      <span class="stat-label">Edges</span>
      <span class="stat-value">{edgeCount}</span>
    </div>
    
    <div class="stat">
      <span class="stat-label">Version</span>
      <span class="stat-value">{version}</span>
    </div>
    
    <div class="stat">
      <span class="stat-label">Updated</span>
      <span class="stat-value">{lastUpdated}</span>
    </div>
  </div>
  
  {#if summary.keys && summary.keys.length > 0}
    <div class="mesh-keys">
      <h4>Active Keys</h4>
      <ul>
        {#each summary.keys.slice(0, 5) as key}
          <li>{key}</li>
        {/each}
        {#if summary.keys.length > 5}
          <li class="more">+{summary.keys.length - 5} more</li>
        {/if}
      </ul>
    </div>
  {/if}
</div>

<style>
  .mesh-summary-panel {
    background: #1e293b;
    border-radius: 8px;
    padding: 1rem;
    margin-top: 1rem;
  }
  
  h3 {
    margin: 0 0 1rem;
    font-size: 1rem;
    color: #e2e8f0;
  }
  
  h4 {
    margin: 1rem 0 0.5rem;
    font-size: 0.875rem;
    color: #cbd5e1;
  }
  
  .mesh-stats {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.75rem;
  }
  
  .stat {
    display: flex;
    flex-direction: column;
    padding: 0.5rem;
    background: #0f172a;
    border-radius: 4px;
    border: 1px solid #334155;
  }
  
  .stat-label {
    font-size: 0.75rem;
    color: #64748b;
    margin-bottom: 0.25rem;
  }
  
  .stat-value {
    font-size: 1.25rem;
    font-weight: 600;
    color: #3b82f6;
  }
  
  .mesh-keys ul {
    list-style: none;
    padding: 0;
    margin: 0;
  }
  
  .mesh-keys li {
    padding: 0.25rem 0.5rem;
    margin-bottom: 0.25rem;
    background: #0f172a;
    border-radius: 4px;
    font-size: 0.875rem;
    color: #94a3b8;
    font-family: monospace;
  }
  
  .mesh-keys li.more {
    color: #64748b;
    font-style: italic;
    font-family: inherit;
  }
</style>
