<script lang="ts">
  export let logs = [];
  
  let autoScroll = true;
  let logContainer;
  
  $: if (autoScroll && logContainer) {
    setTimeout(() => {
      logContainer.scrollTop = logContainer.scrollHeight;
    }, 0);
  }
  
  function getLevelClass(level: string) {
    switch(level?.toLowerCase()) {
      case 'error': return 'log-error';
      case 'warning':
      case 'warn': return 'log-warning';
      case 'info': return 'log-info';
      case 'debug': return 'log-debug';
      default: return 'log-info';
    }
  }
  
  function formatTimestamp(timestamp: string) {
    try {
      return new Date(timestamp).toLocaleTimeString();
    } catch {
      return timestamp || '';
    }
  }
  
  function clearLogs() {
    logs = [];
  }
</script>

<div class="log-panel">
  <div class="log-header">
    <h3>Event Log</h3>
    <div class="log-controls">
      <label>
        <input 
          type="checkbox" 
          bind:checked={autoScroll}
        />
        Auto-scroll
      </label>
      <button on:click={clearLogs}>Clear</button>
    </div>
  </div>
  
  <div class="log-container" bind:this={logContainer}>
    {#if logs.length === 0}
      <div class="no-logs">No events yet...</div>
    {:else}
      {#each logs as log}
        <div class="log-entry {getLevelClass(log.level)}">
          <span class="log-time">{formatTimestamp(log.timestamp)}</span>
          <span class="log-level">[{log.level || 'INFO'}]</span>
          <span class="log-message">{log.message}</span>
        </div>
      {/each}
    {/if}
  </div>
</div>

<style>
  .log-panel {
    display: flex;
    flex-direction: column;
    height: 100%;
  }
  
  .log-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 1rem;
    border-bottom: 1px solid #334155;
  }
  
  h3 {
    margin: 0;
    font-size: 1rem;
    color: #e2e8f0;
  }
  
  .log-controls {
    display: flex;
    gap: 0.5rem;
    align-items: center;
  }
  
  .log-controls label {
    display: flex;
    align-items: center;
    gap: 0.25rem;
    font-size: 0.875rem;
    color: #94a3b8;
  }
  
  .log-controls button {
    padding: 0.25rem 0.75rem;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 4px;
    color: #e2e8f0;
    font-size: 0.875rem;
    cursor: pointer;
    transition: all 0.2s;
  }
  
  .log-controls button:hover {
    background: #334155;
  }
  
  .log-container {
    flex: 1;
    overflow-y: auto;
    margin-top: 1rem;
    padding: 0.5rem;
    background: #0f172a;
    border-radius: 4px;
    font-family: 'Consolas', 'Monaco', monospace;
    font-size: 0.8125rem;
  }
  
  .no-logs {
    text-align: center;
    color: #64748b;
    padding: 2rem;
  }
  
  .log-entry {
    display: flex;
    gap: 0.5rem;
    padding: 0.25rem 0.5rem;
    margin-bottom: 0.25rem;
    border-radius: 4px;
    line-height: 1.4;
  }
  
  .log-entry:hover {
    background: rgba(255, 255, 255, 0.05);
  }
  
  .log-time {
    color: #64748b;
    white-space: nowrap;
  }
  
  .log-level {
    font-weight: 600;
    white-space: nowrap;
  }
  
  .log-message {
    flex: 1;
    color: #e2e8f0;
    word-break: break-word;
  }
  
  .log-error .log-level {
    color: #ef4444;
  }
  
  .log-warning .log-level {
    color: #f59e0b;
  }
  
  .log-info .log-level {
    color: #3b82f6;
  }
  
  .log-debug .log-level {
    color: #8b5cf6;
  }
</style>
