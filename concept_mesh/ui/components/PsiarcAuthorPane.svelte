<script>
  import { invoke } from '@tauri-apps/api/tauri';
  import { createEventDispatcher } from 'svelte';
  
  // Props
  export let psiarc = null;
  export let currentFrameId = null;
  
  // State
  let loading = false;
  let error = '';
  let currentDiff = null;
  let authorInfo = null;
  let attributionHistory = [];
  
  // Event dispatcher
  const dispatch = createEventDispatcher();
  
  // Watch for frame changes
  $: if (psiarc && currentFrameId !== null) {
    loadFrameDiff(currentFrameId);
  }
  
  // Load diff for the current frame
  async function loadFrameDiff(frameId) {
    if (!psiarc || frameId === null) return;
    
    loading = true;
    error = '';
    
    try {
      // Fetch the diff for the current frame
      const diff = await invoke('psiarc_viewer:get_diff', { 
        psiarc, 
        frameId 
      });
      
      currentDiff = diff;
      
      // Extract author information
      if (diff && diff.metadata) {
        const userId = diff.metadata.user_id;
        const personaId = diff.metadata.persona_id;
        const sessionId = diff.metadata.session_id;
        
        if (userId) {
          // Fetch user details
          const user = await invoke('get_user_details', { userId });
          
          authorInfo = {
            userId,
            personaId,
            sessionId,
            user: user || { name: 'Unknown User' },
            timestamp: diff.timestamp_ms ? new Date(diff.timestamp_ms).toLocaleString() : 'Unknown',
          };
          
          // Add to history if not already present
          if (!attributionHistory.some(h => h.userId === userId && h.personaId === personaId)) {
            attributionHistory = [...attributionHistory, authorInfo];
          }
          
          // Keep history sorted by most recent first
          attributionHistory.sort((a, b) => {
            // If the same user, sort by timestamp if available
            if (a.userId === b.userId) {
              const aTime = a.timestamp !== 'Unknown' ? new Date(a.timestamp).getTime() : 0;
              const bTime = b.timestamp !== 'Unknown' ? new Date(b.timestamp).getTime() : 0;
              return bTime - aTime;
            }
            // Otherwise sort by user ID
            return a.userId.localeCompare(b.userId);
          });
        } else {
          authorInfo = null;
        }
      } else {
        authorInfo = null;
      }
    } catch (err) {
      error = `Failed to load frame diff: ${err}`;
      authorInfo = null;
    } finally {
      loading = false;
    }
  }
  
  // Clear history
  function clearHistory() {
    attributionHistory = [];
  }
  
  // Navigate to user
  function navigateToUser(userId) {
    dispatch('navigateToUser', { userId });
  }
  
  // Filter by user
  function filterByUser(userId) {
    dispatch('filterByUser', { userId });
  }
  
  // Filter by persona
  function filterByPersona(personaId) {
    dispatch('filterByPersona', { personaId });
  }
  
  // Get persona display name
  function getPersonaName(personaId) {
    switch (personaId) {
      case 'creative_agent': return 'Creative Agent';
      case 'glyphsmith': return 'Glyphsmith';
      case 'memory_pruner': return 'Memory Pruner';
      case 'researcher': return 'Researcher';
      default: return personaId || 'Unknown';
    }
  }
  
  // Get avatar initials
  function getInitials(name) {
    if (!name) return '?';
    return name.split(' ').map(n => n[0]).join('').toUpperCase();
  }
</script>

<div class="author-pane">
  <div class="header">
    <h2>Attribution</h2>
    {#if attributionHistory.length > 0}
      <button class="clear-button" on:click={clearHistory}>Clear History</button>
    {/if}
  </div>
  
  {#if loading}
    <div class="loading">Loading attribution data...</div>
  {:else if error}
    <div class="error">{error}</div>
  {:else if authorInfo}
    <div class="current-author">
      <div class="author-card">
        <div class="avatar">
          {#if authorInfo.user.avatar_url}
            <img src={authorInfo.user.avatar_url} alt="Avatar" />
          {:else}
            <div class="avatar-placeholder">{getInitials(authorInfo.user.name)}</div>
          {/if}
        </div>
        
        <div class="author-details">
          <h3>{authorInfo.user.name || 'Unknown User'}</h3>
          <div class="persona-badge">{getPersonaName(authorInfo.personaId)}</div>
          <div class="timestamp">{authorInfo.timestamp}</div>
        </div>
        
        <div class="author-actions">
          <button class="action-button" on:click={() => navigateToUser(authorInfo.userId)}>
            <span class="icon">👤</span>
          </button>
          <button class="action-button" on:click={() => filterByUser(authorInfo.userId)}>
            <span class="icon">🔍</span>
          </button>
        </div>
      </div>
    </div>
  {:else}
    <div class="no-author">
      <p>No attribution data available for this diff.</p>
    </div>
  {/if}
  
  {#if attributionHistory.length > 0}
    <div class="history">
      <h3>History</h3>
      <div class="history-list">
        {#each attributionHistory as history}
          <div class="history-item" class:active={authorInfo && authorInfo.userId === history.userId && authorInfo.personaId === history.personaId}>
            <div class="history-avatar">
              {#if history.user.avatar_url}
                <img src={history.user.avatar_url} alt="Avatar" />
              {:else}
                <div class="avatar-placeholder small">{getInitials(history.user.name)}</div>
              {/if}
            </div>
            
            <div class="history-details">
              <div class="history-name">{history.user.name || 'Unknown'}</div>
              <div class="history-persona">{getPersonaName(history.personaId)}</div>
            </div>
            
            <div class="history-actions">
              <button class="action-button small" on:click={() => filterByUser(history.userId)}>
                <span class="icon small">🔍</span>
              </button>
            </div>
          </div>
        {/each}
      </div>
    </div>
  {/if}
  
  {#if currentDiff && currentDiff.metadata}
    <div class="metadata">
      <h3>Metadata</h3>
      <div class="metadata-list">
        {#each Object.entries(currentDiff.metadata) as [key, value]}
          <div class="metadata-item">
            <div class="metadata-key">{key}</div>
            <div class="metadata-value">{typeof value === 'object' ? JSON.stringify(value) : value}</div>
          </div>
        {/each}
      </div>
    </div>
  {/if}
</div>

<style>
  .author-pane {
    background-color: var(--card-bg, #2a2a2a);
    border-radius: 8px;
    padding: 1rem;
    height: 100%;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
  }
  
  .header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
    border-bottom: 1px solid var(--border-color, #444);
    padding-bottom: 0.5rem;
  }
  
  .header h2 {
    margin: 0;
    font-size: 1.2rem;
    color: var(--text-color, #fff);
  }
  
  .clear-button {
    background: transparent;
    border: none;
    color: var(--text-muted, #aaa);
    cursor: pointer;
    font-size: 0.8rem;
    padding: 0;
  }
  
  .clear-button:hover {
    color: var(--text-color, #fff);
    text-decoration: underline;
  }
  
  .loading, .error, .no-author {
    padding: 1rem;
    text-align: center;
    color: var(--text-muted, #aaa);
    font-style: italic;
  }
  
  .error {
    color: var(--error-color, #ff6b6b);
  }
  
  .author-card {
    display: flex;
    align-items: center;
    padding: 1rem;
    background-color: var(--card-hover-bg, #333);
    border-radius: 6px;
    margin-bottom: 1rem;
  }
  
  .avatar {
    width: 50px;
    height: 50px;
    border-radius: 50%;
    overflow: hidden;
    margin-right: 1rem;
  }
  
  .avatar img {
    width: 100%;
    height: 100%;
    object-fit: cover;
  }
  
  .avatar-placeholder {
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    background-color: var(--primary-color, #4a9eff);
    color: white;
    font-weight: bold;
  }
  
  .avatar-placeholder.small {
    font-size: 0.8rem;
  }
  
  .author-details {
    flex: 1;
  }
  
  .author-details h3 {
    margin: 0 0 0.25rem 0;
    font-size: 1rem;
  }
  
  .persona-badge {
    display: inline-block;
    background-color: var(--badge-bg, rgba(74, 158, 255, 0.2));
    color: var(--primary-color, #4a9eff);
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    font-size: 0.8rem;
    margin-bottom: 0.25rem;
  }
  
  .timestamp {
    font-size: 0.75rem;
    color: var(--text-muted, #aaa);
  }
  
  .author-actions {
    display: flex;
    gap: 0.5rem;
  }
  
  .action-button {
    background: transparent;
    border: 1px solid var(--border-color, #444);
    color: var(--text-color, #fff);
    width: 30px;
    height: 30px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: all 0.2s ease;
  }
  
  .action-button:hover {
    background-color: var(--primary-color, #4a9eff);
    border-color: var(--primary-color, #4a9eff);
  }
  
  .action-button.small {
    width: 24px;
    height: 24px;
    font-size: 0.8rem;
  }
  
  .history {
    margin-top: 1rem;
  }
  
  .history h3 {
    font-size: 1rem;
    margin: 0 0 0.5rem 0;
    padding-bottom: 0.25rem;
    border-bottom: 1px solid var(--border-color, #444);
  }
  
  .history-list {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }
  
  .history-item {
    display: flex;
    align-items: center;
    padding: 0.5rem;
    border-radius: 4px;
    background-color: var(--card-bg, #2a2a2a);
    border: 1px solid transparent;
  }
  
  .history-item.active {
    border-color: var(--primary-color, #4a9eff);
    background-color: var(--card-hover-bg, #333);
  }
  
  .history-item:hover {
    background-color: var(--card-hover-bg, #333);
  }
  
  .history-avatar {
    width: 30px;
    height: 30px;
    border-radius: 50%;
    overflow: hidden;
    margin-right: 0.75rem;
  }
  
  .history-details {
    flex: 1;
  }
  
  .history-name {
    font-size: 0.9rem;
    margin-bottom: 0.1rem;
  }
  
  .history-persona {
    font-size: 0.75rem;
    color: var(--text-muted, #aaa);
  }
  
  .metadata {
    margin-top: 1rem;
  }
  
  .metadata h3 {
    font-size: 1rem;
    margin: 0 0 0.5rem 0;
    padding-bottom: 0.25rem;
    border-bottom: 1px solid var(--border-color, #444);
  }
  
  .metadata-list {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }
  
  .metadata-item {
    display: flex;
    font-size: 0.8rem;
    padding: 0.25rem 0;
  }
  
  .metadata-key {
    width: 40%;
    color: var(--text-muted, #aaa);
    padding-right: 0.5rem;
  }
  
  .metadata-value {
    width: 60%;
    word-break: break-word;
  }
</style>
