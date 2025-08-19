<script>
  import { createEventDispatcher } from 'svelte';
  import { fade, fly } from 'svelte/transition';
  import { invoke } from '@tauri-apps/api/tauri';
  import AvatarOverlay from './AvatarOverlay.svelte';
  
  // Props
  export let concept = null;
  export let visible = false;
  export let showSourceRecall = true;
  export let showRelatedConcepts = true;
  
  // Local state
  let loading = false;
  let error = '';
  let sourceContent = null;
  let relatedConcepts = [];
  let psiarc = null;
  let activeTab = 'info';
  
  // Event dispatcher
  const dispatch = createEventDispatcher();
  
  // Watch for concept changes
  $: if (concept && visible) {
    loadConceptDetails();
  }
  
  // Close the panel
  function close() {
    dispatch('close');
  }
  
  // Load concept details
  async function loadConceptDetails() {
    if (!concept) return;
    
    loading = true;
    error = '';
    
    try {
      // Load related concepts
      if (showRelatedConcepts) {
        relatedConcepts = await invoke('concept_mesh:get_related_concepts', {
          conceptId: concept.id
        });
      }
      
      // Check if we need to load the source information
      if (concept.document_id && showSourceRecall) {
        // Find the PsiArc the concept came from
        if (concept.psiarc_id) {
          psiarc = concept.psiarc_id;
        } else if (concept.concept_ingest_origin && concept.concept_ingest_origin.psiarc_id) {
          psiarc = concept.concept_ingest_origin.psiarc_id;
        }
      }
    } catch (err) {
      error = `Failed to load concept details: ${err}`;
    } finally {
      loading = false;
    }
  }
  
  // Recall source content
  async function recallSource() {
    if (!concept || !concept.document_id) return;
    
    loading = true;
    error = '';
    sourceContent = null;
    
    try {
      // Invoke the backend to get source content
      sourceContent = await invoke('concept_mesh:recall_source_content', {
        conceptId: concept.id,
        documentId: concept.document_id,
        psiarc: psiarc
      });
    } catch (err) {
      error = `Failed to recall source content: ${err}`;
      sourceContent = null;
    } finally {
      loading = false;
    }
  }
  
  // Format a timestamp
  function formatTimestamp(timestamp) {
    if (!timestamp) return 'Unknown';
    
    try {
      const date = new Date(timestamp);
      return date.toLocaleString();
    } catch (e) {
      return timestamp;
    }
  }
  
  // Get persona display name
  function getPersonaName(personaId) {
    if (!personaId) return 'Unknown';
    
    switch (personaId) {
      case 'creative_agent': return 'Creative Agent';
      case 'glyphsmith': return 'Glyphsmith';
      case 'memory_pruner': return 'Memory Pruner';
      case 'researcher': return 'Researcher';
      default: return personaId.charAt(0).toUpperCase() + personaId.slice(1).replace('_', ' ');
    }
  }
  
  // Get source icon
  function getSourceIcon(path) {
    if (!path) return '📄';
    
    const ext = path.split('.').pop()?.toLowerCase();
    
    switch (ext) {
      case 'pdf': return '📕';
      case 'docx':
      case 'doc': return '📘';
      case 'txt': return '📃';
      case 'md': return '📝';
      case 'jpg':
      case 'jpeg':
      case 'png':
      case 'gif': return '🖼️';
      case 'csv':
      case 'xlsx':
      case 'xls': return '📊';
      case 'json': return '📋';
      case 'html':
      case 'htm': return '🌐';
      default: return '📄';
    }
  }
  
  // Handle keyboard shortcuts
  function handleKeydown(event) {
    if (!visible) return;
    
    // 'R' key to recall source
    if (event.key.toLowerCase() === 'r' && showSourceRecall && concept?.document_id) {
      recallSource();
    }
    
    // Escape key to close
    if (event.key === 'Escape') {
      close();
    }
  }
</script>

<svelte:window on:keydown={handleKeydown} />

{#if visible && concept}
  <div class="concept-info-panel" transition:fly={{ x: 300, duration: 300 }}>
    <div class="panel-header">
      <div class="title-area">
        <h2>{concept.title || concept.name || concept.id || 'Concept Details'}</h2>
        {#if concept.type}
          <span class="type-badge">{concept.type}</span>
        {/if}
      </div>
      
      <button class="close-button" on:click={close}>×</button>
    </div>
    
    <div class="panel-tabs">
      <button 
        class="tab-button" 
        class:active={activeTab === 'info'}
        on:click={() => activeTab = 'info'}
      >
        Info
      </button>
      
      {#if concept.document_id && showSourceRecall}
        <button 
          class="tab-button" 
          class:active={activeTab === 'source'}
          on:click={() => { activeTab = 'source'; recallSource(); }}
        >
          Source
        </button>
      {/if}
      
      {#if showRelatedConcepts && relatedConcepts.length > 0}
        <button 
          class="tab-button" 
          class:active={activeTab === 'related'}
          on:click={() => activeTab = 'related'}
        >
          Related ({relatedConcepts.length})
        </button>
      {/if}
    </div>
    
    <div class="panel-content">
      {#if loading}
        <div class="loading">Loading...</div>
      {:else if error}
        <div class="error">{error}</div>
      {:else if activeTab === 'info'}
        <div class="info-tab">
          {#if concept.description}
            <div class="section">
              <h3>Description</h3>
              <p>{concept.description}</p>
            </div>
          {/if}
          
          {#if concept.content}
            <div class="section">
              <h3>Content</h3>
              <div class="content-preview">{concept.content}</div>
            </div>
          {/if}
          
          <div class="section">
            <h3>Metadata</h3>
            <div class="metadata-grid">
              <div class="metadata-row">
                <div class="metadata-key">ID</div>
                <div class="metadata-value">{concept.id}</div>
              </div>
              
              {#if concept.created_at}
                <div class="metadata-row">
                  <div class="metadata-key">Created</div>
                  <div class="metadata-value">{formatTimestamp(concept.created_at)}</div>
                </div>
              {/if}
              
              {#if concept.updated_at}
                <div class="metadata-row">
                  <div class="metadata-key">Updated</div>
                  <div class="metadata-value">{formatTimestamp(concept.updated_at)}</div>
                </div>
              {/if}
              
              {#if concept.user_id}
                <div class="metadata-row">
                  <div class="metadata-key">Created By</div>
                  <div class="metadata-value attribution">
                    <div class="avatar-placeholder"></div>
                    <span>{concept.user_name || concept.user_id}</span>
                  </div>
                </div>
              {/if}
              
              {#if concept.persona_id}
                <div class="metadata-row">
                  <div class="metadata-key">Persona</div>
                  <div class="metadata-value">{getPersonaName(concept.persona_id)}</div>
                </div>
              {/if}
            </div>
          </div>
          
          {#if concept.source_title || concept.imported_from || concept.document_id}
            <div class="section">
              <h3>Source</h3>
              <div class="source-info">
                {#if concept.source_title}
                  <div class="source-title">
                    <span class="source-icon">{getSourceIcon(concept.imported_from)}</span>
                    <span>{concept.source_title}</span>
                  </div>
                {/if}
                
                {#if concept.imported_from}
                  <div class="source-path">{concept.imported_from}</div>
                {/if}
                
                {#if concept.document_id}
                  <div class="document-id">ID: {concept.document_id}</div>
                {/if}
                
                <div class="source-actions">
                  {#if concept.document_id && showSourceRecall}
                    <button class="action-button" on:click={() => { activeTab = 'source'; recallSource(); }}>
                      Recall Source <span class="shortcut">(R)</span>
                    </button>
                  {/if}
                  
                  {#if concept.source_title || concept.document_id}
                    <button class="action-button" on:click={() => dispatch('showAllFromSource', { 
                      documentId: concept.document_id,
                      sourceTitle: concept.source_title
                    })}>
                      See All from This Source
                    </button>
                  {/if}
                </div>
              </div>
            </div>
          {/if}
          
          {#if concept.tags && concept.tags.length > 0}
            <div class="section">
              <h3>Tags</h3>
              <div class="tags">
                {#each concept.tags as tag}
                  <span class="tag">{tag}</span>
                {/each}
              </div>
            </div>
          {/if}
        </div>
      {:else if activeTab === 'source' && concept.document_id}
        <div class="source-tab">
          {#if loading}
            <div class="loading">Loading source content...</div>
          {:else if error}
            <div class="error">{error}</div>
          {:else if sourceContent}
            <div class="source-content">
              <div class="source-header">
                <div class="source-title">
                  <span class="source-icon">{getSourceIcon(concept.imported_from)}</span>
                  <span>{concept.source_title || 'Source Content'}</span>
                </div>
                
                {#if concept.imported_from}
                  <div class="source-path">{concept.imported_from}</div>
                {/if}
              </div>
              
              <div class="content-display">
                {#if sourceContent.content_type === 'text'}
                  <div class="text-content">{sourceContent.content}</div>
                {:else if sourceContent.content_type === 'html'}
                  <div class="html-content">
                    {@html sourceContent.content}
                  </div>
                {:else if sourceContent.content_type === 'image'}
                  <div class="image-content">
                    <img src={sourceContent.content} alt="Source content" />
                  </div>
                {:else}
                  <div class="unknown-content">
                    Content type "{sourceContent.content_type}" not supported for preview.
                  </div>
                {/if}
              </div>
            </div>
          {:else}
            <div class="no-source">
              <p>No source content available. Click the button below to recall the source.</p>
              <button class="recall-button" on:click={recallSource}>
                Recall Source
              </button>
            </div>
          {/if}
        </div>
      {:else if activeTab === 'related' && showRelatedConcepts}
        <div class="related-tab">
          {#if relatedConcepts.length === 0}
            <div class="no-related">
              <p>No related concepts found.</p>
            </div>
          {:else}
            <div class="related-list">
              {#each relatedConcepts as related}
                <div class="related-item" on:click={() => dispatch('select', { concept: related })}>
                  <div class="related-content">
                    <div class="related-title">{related.title || related.name || related.id}</div>
                    
                    {#if related.type}
                      <div class="related-type">{related.type}</div>
                    {/if}
                    
                    {#if related.description}
                      <div class="related-description">{related.description}</div>
                    {/if}
                  </div>
                  
                  {#if related.user_id || related.persona_id}
                    <div class="related-attribution">
                      {#if related.user_id}
                        <div class="related-user">{related.user_name || related.user_id}</div>
                      {/if}
                      
                      {#if related.persona_id}
                        <div class="related-persona">{getPersonaName(related.persona_id)}</div>
                      {/if}
                    </div>
                  {/if}
                </div>
              {/each}
            </div>
          {/if}
        </div>
      {/if}
    </div>
    
    {#if concept.user_id || concept.persona_id}
      <div class="attribution-footer">
        <div class="footer-content">
          <div class="attribution-info">
            <div class="attribution-user">
              Created by: <span class="user-name">{concept.user_name || concept.user_id || 'Unknown'}</span>
            </div>
            
            {#if concept.persona_id}
              <div class="attribution-persona">
                Persona: <span class="persona-name">{getPersonaName(concept.persona_id)}</span>
              </div>
            {/if}
            
            {#if concept.created_at}
              <div class="attribution-time">
                on {formatTimestamp(concept.created_at)}
              </div>
            {/if}
          </div>
          
          {#if concept.user_id && concept.persona_id}
            <AvatarOverlay 
              user={{ concept_id: concept.user_id, name: concept.user_name }} 
              personaId={concept.persona_id}
              timestamp={concept.created_at}
              size="medium"
              showTooltip={false}
              interactive={false}
            />
          {/if}
        </div>
      </div>
    {/if}
  </div>
{/if}

<style>
  .concept-info-panel {
    position: fixed;
    top: 0;
    right: 0;
    height: 100%;
    width: 400px;
    background-color: var(--panel-bg, #1e1e1e);
    box-shadow: -2px 0 10px rgba(0, 0, 0, 0.2);
    z-index: 100;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  
  .panel-header {
    padding: 16px;
    border-bottom: 1px solid var(--border-color, #333);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  
  .title-area {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  
  .panel-header h2 {
    margin: 0;
    font-size: 1.5rem;
    color: var(--heading-color, #fff);
  }
  
  .type-badge {
    display: inline-block;
    font-size: 0.7rem;
    padding: 2px 6px;
    border-radius: 4px;
    background-color: var(--type-badge-bg, rgba(74, 158, 255, 0.2));
    color: var(--type-badge-color, #4a9eff);
    text-transform: uppercase;
    align-self: flex-start;
  }
  
  .close-button {
    background: transparent;
    border: none;
    color: var(--text-color, #ccc);
    font-size: 1.5rem;
    cursor: pointer;
    width: 32px;
    height: 32px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 4px;
  }
  
  .close-button:hover {
    background-color: var(--hover-bg, rgba(255, 255, 255, 0.1));
  }
  
  .panel-tabs {
    display: flex;
    border-bottom: 1px solid var(--border-color, #333);
  }
  
  .tab-button {
    padding: 12px 16px;
    background: transparent;
    border: none;
    color: var(--text-color, #ccc);
    font-size: 1rem;
    cursor: pointer;
    transition: all 0.2s ease;
    flex: 1;
    text-align: center;
  }
  
  .tab-button.active {
    color: var(--primary-color, #4a9eff);
    border-bottom: 2px solid var(--primary-color, #4a9eff);
  }
  
  .tab-button:hover:not(.active) {
    background-color: var(--hover-bg, rgba(255, 255, 255, 0.05));
  }
  
  .panel-content {
    flex: 1;
    overflow-y: auto;
    padding: 16px;
  }
  
  .section {
    margin-bottom: 24px;
  }
  
  .section h3 {
    margin: 0 0 8px 0;
    font-size: 1.1rem;
    color: var(--heading-color, #fff);
    padding-bottom: 4px;
    border-bottom: 1px solid var(--border-color, #333);
  }
  
  .loading, .error, .no-source, .no-related {
    padding: 32px;
    text-align: center;
    color: var(--text-muted, #888);
  }
  
  .error {
    color: var(--error-color, #ff5252);
  }
  
  .content-preview {
    font-size: 0.9rem;
    line-height: 1.5;
    color: var(--text-color, #ccc);
    max-height: 200px;
    overflow-y: auto;
    padding: 8px;
    background-color: var(--code-bg, rgba(0, 0, 0, 0.2));
    border-radius: 4px;
  }
  
  .metadata-grid {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  
  .metadata-row {
    display: flex;
    border-bottom: 1px solid var(--border-color-light, #2a2a2a);
    padding-bottom: 8px;
  }
  
  .metadata-key {
    width: 100px;
    font-weight: bold;
    color: var(--text-muted, #888);
  }
  
  .metadata-value {
    flex: 1;
    color: var(--text-color, #ccc);
  }
  
  .source-info {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  
  .source-title {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 1.1rem;
    color: var(--text-color, #ccc);
  }
  
  .source-icon {
    font-size: 1.2em;
  }
  
  .source-path {
    font-size: 0.8rem;
    color: var(--text-muted, #888);
    word-break: break-all;
  }
  
  .document-id {
    font-size: 0.8rem;
    color: var(--text-muted, #888);
    font-family: monospace;
  }
  
  .tags {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }
  
  .tag {
    font-size: 0.8rem;
    padding: 4px 8px;
    border-radius: 4px;
    background-color: var(--tag-bg, rgba(0, 150, 136, 0.2));
    color: var(--tag-color, #00bfa5);
  }
  
  .source-actions {
    margin-top: 12px;
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }
  
  .action-button {
    padding: 8px 16px;
    background-color: var(--button-bg, #2a2a2a);
    color: var(--button-text, #ccc);
    border: 1px solid var(--border-color, #333);
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.9rem;
    transition: all 0.2s ease;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    flex-grow: 1;
    min-width: 120px;
  }
  
  .action-button:hover {
    background-color: var(--button-hover-bg, #333);
  }
  
  .shortcut {
    opacity: 0.7;
    font-size: 0.8rem;
  }
  
  .source-content {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }
  
  .source-header {
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border-color, #333);
  }
  
  .content-display {
    padding: 16px;
    background-color: var(--code-bg, rgba(0, 0, 0, 0.2));
    border-radius: 4px;
    overflow-y: auto;
    max-height: 500px;
  }
  
  .text-content {
    white-space: pre-wrap;
    font-family: monospace;
    font-size: 0.9rem;
    line-height: 1.5;
    color: var(--text-color, #ccc);
  }
  
  .html-content {
    font-size: 0.9rem;
    line-height: 1.5;
    color: var(--text-color, #ccc);
  }
  
  .image-content {
    display: flex;
    justify-content: center;
  }
  
  .image-content img {
    max-width: 100%;
    max-height: 500px;
    object-fit: contain;
  }
  
  .unknown-content {
    padding: 16px;
    text-align: center;
    color: var(--text-muted, #888);
  }
  
  .related-list {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  
  .related-item {
    padding: 12px;
    background-color: var(--card-bg, #2a2a2a);
    border-radius: 4px;
    cursor: pointer;
    transition: all 0.2s ease;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  
  .related-item:hover {
    background-color: var(--card-hover-bg, #333);
  }
  
  .related-title {
    font-size: 1rem;
    font-weight: bold;
    color: var(--text-color, #ccc);
  }
  
  .related-type {
    font-size: 0.8rem;
    color: var(--text-muted, #888);
    text-transform: uppercase;
  }
  
  .related-description {
    font-size: 0.9rem;
    color: var(--text-color, #ccc);
    margin-top: 4px;
  }
  
  .related-attribution {
    font-size: 0.8rem;
    color: var(--text-muted, #888);
    display: flex;
    justify-content: space-between;
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px solid var(--border-color-light, #2a2a2a);
  }
  
  .attribution-footer {
    padding: 16px;
    border-top: 1px solid var(--border-color, #333);
    background-color: var(--footer-bg, rgba(0, 0, 0, 0.2));
  }
  
  .footer-content {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  
  .attribution-info {
    display: flex;
    flex-direction: column;
    gap: 4px;
    font-size: 0.8rem;
    color: var(--text-muted, #888);
  }
  
  .user-name {
    color: var(--user-color, #4a9eff);
    font-weight: bold;
  }
  
  .persona-name {
    color: var(--persona-color, #9c27b0);
    font-weight: bold;
  }
  
  .attribution-time {
    font-style: italic;
  }
</style>
