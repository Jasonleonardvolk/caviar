<script>
  import { createEventDispatcher } from 'svelte';
  import AvatarOverlay from './AvatarOverlay.svelte';
  
  // Props
  export let concept = null;
  export let selected = false;
  export let showAttribution = true;
  export let size = 'medium'; // small, medium, large
  export let interactive = true;
  export let hoverable = true;
  
  // Local state
  let hover = false;
  
  // Event dispatcher
  const dispatch = createEventDispatcher();
  
  // Derived values
  $: nodeClass = `node ${size} ${selected ? 'selected' : ''} ${hoverable && hover ? 'hover' : ''} ${interactive ? 'interactive' : ''}`;
  $: hasSource = concept && (concept.imported_from || concept.source_title || concept.document_id);
  $: hasAttribution = concept && (concept.user_id || concept.persona_id);
  $: sourceTitle = getSourceTitle();
  $: sourceIcon = getSourceIcon();
  
  // Click handler
  function handleClick() {
    if (!interactive) return;
    dispatch('click', { concept });
  }
  
  // Get the concept type class
  function getTypeClass(type) {
    switch (type?.toLowerCase()) {
      case 'document': return 'document';
      case 'section': return 'section';
      case 'paragraph': return 'paragraph';
      case 'image': return 'image';
      case 'code': return 'code';
      case 'table': return 'table';
      case 'quote': return 'quote';
      case 'note': return 'note';
      default: return 'default';
    }
  }
  
  // Get the concept title or a default
  function getTitle() {
    if (!concept) return 'Unknown';
    return concept.title || concept.name || concept.id || 'Untitled';
  }
  
  // Get the source title
  function getSourceTitle() {
    if (!concept) return '';
    
    return concept.source_title 
      || (concept.imported_from ? extractFilename(concept.imported_from) : '')
      || '';
  }
  
  // Extract filename from a path
  function extractFilename(path) {
    if (!path) return '';
    
    // Remove directory path
    const parts = path.split(/[\/\\]/);
    return parts[parts.length - 1];
  }
  
  // Get an icon for the source
  function getSourceIcon() {
    if (!concept) return '📄';
    
    if (concept.imported_from) {
      const ext = concept.imported_from.split('.').pop()?.toLowerCase();
      
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
    
    return '📄';
  }
  
  // Format creation timestamp
  function formatTimestamp(timestamp) {
    if (!timestamp) return '';
    
    try {
      const date = new Date(timestamp);
      return date.toLocaleString();
    } catch (e) {
      return timestamp;
    }
  }
</script>

<div 
  class={nodeClass}
  class:has-source={hasSource}
  on:click={handleClick}
  on:mouseenter={() => hover = true}
  on:mouseleave={() => hover = false}
>
  <div class="node-content {getTypeClass(concept?.type)}">
    <div class="title-area">
      {#if concept && concept.type}
        <span class="type-badge">{concept.type}</span>
      {/if}
      <h3 class="title">{getTitle()}</h3>
    </div>
    
    {#if concept && concept.description}
      <div class="description">{concept.description}</div>
    {/if}
    
    {#if concept && concept.preview}
      <div class="preview">{concept.preview}</div>
    {/if}
    
    {#if hasSource && showAttribution}
      <div class="source-badge" title="Imported from: {sourceTitle}">
        <span class="source-icon">{sourceIcon}</span>
        <span class="source-label">{sourceTitle}</span>
      </div>
    {/if}
    
    {#if concept && concept.created_at}
      <div class="timestamp">Created: {formatTimestamp(concept.created_at)}</div>
    {/if}
    
    {#if concept && concept.tags && concept.tags.length > 0}
      <div class="tags">
        {#each concept.tags as tag}
          <span class="tag">{tag}</span>
        {/each}
      </div>
    {/if}
  </div>
  
  {#if hasAttribution && showAttribution}
    <AvatarOverlay 
      user={{ concept_id: concept.user_id, name: concept.user_name }} 
      personaId={concept.persona_id}
      timestamp={concept.created_at}
      conceptId={concept.id}
      position="bottom-right"
      size={size}
    />
  {/if}
</div>

<style>
  .node {
    position: relative;
    border-radius: 8px;
    background-color: var(--node-bg, #2a2a2a);
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    transition: all 0.2s ease;
    overflow: hidden;
  }
  
  .node.small {
    width: 200px;
    min-height: 80px;
    font-size: 0.8rem;
  }
  
  .node.medium {
    width: 280px;
    min-height: 120px;
    font-size: 1rem;
  }
  
  .node.large {
    width: 360px;
    min-height: 160px;
    font-size: 1.1rem;
  }
  
  .node.selected {
    box-shadow: 0 0 0 2px var(--selection-color, #4a9eff);
  }
  
  .node.hover {
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    transform: translateY(-2px);
  }
  
  .node.interactive {
    cursor: pointer;
  }
  
  .node-content {
    padding: 16px;
    height: 100%;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  
  .title-area {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  
  .title {
    margin: 0;
    font-size: 1.1em;
    font-weight: 600;
    color: var(--title-color, #ffffff);
    word-break: break-word;
  }
  
  .small .title {
    font-size: 1em;
  }
  
  .large .title {
    font-size: 1.2em;
  }
  
  .type-badge {
    display: inline-block;
    font-size: 0.7em;
    padding: 2px 6px;
    border-radius: 4px;
    background-color: var(--type-badge-bg, rgba(74, 158, 255, 0.2));
    color: var(--type-badge-color, #4a9eff);
    text-transform: uppercase;
    align-self: flex-start;
  }
  
  .description {
    font-size: 0.9em;
    color: var(--description-color, #cccccc);
    margin-top: 4px;
    word-break: break-word;
  }
  
  .preview {
    font-size: 0.85em;
    color: var(--preview-color, #aaaaaa);
    margin-top: 4px;
    overflow: hidden;
    text-overflow: ellipsis;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
  }
  
  .source-badge {
    margin-top: auto;
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.8em;
    padding: 4px 8px;
    border-radius: 4px;
    background-color: var(--source-badge-bg, rgba(0, 0, 0, 0.2));
    color: var(--source-badge-color, #bbbbbb);
    align-self: flex-start;
  }
  
  .source-icon {
    font-size: 1.2em;
  }
  
  .source-label {
    max-width: 150px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  
  .timestamp {
    font-size: 0.75em;
    color: var(--timestamp-color, #888888);
    margin-top: 8px;
  }
  
  .tags {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin-top: 8px;
  }
  
  .tag {
    font-size: 0.75em;
    padding: 2px 6px;
    border-radius: 4px;
    background-color: var(--tag-bg, rgba(0, 150, 136, 0.2));
    color: var(--tag-color, #00bfa5);
  }
  
  /* Node type styling */
  .document {
    border-left: 4px solid var(--document-color, #3f51b5);
  }
  
  .section {
    border-left: 4px solid var(--section-color, #9c27b0);
  }
  
  .paragraph {
    border-left: 4px solid var(--paragraph-color, #4caf50);
  }
  
  .image {
    border-left: 4px solid var(--image-color, #ff9800);
  }
  
  .code {
    border-left: 4px solid var(--code-color, #607d8b);
  }
  
  .table {
    border-left: 4px solid var(--table-color, #795548);
  }
  
  .quote {
    border-left: 4px solid var(--quote-color, #ff5722);
  }
  
  .note {
    border-left: 4px solid var(--note-color, #8bc34a);
  }
  
  .default {
    border-left: 4px solid var(--default-color, #757575);
  }
</style>
