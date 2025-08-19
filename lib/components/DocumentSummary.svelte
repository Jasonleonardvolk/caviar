<!-- components/DocumentSummary.svelte -->
<script lang="ts">
  import { removeConceptDiff, type ConceptDiff } from '$lib/stores/conceptMesh';
  
  export let diff: ConceptDiff;
  export let title: string = diff.title;
  export let concepts: string[] = diff.concepts || [];
  export let summary: string | undefined = diff.summary;
  
  let showDetails = false;
  
  function toggleDetails() {
    showDetails = !showDetails;
  }
  
  function handleRemove() {
    if (confirm('Remove this document from memory?')) {
      removeConceptDiff(diff.id);
    }
  }
  
  function formatTimestamp(date: Date): string {
    return new Intl.RelativeTimeFormat('en', { numeric: 'auto' }).format(
      Math.round((date.getTime() - Date.now()) / (1000 * 60 * 60 * 24)),
      'day'
    );
  }
  
  function getFileIcon(filename: string): string {
    const ext = filename.split('.').pop()?.toLowerCase();
    switch (ext) {
      case 'pdf': return '📄';
      case 'doc':
      case 'docx': return '📝';
      case 'txt': return '📃';
      case 'md': return '📋';
      case 'rtf': return '📝';
      default: return '📄';
    }
  }
</script>

<div class="bg-white rounded-md shadow-sm border border-gray-200 p-3 mb-2 hover:shadow-md transition-shadow duration-200">
  <!-- Document header -->
  <div class="flex items-start justify-between">
    <div class="flex-1 min-w-0">
      <h3 class="text-sm font-semibold text-tori-text mb-1 truncate flex items-center gap-2" title={title}>
        <span class="text-base">{getFileIcon(title)}</span>
        <span>{title}</span>
      </h3>
      
      <!-- Timestamp -->
      <p class="text-xs text-gray-500 mb-2">
        Added {formatTimestamp(diff.timestamp)}
      </p>
    </div>
    
    <!-- Actions -->
    <div class="flex items-center gap-1 ml-2">
      <button 
        class="text-xs text-gray-500 hover:text-gray-700 p-1"
        on:click={toggleDetails}
        title="Toggle details"
      >
        {showDetails ? '▼' : '▶'}
      </button>
      <button 
        class="text-xs text-red-500 hover:text-red-700 p-1"
        on:click={handleRemove}
        title="Remove document"
      >
        ✕
      </button>
    </div>
  </div>
  
  <!-- Extracted concepts as tags -->
  {#if concepts && concepts.length > 0}
    <div class="mb-2 flex flex-wrap gap-1">
      {#each concepts as concept}
        <span class="inline-block bg-blue-100 text-blue-800 text-xs font-medium 
                     rounded-full px-2 py-0.5 border border-blue-200">
          {concept}
        </span>
      {/each}
    </div>
  {/if}
  
  <!-- Summary text -->
  {#if summary}
    <p class="text-xs text-gray-600 italic mb-2">{summary}</p>
  {/if}
  
  <!-- Expandable details -->
  {#if showDetails}
    <div class="border-t border-gray-100 pt-2 mt-2">
      <div class="text-xs text-gray-500 space-y-1">
        <div><strong>ID:</strong> {diff.id}</div>
        <div><strong>Type:</strong> {diff.type}</div>
        {#if diff.metadata}
          {#if diff.metadata.size}
            <div><strong>Size:</strong> {diff.metadata.size} bytes</div>
          {/if}
          {#if diff.metadata.type}
            <div><strong>MIME Type:</strong> {diff.metadata.type}</div>
          {/if}
          {#if diff.metadata.lastModified}
            <div><strong>Modified:</strong> {new Date(diff.metadata.lastModified).toLocaleDateString()}</div>
          {/if}
        {/if}
        <div><strong>Concepts Count:</strong> {concepts.length}</div>
      </div>
    </div>
  {/if}
  
  <!-- Action buttons for expanded view -->
  {#if showDetails}
    <div class="border-t border-gray-100 pt-2 mt-2 flex gap-2">
      <button class="tori-button-secondary text-xs px-2 py-1">
        📝 Summarize
      </button>
      <button class="tori-button-secondary text-xs px-2 py-1">
        🔍 Analyze
      </button>
      <button class="tori-button-secondary text-xs px-2 py-1">
        💭 Discuss
      </button>
    </div>
  {/if}
</div>