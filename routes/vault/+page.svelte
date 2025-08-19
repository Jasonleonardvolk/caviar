<!-- Vault page - Memory vault interface -->
<script lang="ts">
  import { onMount } from 'svelte';
  import { conceptMesh, systemCoherence, clearConceptMesh } from '$lib/stores/conceptMesh';
  import { ghostPersona } from '$lib/stores/ghostPersona';
  
  let mounted = false;
  let searchQuery = '';
  let selectedEntry: any = null;
  let filterType: 'all' | 'documents' | 'conversations' | 'concepts' = 'all';
  let sortBy: 'recent' | 'alphabetical' | 'size' = 'recent';
  
  onMount(() => {
    mounted = true;
  });
  
  // Filtered and sorted entries
  $: filteredEntries = $conceptMesh
    .filter(entry => {
      // Filter by type
      if (filterType === 'documents' && entry.type !== 'document') return false;
      if (filterType === 'conversations' && entry.type !== 'chat') return false;
      if (filterType === 'concepts' && !entry.concepts.length) return false;
      
      // Filter by search query
      if (searchQuery) {
        const query = searchQuery.toLowerCase();
        return (
          entry.title.toLowerCase().includes(query) ||
          entry.summary?.toLowerCase().includes(query) ||
          entry.concepts.some(concept => concept.toLowerCase().includes(query))
        );
      }
      
      return true;
    })
    .sort((a, b) => {
      switch (sortBy) {
        case 'recent':
          return b.timestamp.getTime() - a.timestamp.getTime();
        case 'alphabetical':
          return a.title.localeCompare(b.title);
        case 'size':
          return b.concepts.length - a.concepts.length;
        default:
          return 0;
      }
    });
  
  function formatFileSize(bytes: number | undefined): string {
    if (!bytes) return 'Unknown size';
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
  }
  
  function getEntryIcon(entry: any): string {
    if (entry.type === 'document') {
      const ext = entry.title.split('.').pop()?.toLowerCase();
      switch (ext) {
        case 'pdf': return '📕';
        case 'doc':
        case 'docx': return '📝';
        case 'txt': return '📄';
        case 'md': return '📋';
        case 'json': return '🔧';
        case 'csv':
        case 'xlsx': return '📊';
        default: return '📄';
      }
    }
    if (entry.type === 'chat') return '💬';
    return '🧠';
  }
  
  function getProcessingBadge(entry: any): string | null {
    if (entry.metadata?.processedBy) return entry.metadata.processedBy;
    if (entry.metadata?.fallbackMode) return 'Fallback';
    if (entry.metadata?.elfinScript) return 'ELFIN++';
    return null;
  }
  
  function formatTimestamp(date: Date): string {
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    const diffMinutes = Math.floor(diffMs / (1000 * 60));
    
    if (diffMinutes < 1) return 'Just now';
    if (diffMinutes < 60) return `${diffMinutes}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString();
  }
  
  function openEntryDetails(entry: any) {
    selectedEntry = entry;
  }
  
  function closeDetails() {
    selectedEntry = null;
  }
  
  function deleteEntry(entryId: string) {
    if (confirm('Are you sure you want to delete this entry?')) {
      const { removeConceptDiff } = conceptMesh;
      // Implementation would call removeConceptDiff(entryId)
      console.log('Deleting entry:', entryId);
    }
  }
  
  function exportVault() {
    const data = JSON.stringify($conceptMesh, null, 2);
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `tori-vault-export-${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }
  
  function clearVault() {
    if (confirm('Are you sure you want to clear the entire vault? This action cannot be undone.')) {
      clearConceptMesh();
    }
  }
</script>

<svelte:head>
  <title>TORI - Memory Vault</title>
</svelte:head>

<!-- Vault interface -->
<div class="flex flex-col h-full bg-white">
  
  <!-- Vault header -->
  <div class="px-6 py-4 border-b border-gray-200 bg-gray-50">
    <div class="flex items-center justify-between">
      <div>
        <h1 class="text-xl font-bold text-gray-900">Memory Vault</h1>
        <p class="text-sm text-gray-600 mt-1">
          {filteredEntries.length} of {$conceptMesh.length} entries
          • System coherence: {Math.round($systemCoherence * 100)}%
        </p>
      </div>
      
      <!-- Vault actions -->
      <div class="flex items-center space-x-2">
        <button
          on:click={exportVault}
          class="px-3 py-2 text-sm bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
        >
          📤 Export
        </button>
        <button
          on:click={clearVault}
          class="px-3 py-2 text-sm bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors"
        >
          🗑️ Clear
        </button>
      </div>
    </div>
  </div>
  
  <!-- Search and filters -->
  <div class="px-6 py-4 border-b border-gray-200 bg-white">
    <div class="flex items-center space-x-4">
      <!-- Search input -->
      <div class="flex-1">
        <input
          type="text"
          bind:value={searchQuery}
          placeholder="Search entries, concepts, or content..."
          class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        />
      </div>
      
      <!-- Filters -->
      <select
        bind:value={filterType}
        class="px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
      >
        <option value="all">All Types</option>
        <option value="documents">Documents</option>
        <option value="conversations">Conversations</option>
        <option value="concepts">Concepts</option>
      </select>
      
      <!-- Sort -->
      <select
        bind:value={sortBy}
        class="px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
      >
        <option value="recent">Most Recent</option>
        <option value="alphabetical">Alphabetical</option>
        <option value="size">By Concepts</option>
      </select>
    </div>
  </div>
  
  <!-- Vault contents -->
  <div class="flex-1 overflow-hidden">
    {#if filteredEntries.length === 0}
      <!-- Empty state -->
      <div class="flex flex-col items-center justify-center h-full text-center">
        <div class="text-6xl mb-4">🏛️</div>
        <h2 class="text-xl font-semibold text-gray-900 mb-2">
          {searchQuery ? 'No matching entries' : 'Vault is empty'}
        </h2>
        <p class="text-gray-600 max-w-md">
          {searchQuery 
            ? `No entries found matching "${searchQuery}". Try adjusting your search or filters.`
            : 'Upload documents or start conversations to begin populating your memory vault.'
          }
        </p>
      </div>
    {:else}
      <!-- Entry grid -->
      <div class="h-full overflow-y-auto p-6">
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {#each filteredEntries as entry}
            <div 
              class="border border-gray-200 rounded-lg p-4 hover:border-gray-300 hover:shadow-sm transition-all duration-200 cursor-pointer"
              on:click={() => openEntryDetails(entry)}
            >
              <!-- Entry header -->
              <div class="flex items-start justify-between mb-3">
                <div class="flex items-center space-x-2 flex-1 min-w-0">
                  <div class="text-2xl flex-shrink-0">
                    {getEntryIcon(entry)}
                  </div>
                  <div class="min-w-0 flex-1">
                    <h3 class="font-medium text-gray-900 truncate" title={entry.title}>
                      {entry.title}
                    </h3>
                    <p class="text-xs text-gray-500 mt-1">
                      {formatTimestamp(entry.timestamp)}
                    </p>
                  </div>
                </div>
                
                <!-- Processing badge -->
                {#if getProcessingBadge(entry)}
                  <span class="px-2 py-1 text-xs rounded-full flex-shrink-0 ml-2
                                {entry.metadata?.processedBy ? 'bg-purple-100 text-purple-700' :
                                 entry.metadata?.fallbackMode ? 'bg-yellow-100 text-yellow-700' :
                                 'bg-blue-100 text-blue-700'}">
                    {getProcessingBadge(entry)}
                  </span>
                {/if}
              </div>
              
              <!-- Entry summary -->
              {#if entry.summary}
                <p class="text-sm text-gray-600 mb-3 line-clamp-2">
                  {entry.summary}
                </p>
              {/if}
              
              <!-- Concepts -->
              {#if entry.concepts && entry.concepts.length > 0}
                <div class="flex flex-wrap gap-1 mb-3">
                  {#each entry.concepts.slice(0, 4) as concept}
                    <span class="px-2 py-1 text-xs bg-blue-100 text-blue-700 rounded-full">
                      {concept}
                    </span>
                  {/each}
                  {#if entry.concepts.length > 4}
                    <span class="px-2 py-1 text-xs bg-gray-100 text-gray-600 rounded-full">
                      +{entry.concepts.length - 4}
                    </span>
                  {/if}
                </div>
              {/if}
              
              <!-- Entry metadata -->
              <div class="flex items-center justify-between text-xs text-gray-500">
                <span>
                  {entry.type === 'document' ? 'Document' : 
                   entry.type === 'chat' ? 'Conversation' : 'Concept'}
                </span>
                
                {#if entry.metadata?.size}
                  <span>{formatFileSize(entry.metadata.size)}</span>
                {/if}
              </div>
            </div>
          {/each}
        </div>
      </div>
    {/if}
  </div>
</div>

<!-- Entry details modal -->
{#if selectedEntry}
  <div class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
    <div class="bg-white rounded-lg max-w-2xl w-full max-h-[80vh] overflow-hidden">
      <!-- Modal header -->
      <div class="flex items-center justify-between px-6 py-4 border-b border-gray-200">
        <div class="flex items-center space-x-3">
          <div class="text-2xl">{getEntryIcon(selectedEntry)}</div>
          <div>
            <h2 class="text-lg font-semibold text-gray-900">{selectedEntry.title}</h2>
            <p class="text-sm text-gray-500">{formatTimestamp(selectedEntry.timestamp)}</p>
          </div>
        </div>
        
        <button
          on:click={closeDetails}
          class="p-2 hover:bg-gray-100 rounded-lg transition-colors"
        >
          ✕
        </button>
      </div>
      
      <!-- Modal content -->
      <div class="px-6 py-4 overflow-y-auto max-h-[60vh]">
        <!-- Summary -->
        {#if selectedEntry.summary}
          <div class="mb-4">
            <h3 class="font-medium text-gray-900 mb-2">Summary</h3>
            <p class="text-gray-700">{selectedEntry.summary}</p>
          </div>
        {/if}
        
        <!-- Concepts -->
        {#if selectedEntry.concepts && selectedEntry.concepts.length > 0}
          <div class="mb-4">
            <h3 class="font-medium text-gray-900 mb-2">
              Concepts ({selectedEntry.concepts.length})
            </h3>
            <div class="flex flex-wrap gap-2">
              {#each selectedEntry.concepts as concept}
                <span class="px-3 py-1 text-sm bg-blue-100 text-blue-700 rounded-full">
                  {concept}
                </span>
              {/each}
            </div>
          </div>
        {/if}
        
        <!-- Metadata -->
        {#if selectedEntry.metadata}
          <div class="mb-4">
            <h3 class="font-medium text-gray-900 mb-2">Metadata</h3>
            <div class="bg-gray-50 rounded-lg p-3 text-sm">
              <dl class="space-y-1">
                {#if selectedEntry.metadata.size}
                  <div class="flex justify-between">
                    <dt class="text-gray-600">Size:</dt>
                    <dd class="text-gray-900">{formatFileSize(selectedEntry.metadata.size)}</dd>
                  </div>
                {/if}
                {#if selectedEntry.metadata.type}
                  <div class="flex justify-between">
                    <dt class="text-gray-600">Type:</dt>
                    <dd class="text-gray-900">{selectedEntry.metadata.type}</dd>
                  </div>
                {/if}
                {#if selectedEntry.metadata.processedBy}
                  <div class="flex justify-between">
                    <dt class="text-gray-600">Processed by:</dt>
                    <dd class="text-gray-900">{selectedEntry.metadata.processedBy}</dd>
                  </div>
                {/if}
                {#if selectedEntry.metadata.elfinScript}
                  <div class="flex justify-between">
                    <dt class="text-gray-600">ELFIN++ Script:</dt>
                    <dd class="text-gray-900">{selectedEntry.metadata.elfinScript}</dd>
                  </div>
                {/if}
                {#if selectedEntry.metadata.extractedConcepts}
                  <div class="flex justify-between">
                    <dt class="text-gray-600">Concepts extracted:</dt>
                    <dd class="text-gray-900">{selectedEntry.metadata.extractedConcepts}</dd>
                  </div>
                {/if}
                {#if selectedEntry.metadata.processingTime}
                  <div class="flex justify-between">
                    <dt class="text-gray-600">Processing time:</dt>
                    <dd class="text-gray-900">{Math.round(selectedEntry.metadata.processingTime)}ms</dd>
                  </div>
                {/if}
              </dl>
            </div>
          </div>
        {/if}
      </div>
      
      <!-- Modal footer -->
      <div class="flex items-center justify-between px-6 py-4 border-t border-gray-200 bg-gray-50">
        <div class="text-xs text-gray-500">
          Entry ID: {selectedEntry.id}
        </div>
        
        <div class="flex space-x-2">
          <button
            on:click={() => deleteEntry(selectedEntry.id)}
            class="px-3 py-2 text-sm bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors"
          >
            Delete
          </button>
          <button
            on:click={closeDetails}
            class="px-3 py-2 text-sm bg-gray-600 hover:bg-gray-700 text-white rounded-lg transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  </div>
{/if}

<style>
  .line-clamp-2 {
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }
</style>