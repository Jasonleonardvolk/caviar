<!-- Clean MemoryPanel without debug comments or extra sign-in -->
<script lang="ts">
  import { conceptMesh, clearConceptMesh, removeConceptDiff } from '$lib/stores/conceptMesh';
  import { userSession } from '$lib/stores/user';
  import { ghostPersona } from '$lib/stores/ghostPersona';
  import UploadPanel from './UploadPanel.svelte';
  import DocumentSummary from './DocumentSummary.svelte';
  
  let activeTab = 'all'; // 'all', 'docs', 'concepts'
  
  // Filtered content based on active tab
  $: filteredContent = $conceptMesh.filter(diff => {
    switch (activeTab) {
      case 'docs':
        return diff.type === 'document';
      case 'concepts':
        return diff.concepts.length > 0;
      default:
        return true;
    }
  });
  
  // Statistics
  $: stats = {
    total: $conceptMesh.length,
    documents: $conceptMesh.filter(d => d.type === 'document').length,
    conversations: $conceptMesh.filter(d => d.type === 'chat').length,
    concepts: [...new Set($conceptMesh.flatMap(d => d.concepts))].length
  };
  
  function formatRelativeTime(date: Date): string {
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / (1000 * 60));
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
    
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString();
  }
</script>

<div class="flex flex-col h-full bg-gray-50">
  <!-- Header -->
  <div class="p-4 border-b border-gray-200 bg-white">
    <div class="flex items-center justify-between mb-4">
      <h2 class="text-lg font-semibold text-gray-800">Memory System</h2>
      <div class="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded-full">
        {stats.total}
      </div>
    </div>
    
    <!-- User info (only if authenticated) -->
    {#if $userSession.isAuthenticated && $userSession.user}
      <div class="p-3 bg-blue-50 rounded-lg border border-blue-100 mb-4">
        <div class="text-sm font-medium text-blue-900">
          TORI User
        </div>
        <div class="text-xs text-blue-700 mt-1">
          {stats.documents} docs • {stats.conversations} chats • {stats.concepts} concepts
        </div>
        <div class="text-xs text-blue-600 mt-1">
          Current mood: {$ghostPersona.mood || 'Contemplative'}
        </div>
      </div>
    {/if}
  </div>
  
  <!-- Upload Panel -->
  <div class="p-4 border-b border-gray-200 bg-white">
    <UploadPanel />
  </div>
  
  <!-- Navigation Tabs -->
  <div class="flex border-b border-gray-200 bg-white">
    <button 
      class="flex-1 px-4 py-3 text-sm font-medium transition-colors {activeTab === 'all' ? 'text-blue-600 border-b-2 border-blue-600 bg-blue-50' : 'text-gray-600 hover:text-gray-800'}"
      on:click={() => activeTab = 'all'}
    >
      All ({stats.total})
    </button>
    <button 
      class="flex-1 px-4 py-3 text-sm font-medium transition-colors {activeTab === 'concepts' ? 'text-purple-600 border-b-2 border-purple-600 bg-purple-50' : 'text-gray-600 hover:text-gray-800'}"
      on:click={() => activeTab = 'concepts'}
    >
      Concepts ({stats.concepts})
    </button>
    <button 
      class="flex-1 px-4 py-3 text-sm font-medium transition-colors {activeTab === 'docs' ? 'text-green-600 border-b-2 border-green-600 bg-green-50' : 'text-gray-600 hover:text-gray-800'}"
      on:click={() => activeTab = 'docs'}
    >
      Docs ({stats.documents})
    </button>
  </div>
  
  <!-- Content List -->
  <div class="flex-1 overflow-y-auto p-4 space-y-3">
    {#if filteredContent.length === 0}
      <div class="text-center py-12 text-gray-500">
        <div class="text-4xl mb-4">🧠</div>
        <p class="text-sm mb-2">No memory entries yet</p>
        <p class="text-xs text-gray-400">
          Upload documents or start conversations to begin building your knowledge base
        </p>
      </div>
    {:else}
      {#each filteredContent.slice().reverse() as diff (diff.id)}
        {#if diff.type === 'document'}
          <DocumentSummary {diff} />
        {:else if diff.type === 'chat'}
          <!-- Chat conversation entry -->
          <div class="bg-white rounded-lg shadow-sm border border-gray-200 p-4 hover:shadow-md transition-shadow">
            <div class="flex items-start justify-between">
              <div class="flex-1 min-w-0">
                <h3 class="text-sm font-medium text-gray-800 mb-1 flex items-center gap-2">
                  <span class="text-purple-600">💬</span>
                  <span>Conversation</span>
                </h3>
                
                <p class="text-xs text-gray-500 mb-2">
                  {formatRelativeTime(diff.timestamp)}
                </p>
                
                {#if diff.summary}
                  <p class="text-xs text-gray-700 mb-2">{diff.summary}</p>
                {/if}
                
                <!-- Concepts from conversation -->
                {#if diff.concepts && diff.concepts.length > 0}
                  <div class="flex flex-wrap gap-1">
                    {#each diff.concepts as concept}
                      <span class="inline-block bg-purple-100 text-purple-700 text-xs 
                                   rounded-full px-2 py-0.5 border border-purple-200">
                        {concept}
                      </span>
                    {/each}
                  </div>
                {/if}
              </div>
              
              <button 
                class="text-xs text-red-500 hover:text-red-700 p-1 ml-2"
                on:click={() => removeConceptDiff(diff.id)}
                title="Remove conversation"
              >
                ✕
              </button>
            </div>
          </div>
        {/if}
      {/each}
    {/if}
  </div>
  
  <!-- Footer Stats -->
  <div class="border-t border-gray-200 bg-white p-4">
    <div class="grid grid-cols-3 gap-3 mb-3">
      <div class="text-center">
        <div class="text-lg font-bold text-blue-600">{stats.concepts}</div>
        <div class="text-xs text-gray-500">Concepts</div>
      </div>
      <div class="text-center">
        <div class="text-lg font-bold text-green-600">{stats.documents}</div>
        <div class="text-xs text-gray-500">Documents</div>
      </div>
      <div class="text-center">
        <div class="text-lg font-bold text-purple-600">{stats.conversations}</div>
        <div class="text-xs text-gray-500">Chats</div>
      </div>
    </div>
    
    <div class="text-xs text-gray-500 text-center">
      • Coherent ({Math.round($conceptMesh.length > 0 ? 80 : 0)}%) •
    </div>
  </div>
</div>

<style>
  /* Custom scrollbar for memory list */
  .overflow-y-auto::-webkit-scrollbar {
    width: 4px;
  }
  
  .overflow-y-auto::-webkit-scrollbar-track {
    background: transparent;
  }
  
  .overflow-y-auto::-webkit-scrollbar-thumb {
    background: rgba(156, 163, 175, 0.5);
    border-radius: 2px;
  }
  
  .overflow-y-auto::-webkit-scrollbar-thumb:hover {
    background: rgba(156, 163, 175, 0.8);
  }
</style>
