<!-- components/MemoryDrawer.svelte (Complete) -->
<script lang="ts">
  import UploadPanel from '$lib/components/UploadPanel.svelte';
  import DocumentSummary from '$lib/components/DocumentSummary.svelte';
  import { conceptMesh, clearConceptMesh, removeConceptDiff } from '$lib/stores/conceptMesh';
  import { userSession } from '$lib/stores/user';
  import { ghostPersona } from '$lib/stores/ghostPersona';
  
  // Drawer state
  let expanded = true;
  let activeTab = 'all'; // 'all', 'documents', 'conversations', 'concepts'
  
  // Filtered content based on active tab
  $: filteredContent = $conceptMesh.filter(diff => {
    switch (activeTab) {
      case 'documents':
        return diff.type === 'document';
      case 'conversations':
        return diff.type === 'chat';
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
  
  function toggleDrawer() {
    expanded = !expanded;
  }
  
  function handleClearMemory() {
    if (confirm('Clear all memory? This will remove all documents, conversations, and concepts.')) {
      clearConceptMesh();
    }
  }
  
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

<!-- Memory Drawer Container -->
<div class="flex flex-col h-full bg-gray-50 border-l border-gray-200 transition-all duration-300"
     style="width: {expanded ? '360px' : '40px'}">
  
  <!-- Header with toggle -->
  <div class="flex items-center justify-between p-2 bg-white border-b border-gray-200">
    {#if expanded}
      <div class="flex items-center space-x-2">
        <h2 class="text-sm font-semibold text-gray-800">🧠 Memory System</h2>
        <div class="text-xs bg-blue-100 text-blue-700 px-2 py-0.5 rounded-full">
          {stats.total}
        </div>
      </div>
    {/if}
    
    <button 
      class="p-1 hover:bg-gray-100 rounded transition-colors"
      on:click={toggleDrawer} 
      aria-label="Toggle memory drawer"
    >
      <span class="text-lg">
        {expanded ? '🔽' : '🔼'}
      </span>
    </button>
  </div>
  
  {#if expanded}
    <!-- User info (if logged in) -->
    {#if $userSession.isAuthenticated && $userSession.user}
      <div class="p-3 bg-blue-50 border-b border-blue-200">
        <div class="text-xs font-medium text-blue-800">
          👤 {$userSession.user.name}
        </div>
        <div class="text-xs text-blue-600 mt-1">
          {$userSession.user.stats.documentsUploaded} docs • 
          {$userSession.user.stats.conversationsCount} chats • 
          {stats.concepts} concepts
        </div>
      </div>
    {/if}
    
    <!-- Upload panel at top -->
    <div class="p-3 border-b border-gray-200">
      <UploadPanel />
    </div>
    
    <!-- Memory navigation tabs -->
    <div class="flex border-b border-gray-200 bg-white">
      <button 
        class="flex-1 px-3 py-2 text-xs font-medium transition-colors
               {activeTab === 'all' ? 'text-blue-600 border-b-2 border-blue-600 bg-blue-50' : 'text-gray-600 hover:text-gray-800'}"
        on:click={() => activeTab = 'all'}
      >
        All ({stats.total})
      </button>
      <button 
        class="flex-1 px-3 py-2 text-xs font-medium transition-colors
               {activeTab === 'documents' ? 'text-green-600 border-b-2 border-green-600 bg-green-50' : 'text-gray-600 hover:text-gray-800'}"
        on:click={() => activeTab = 'documents'}
      >
        📄 ({stats.documents})
      </button>
      <button 
        class="flex-1 px-3 py-2 text-xs font-medium transition-colors
               {activeTab === 'conversations' ? 'text-purple-600 border-b-2 border-purple-600 bg-purple-50' : 'text-gray-600 hover:text-gray-800'}"
        on:click={() => activeTab = 'conversations'}
      >
        💬 ({stats.conversations})
      </button>
    </div>
    
    <!-- Memory content list -->
    <div class="flex-1 overflow-y-auto p-3 space-y-2">
      {#if filteredContent.length === 0}
        <div class="text-center py-8 text-gray-500">
          <div class="text-2xl mb-2">
            {#if activeTab === 'documents'}
              📄
            {:else if activeTab === 'conversations'}
              💬
            {:else}
              🧠
            {/if}
          </div>
          <p class="text-sm">
            {#if activeTab === 'documents'}
              No documents uploaded yet
            {:else if activeTab === 'conversations'}
              No conversations recorded
            {:else}
              No memory entries yet
            {/if}
          </p>
          <p class="text-xs mt-1 text-gray-400">
            {#if activeTab === 'documents'}
              Upload files using the panel above
            {:else if activeTab === 'conversations'}
              Start chatting to create memories
            {:else}
              Upload documents or start conversations
            {/if}
          </p>
        </div>
      {:else}
        {#each filteredContent.slice().reverse() as diff (diff.id)}
          {#if diff.type === 'document'}
            <!-- Document entry -->
            <DocumentSummary {diff} />
          {:else if diff.type === 'chat'}
            <!-- Chat conversation entry -->
            <div class="bg-white rounded-md shadow-sm border border-gray-200 p-3 hover:shadow-md transition-shadow">
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
          {:else if diff.type === 'memory'}
            <!-- General memory entry -->
            <div class="bg-white rounded-md shadow-sm border border-gray-200 p-3 hover:shadow-md transition-shadow">
              <div class="flex items-start justify-between">
                <div class="flex-1 min-w-0">
                  <h3 class="text-sm font-medium text-gray-800 mb-1 flex items-center gap-2">
                    <span class="text-blue-600">🧠</span>
                    <span>{diff.title}</span>
                  </h3>
                  
                  <p class="text-xs text-gray-500 mb-2">
                    {formatRelativeTime(diff.timestamp)}
                  </p>
                  
                  {#if diff.summary}
                    <p class="text-xs text-gray-700 mb-2">{diff.summary}</p>
                  {/if}
                  
                  <!-- Concepts -->
                  {#if diff.concepts && diff.concepts.length > 0}
                    <div class="flex flex-wrap gap-1">
                      {#each diff.concepts as concept}
                        <span class="inline-block bg-blue-100 text-blue-700 text-xs 
                                     rounded-full px-2 py-0.5 border border-blue-200">
                          {concept}
                        </span>
                      {/each}
                    </div>
                  {/if}
                </div>
                
                <button 
                  class="text-xs text-red-500 hover:text-red-700 p-1 ml-2"
                  on:click={() => removeConceptDiff(diff.id)}
                  title="Remove memory"
                >
                  ✕
                </button>
              </div>
            </div>
          {/if}
        {/each}
      {/if}
    </div>
    
    <!-- Memory management footer -->
    <div class="border-t border-gray-200 bg-white p-3">
      <!-- System info -->
      <div class="flex items-center justify-between text-xs text-gray-500 mb-2">
        <span>Ghost: {$ghostPersona.persona}</span>
        <span>Stability: {($ghostPersona.stability * 100).toFixed(0)}%</span>
      </div>
      
      <!-- Quick stats -->
      <div class="grid grid-cols-3 gap-2 mb-3">
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
      
      <!-- Actions -->
      <div class="flex gap-2">
        <button 
          class="flex-1 tori-button-secondary text-xs px-2 py-1"
          on:click={() => activeTab = 'all'}
        >
          🗺️ View All
        </button>
        <button 
          class="tori-button-secondary text-xs px-2 py-1 text-red-600"
          on:click={handleClearMemory}
          title="Clear all memory"
        >
          🗑️
        </button>
      </div>
    </div>
  {/if}
</div>

<style>
  /* Smooth tab transitions */
  button {
    transition: all 0.2s ease-in-out;
  }
  
  /* Enhanced scrollbar for memory list */
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