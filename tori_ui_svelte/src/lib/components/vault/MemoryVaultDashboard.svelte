<!-- Enhanced Memory Vault Dashboard Component -->
<script context="module" lang="ts">
  export type MemoryType = 'soliton' | 'concept' | 'ghost' | 'document' | 'chat' | 'memory' | 'phase' | 'quantum' | 'user';
  
  export interface MemoryEntry {
    id: string;
    type: MemoryType;
    timestamp: Date;
    content: any;
    phase?: string;
    coherence?: number;
    importance?: number;
    metadata?: Record<string, any>;
    tags?: string[];
    relationships?: string[];
    source?: string;
  }
  
  export interface MemoryStats {
    total: number;
    byType: Record<MemoryType, number>;
    recentActivity: number;
    storageUsed: string;
  }
</script>

<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { writable, derived } from 'svelte/store';
  import { page } from '$app/stores';
  import { browser } from '$app/environment';
  import { 
    solitonMemory, 
    memoryStats, 
    currentPhase, 
    phaseCoherence,
    quantumState,
    isLearning,
    conceptMesh
  } from '$lib/stores/solitonMemory';
  import { concepts as conceptStore } from '$lib/stores/conceptMesh';
  import { ghostPersona } from '$lib/stores/ghostPersona';
  
  // Create local systemCoherence store if not imported
  const systemCoherence = writable(0.85);
  
    
  interface VaultMetrics {
    totalMemories: number;
    activePhases: number;
    coherenceLevel: number;
    quantumEntanglement: number;
    conceptDensity: number;
    ghostActivity: number;
  }
  
  interface PhaseTransition {
    from: string;
    to: string;
    timestamp: Date;
    trigger: string;
    coherenceDelta: number;
  }
  
  // Component state
  const memories = writable<MemoryEntry[]>([]);
  const metrics = writable<VaultMetrics>({
    totalMemories: 0,
    activePhases: 0,
    coherenceLevel: 0,
    quantumEntanglement: 0,
    conceptDensity: 0,
    ghostActivity: 0
  });
  const phaseTransitions = writable<PhaseTransition[]>([]);
  const memoryGraph = writable<{ nodes: any[]; edges: any[] }>({ nodes: [], edges: [] });
  
  const searchQuery = writable('');
  const filterType = writable<'all' | MemoryType>('all');
  
  const filteredMemories = derived(
    [memories, searchQuery, filterType],
    ([$memories, $search, $filter]) => {
      const query = $search.trim().toLowerCase();
      return $memories.filter((mem) => {
        const okType = $filter === 'all' || mem.type === $filter;
        const okText = !query || 
          mem.tags?.some(t => t.toLowerCase().includes(query)) ||
          JSON.stringify(mem.content || {}).toLowerCase().includes(query);
        return okType && okText;
      }).sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
    }
  );
  
  // View state
  let selectedView: 'overview' | 'timeline' | 'graph' | 'quantum' | 'export' = 'overview';
  let selectedMemory: MemoryEntry | null = null;
  let autoRefresh = false;
  let isExporting = false;
  let refreshInterval: ReturnType<typeof setInterval> | undefined;
  
  // Store update functions
  function updateFromSoliton(solitonData: any) {
    if (!solitonData) return;
    
    const newMemory: MemoryEntry = {
      id: `soliton-${Date.now()}`,
      timestamp: new Date(),
      type: 'soliton',
      content: solitonData.currentPhase,
      phase: solitonData.phase || 'unknown',
      coherence: solitonData.coherence || 0.5,
      tags: ['soliton', 'phase'],
      relationships: [],
      importance: 0.7
    };
    
    memories.update(m => [...m, newMemory]);
  }
  
  function updateFromConcepts(conceptData: any) {
    if (!conceptData?.activeConcepts) return;
    
    conceptData.activeConcepts.forEach((concept: any) => {
      const newMemory: MemoryEntry = {
        id: `concept-${concept.id}-${Date.now()}`,
        timestamp: new Date(),
        type: 'concept',
        content: concept,
        phase: $currentPhase || 'exploration',
        coherence: concept.weight || 0.5,
        tags: ['concept', ...concept.tags || []],
        relationships: concept.related || [],
        importance: concept.importance || 0.5
      };
      
      memories.update(m => [...m, newMemory]);
    });
  }
  
  function updateFromGhost(ghostData: any) {
    if (!ghostData?.currentGhost) return;
    
    const newMemory: MemoryEntry = {
      id: `ghost-${Date.now()}`,
      timestamp: new Date(),
      type: 'ghost',
      content: {
        persona: ghostData.currentGhost,
        confidence: ghostData.confidence,
        message: ghostData.lastMessage
      },
      phase: ghostData.phase || 'emergence',
      coherence: ghostData.confidence || 0.5,
      tags: ['ghost', ghostData.currentGhost],
      relationships: [],
      importance: 0.8
    };
    
    memories.update(m => [...m, newMemory]);
  }
  
  // Data loading functions
  async function loadMemoryVault() {
    try {
      const response = await fetch('/api/memory/vault');
      if (response.ok) {
        const data = await response.json();
        processMemoryData(data);
      }
    } catch (error) {
  const msg = error instanceof Error ? error.message : String(error);
      console.error('Failed to load memory vault:', error);
      loadFromLocalStorage();
    
}
  }
  
  function processMemoryData(data: any) {
    if (!data) return;
    
    const processedMemories = (data.memories || []).map((m: any) => ({
      ...m,
      timestamp: new Date(m.timestamp)
    }));
    memories.set(processedMemories);
    
    metrics.set({
      totalMemories: processedMemories.length,
      activePhases: new Set(processedMemories.map((m: any) => m.phase)).size,
      coherenceLevel: data.coherence || 0.5,
      quantumEntanglement: data.entanglement || 0,
      conceptDensity: data.conceptDensity || 0,
      ghostActivity: data.ghostActivity || 0
    });
    
    if (data.transitions) {
      phaseTransitions.set(data.transitions.map((t: any) => ({
        ...t,
        timestamp: new Date(t.timestamp)
      })));
    }
    
    // Build memory graph
    const nodes = new Map<string, any>();
    const edges: any[] = [];
    
    processedMemories.forEach((memory: MemoryEntry) => {
      nodes.set(memory.id, {
        id: memory.id,
        label: memory.type,
        importance: memory.importance,
        phase: memory.phase
      });
      
      memory.relationships.forEach((relId: string) => {
        edges.push({
          from: memory.id,
          to: relId,
          coherence: memory.coherence
        });
      });
    });
    
    memoryGraph.set({ nodes: Array.from(nodes.values()), edges });
  }
  
  function loadFromLocalStorage() {
    try {
      const stored = localStorage.getItem('memory-vault');
      if (stored) {
        processMemoryData(JSON.parse(stored));
      }
    } catch (error) {
  const msg = error instanceof Error ? error.message : String(error);
      console.error('Failed to load from localStorage:', error);
    
}
  }
  
  // Lifecycle
  onMount(() => {
    if (!browser) return;
    
    const unsubscribes: Array<() => void> = [];
    
    const run = async () => {
      try {
        await loadMemoryVault();
        
        if (autoRefresh) {
          try { 
            if (refreshInterval) clearInterval(refreshInterval); 
          } catch {}
          refreshInterval = setInterval(() => { 
            try { 
              loadMemoryVault(); 
            } catch {} 
          }, 5000);
        }
        
        // Only subscribe if the stores exist at runtime
        try {
          if (solitonMemory?.subscribe) {
            unsubscribes.push(solitonMemory.subscribe(updateFromSoliton));
          }
          if (conceptStore?.subscribe) {
            unsubscribes.push(conceptStore.subscribe(updateFromConcepts));
          }
          if (ghostPersona?.subscribe) {
            unsubscribes.push(ghostPersona.subscribe(updateFromGhost));
          }
        } catch (err) {
  const msg = err instanceof Error ? err.message : String(err);
          console.error('Error subscribing to stores:', err);
        
}
      } catch (err) {
  const msg = err instanceof Error ? err.message : String(err);
        console.error('Error in onMount:', err);
      
}
    };
    
    run();
    
    return () => {
      try { 
        unsubscribes.forEach(u => { 
          try { 
            u(); 
          } catch {} 
        }); 
      } catch {}
      try { 
        if (refreshInterval) clearInterval(refreshInterval); 
      } catch {}
    };
  });
  
  onDestroy(() => {
    if (refreshInterval) {
      clearInterval(refreshInterval);
    }
  });
  
  // Actions
  async function exportVault() {
    isExporting = true;
    
    try {
      const exportData = {
        timestamp: new Date().toISOString(),
        memories: $memories,
        metrics: $metrics,
        transitions: $phaseTransitions,
        metadata: {
          version: '1.0.0',
          totalMemories: $memories.length,
          dateRange: {
            start: $memories[0]?.timestamp,
            end: $memories[$memories.length - 1]?.timestamp
          }
        }
      };
      
      const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `memory-vault-${new Date().toISOString().split('T')[0]}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (error) {
  const msg = error instanceof Error ? error.message : String(error);
      console.error('Export failed:', error);
    
} finally {
      isExporting = false;
    }
  }
  
  function clearVault() {
    if (confirm('Are you sure you want to clear all memories? This cannot be undone.')) {
      memories.set([]);
      phaseTransitions.set([]);
      localStorage.removeItem('memory-vault');
    }
  }
  
  function selectMemory(memory: MemoryEntry) {
    selectedMemory = memory;
  }
  
  function closeMemoryDetail() {
    selectedMemory = null;
  }
  
  // Utility functions
  function getTypeIcon(type: string): string {
    const icons: Record<string, string> = {
      concept: '🧠',
      phase: '🌀',
      quantum: '⚛️',
      ghost: '👻',
      user: '👤',
      soliton: '🌊',
      document: '📄',
      chat: '💬',
      memory: '💭'
    };
    return icons[type] || '📝';
  }
  
  function getPhaseColor(phase: string): string {
    const colors: Record<string, string> = {
      exploration: 'text-blue-600',
      consolidation: 'text-green-600',
      integration: 'text-purple-600',
      transformation: 'text-orange-600',
      emergence: 'text-pink-600'
    };
    return colors[phase] || 'text-gray-600';
  }
  
  function formatTimestamp(date: Date): string {
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    
    if (diff < 60000) return 'Just now';
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
    return date.toLocaleDateString();
  }
</script>

<div class="memory-vault-dashboard h-full flex flex-col bg-gray-50">
  <!-- Header -->
  <div class="bg-white border-b border-gray-200 px-6 py-4">
    <div class="flex justify-between items-center">
      <div>
        <h1 class="text-2xl font-bold text-gray-900">Memory Vault</h1>
        <p class="text-sm text-gray-600 mt-1">
          Quantum consciousness storage and retrieval system
        </p>
      </div>
      
      <div class="flex items-center gap-4">
        <!-- Auto-refresh toggle -->
        <label class="flex items-center gap-2">
          <input
            type="checkbox"
            bind:checked={autoRefresh}
            class="rounded border-gray-300"
          />
          <span class="text-sm text-gray-700">Auto-refresh</span>
        </label>
        
        <!-- Export button -->
        <button
          on:click={exportVault}
          disabled={isExporting}
          class="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
        >
          {isExporting ? 'Exporting...' : 'Export Vault'}
        </button>
        
        <!-- Clear button -->
        <button
          on:click={clearVault}
          class="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700"
        >
          Clear Vault
        </button>
      </div>
    </div>
    
    <!-- View tabs -->
    <div class="flex gap-2 mt-4">
      {#each ['overview', 'timeline', 'graph', 'quantum', 'export'] as view}
        <button
          on:click={() => selectedView = view}
          class="px-4 py-2 rounded-lg transition-colors {selectedView === view ? 'bg-blue-100 text-blue-700' : 'text-gray-600 hover:bg-gray-100'}"
        >
          {view.charAt(0).toUpperCase() + view.slice(1)}
        </button>
      {/each}
    </div>
  </div>
  
  <!-- Metrics Bar -->
  <div class="bg-white border-b border-gray-200 px-6 py-3">
    <div class="grid grid-cols-6 gap-4">
      <div class="text-center">
        <div class="text-2xl font-bold text-gray-900">{$metrics.totalMemories}</div>
        <div class="text-xs text-gray-600">Total Memories</div>
      </div>
      <div class="text-center">
        <div class="text-2xl font-bold text-blue-600">{$metrics.activePhases}</div>
        <div class="text-xs text-gray-600">Active Phases</div>
      </div>
      <div class="text-center">
        <div class="text-2xl font-bold text-green-600">
          {Math.round($metrics.coherenceLevel * 100)}%
        </div>
        <div class="text-xs text-gray-600">Coherence</div>
      </div>
      <div class="text-center">
        <div class="text-2xl font-bold text-purple-600">
          {Math.round($systemCoherence * 100)}%
        </div>
        <div class="text-xs text-gray-600">System Coherence</div>
      </div>
      <div class="text-center">
        <div class="text-2xl font-bold text-orange-600">
          {Math.round($metrics.conceptDensity * 100)}%
        </div>
        <div class="text-xs text-gray-600">Concept Density</div>
      </div>
      <div class="text-center">
        <div class="text-2xl font-bold text-pink-600">
          {Math.round($metrics.ghostActivity * 100)}%
        </div>
        <div class="text-xs text-gray-600">Ghost Activity</div>
      </div>
    </div>
  </div>
  
  <!-- Search and filters -->
  <div class="bg-white border-b border-gray-200 px-6 py-3">
    <div class="flex gap-4">
      <input
        type="text"
        bind:value={$searchQuery}
        placeholder="Search memories..."
        class="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
      />
      
      <select
        bind:value={$filterType}
        class="px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
      >
        <option value="all">All Types</option>
        <option value="concept">Concepts</option>
        <option value="phase">Phases</option>
        <option value="quantum">Quantum</option>
        <option value="ghost">Ghost</option>
        <option value="user">User</option>
        <option value="soliton">Soliton</option>
        <option value="document">Document</option>
        <option value="chat">Chat</option>
        <option value="memory">Memory</option>
      </select>
    </div>
  </div>
  
  <!-- Main content area -->
  <div class="flex-1 overflow-auto p-6">
    {#if selectedView === 'overview'}
      <!-- Overview Grid -->
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <!-- Recent Memories -->
        <div class="bg-white rounded-lg border border-gray-200 p-6">
          <h2 class="text-lg font-semibold text-gray-900 mb-4">Recent Memories</h2>
          <div class="space-y-3 max-h-96 overflow-y-auto">
            {#each $filteredMemories.slice(0, 10) as memory}
              <button
                on:click={() => selectMemory(memory)}
                class="w-full text-left p-3 rounded-lg border border-gray-200 hover:border-blue-300 hover:shadow-sm transition-all"
              >
                <div class="flex items-start gap-3">
                  <span class="text-2xl">{getTypeIcon(memory.type)}</span>
                  <div class="flex-1">
                    <div class="flex justify-between items-start">
                      <span class="font-medium text-gray-900">{memory.type}</span>
                      <span class="text-xs text-gray-500">{formatTimestamp(memory.timestamp)}</span>
                    </div>
                    <div class="text-sm text-gray-600 mt-1 line-clamp-2">
                      {JSON.stringify(memory.content).slice(0, 100)}...
                    </div>
                    <div class="flex gap-2 mt-2">
                      <span class="text-xs px-2 py-1 bg-gray-100 text-gray-700 rounded">
                        {memory.phase}
                      </span>
                      <span class="text-xs px-2 py-1 bg-blue-100 text-blue-700 rounded">
                        {Math.round(memory.coherence * 100)}% coherent
                      </span>
                    </div>
                  </div>
                </div>
              </button>
            {/each}
          </div>
        </div>
        
        <!-- Phase Transitions -->
        <div class="bg-white rounded-lg border border-gray-200 p-6">
          <h2 class="text-lg font-semibold text-gray-900 mb-4">Phase Transitions</h2>
          <div class="space-y-3 max-h-96 overflow-y-auto">
            {#each $phaseTransitions as transition}
              <div class="p-3 rounded-lg border border-gray-200">
                <div class="flex items-center justify-between">
                  <div class="flex items-center gap-2">
                    <span class={getPhaseColor(transition.from)}>{transition.from}</span>
                    <span class="text-gray-400">→</span>
                    <span class={getPhaseColor(transition.to)}>{transition.to}</span>
                  </div>
                  <span class="text-xs text-gray-500">
                    {formatTimestamp(transition.timestamp)}
                  </span>
                </div>
                <div class="text-sm text-gray-600 mt-2">
                  Trigger: {transition.trigger}
                </div>
                <div class="text-sm text-gray-600">
                  Coherence Δ: {transition.coherenceDelta > 0 ? '+' : ''}{(transition.coherenceDelta * 100).toFixed(1)}%
                </div>
              </div>
            {/each}
          </div>
        </div>
      </div>
      
    {:else if selectedView === 'timeline'}
      <!-- Timeline View -->
      <div class="bg-white rounded-lg border border-gray-200 p-6">
        <h2 class="text-lg font-semibold text-gray-900 mb-4">Memory Timeline</h2>
        <div class="space-y-4">
          {#each $filteredMemories as memory}
            <button
              on:click={() => selectMemory(memory)}
              class="w-full text-left p-4 rounded-lg border border-gray-200 hover:border-blue-300 hover:shadow-sm transition-all"
            >
              <div class="flex items-start gap-4">
                <div class="text-3xl">{getTypeIcon(memory.type)}</div>
                <div class="flex-1">
                  <div class="flex justify-between items-start mb-2">
                    <div>
                      <span class="font-medium text-gray-900">{memory.type}</span>
                      <span class="ml-2 text-sm {getPhaseColor(memory.phase)}">{memory.phase}</span>
                    </div>
                    <span class="text-sm text-gray-500">{formatTimestamp(memory.timestamp)}</span>
                  </div>
                  <div class="text-sm text-gray-700 mb-2">
                    {JSON.stringify(memory.content).slice(0, 200)}...
                  </div>
                  <div class="flex flex-wrap gap-2">
                    {#each memory.tags as tag}
                      <span class="text-xs px-2 py-1 bg-gray-100 text-gray-700 rounded">
                        {tag}
                      </span>
                    {/each}
                  </div>
                  <div class="flex items-center gap-4 mt-2 text-xs text-gray-600">
                    <span>Coherence: {Math.round(memory.coherence * 100)}%</span>
                    <span>Importance: {Math.round(memory.importance * 100)}%</span>
                    <span>Relations: {memory.relationships.length}</span>
                  </div>
                </div>
              </div>
            </button>
          {/each}
        </div>
      </div>
      
    {:else if selectedView === 'graph'}
      <!-- Graph View -->
      <div class="bg-white rounded-lg border border-gray-200 p-6">
        <h2 class="text-lg font-semibold text-gray-900 mb-4">Memory Graph</h2>
        <div class="text-center py-12">
          <div class="text-4xl mb-4">🕸️</div>
          <div class="text-gray-600">
            Interactive memory relationship graph
          </div>
          <div class="text-sm text-gray-500 mt-2">
            {$memoryGraph.nodes.length} nodes, {$memoryGraph.edges.length} edges
          </div>
        </div>
      </div>
      
    {:else if selectedView === 'quantum'}
      <!-- Quantum View -->
      <div class="bg-white rounded-lg border border-gray-200 p-6">
        <h2 class="text-lg font-semibold text-gray-900 mb-4">Quantum State</h2>
        <div class="grid grid-cols-2 gap-6">
          <div>
            <h3 class="font-medium text-gray-900 mb-3">Quantum Metrics</h3>
            <div class="space-y-3">
              <div class="flex justify-between">
                <span class="text-gray-600">Entanglement:</span>
                <span class="font-medium">{Math.round($metrics.quantumEntanglement * 100)}%</span>
              </div>
              <div class="flex justify-between">
                <span class="text-gray-600">Superposition:</span>
                <span class="font-medium">{Math.round(($quantumState?.superposition || 0) * 100)}%</span>
              </div>
              <div class="flex justify-between">
                <span class="text-gray-600">Decoherence Rate:</span>
                <span class="font-medium">{($quantumState?.decoherence || 0).toFixed(3)}</span>
              </div>
            </div>
          </div>
          
          <div>
            <h3 class="font-medium text-gray-900 mb-3">Phase Space</h3>
            <div class="h-48 bg-gradient-to-br from-purple-100 to-blue-100 rounded-lg flex items-center justify-center">
              <div class="text-center">
                <div class="text-3xl mb-2">⚛️</div>
                <div class="text-sm text-gray-600">Quantum visualization</div>
              </div>
            </div>
          </div>
        </div>
      </div>
      
    {:else if selectedView === 'export'}
      <!-- Export View -->
      <div class="bg-white rounded-lg border border-gray-200 p-6">
        <h2 class="text-lg font-semibold text-gray-900 mb-4">Export & Import</h2>
        <div class="space-y-6">
          <div>
            <h3 class="font-medium text-gray-900 mb-3">Export Options</h3>
            <div class="space-y-3">
              <button
                on:click={exportVault}
                class="w-full px-4 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
              >
                Export Full Vault (JSON)
              </button>
              <button class="w-full px-4 py-3 bg-gray-600 text-white rounded-lg hover:bg-gray-700">
                Export as CSV
              </button>
              <button class="w-full px-4 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700">
                Export Quantum State
              </button>
            </div>
          </div>
          
          <div>
            <h3 class="font-medium text-gray-900 mb-3">Import</h3>
            <div class="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
              <div class="text-3xl mb-2">📁</div>
              <div class="text-gray-600">Drop files here or click to browse</div>
              <input type="file" accept=".json" class="hidden" />
            </div>
          </div>
        </div>
      </div>
    {/if}
  </div>
  
  <!-- Memory Detail Modal -->
  {#if selectedMemory}
    <div class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div class="bg-white rounded-lg max-w-2xl w-full max-h-[80vh] overflow-hidden">
        <div class="p-6 border-b border-gray-200">
          <div class="flex justify-between items-start">
            <div class="flex items-center gap-3">
              <span class="text-3xl">{getTypeIcon(selectedMemory.type)}</span>
              <div>
                <h2 class="text-xl font-semibold text-gray-900">
                  {selectedMemory.type.charAt(0).toUpperCase() + selectedMemory.type.slice(1)} Memory
                </h2>
                <p class="text-sm text-gray-600">{selectedMemory.timestamp.toLocaleString()}</p>
              </div>
            </div>
            <button
              on:click={closeMemoryDetail}
              class="p-2 hover:bg-gray-100 rounded-lg"
            >
              ✕
            </button>
          </div>
        </div>
        
        <div class="p-6 overflow-y-auto max-h-[60vh]">
          <div class="space-y-4">
            <div>
              <h3 class="font-medium text-gray-900 mb-2">Content</h3>
              <pre class="bg-gray-50 p-4 rounded-lg text-sm overflow-x-auto">
{JSON.stringify(selectedMemory.content, null, 2)}
              </pre>
            </div>
            
            <div class="grid grid-cols-2 gap-4">
              <div>
                <h3 class="font-medium text-gray-900 mb-2">Metadata</h3>
                <div class="space-y-2 text-sm">
                  <div class="flex justify-between">
                    <span class="text-gray-600">Phase:</span>
                    <span class={getPhaseColor(selectedMemory.phase)}>{selectedMemory.phase}</span>
                  </div>
                  <div class="flex justify-between">
                    <span class="text-gray-600">Coherence:</span>
                    <span>{Math.round(selectedMemory.coherence * 100)}%</span>
                  </div>
                  <div class="flex justify-between">
                    <span class="text-gray-600">Importance:</span>
                    <span>{Math.round(selectedMemory.importance * 100)}%</span>
                  </div>
                </div>
              </div>
              
              <div>
                <h3 class="font-medium text-gray-900 mb-2">Tags</h3>
                <div class="flex flex-wrap gap-1">
                  {#each selectedMemory.tags as tag}
                    <span class="text-xs px-2 py-1 bg-blue-100 text-blue-700 rounded">
                      {tag}
                    </span>
                  {/each}
                </div>
              </div>
            </div>
            
            {#if selectedMemory.relationships.length > 0}
              <div>
                <h3 class="font-medium text-gray-900 mb-2">Relationships</h3>
                <div class="space-y-1">
                  {#each selectedMemory.relationships as rel}
                    <div class="text-sm text-gray-600">→ {rel}</div>
                  {/each}
                </div>
              </div>
            {/if}
          </div>
        </div>
      </div>
    </div>
  {/if}
</div>

<style>
  .line-clamp-2 {
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }
</style>
