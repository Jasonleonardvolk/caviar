<!-- STEP 4: Enhanced ThoughtspaceRenderer with Holographic Memory -->
<script lang="ts">
  import { onMount } from 'svelte';
  import HolographicVisualization from './HolographicVisualization.svelte';
  
  let mounted = false;
  let showVisualization = true;
  let holographicMemory: any = null;
  
  // Stats for display
  let stats = {
    nodes: 0,
    connections: 0,
    clusters: 0,
    activationWave: false
  };
  
  onMount(async () => {
    mounted = true;
    
    // Load holographic memory system
    try {
      const cognitive = await import('$lib/cognitive');
      holographicMemory = cognitive.holographicMemory;
      
      // Subscribe to updates
      if (holographicMemory) {
        holographicMemory.onUpdate((data: any) => {
          stats = {
            nodes: data.nodes.length,
            connections: data.connections.length,
            clusters: data.clusters.length,
            activationWave: !!data.activationWave
          };
        });
        
        // Initial data load
        const initialData = holographicMemory.getVisualizationData();
        stats = {
          nodes: initialData.nodes.length,
          connections: initialData.connections.length,
          clusters: initialData.clusters.length,
          activationWave: !!initialData.activationWave
        };
      }
      
      console.log('🎯 ThoughtspaceRenderer initialized with holographic memory');
    } catch (error) {
      console.warn('Holographic memory not available:', error);
    }
  });
  
  function createTestNetwork() {
    if (!holographicMemory) return;
    
    const concepts = ['Innovation', 'Technology', 'Future', 'Intelligence', 'Connection'];
    const nodes: any[] = [];
    
    // Create concept nodes
    concepts.forEach((concept, i) => {
      const node = holographicMemory.createConceptNode(concept, 0.6 + Math.random() * 0.4);
      nodes.push(node);
    });
    
    // Create connections
    for (let i = 0; i < nodes.length - 1; i++) {
      holographicMemory.createConnection(
        nodes[i].id,
        nodes[i + 1].id,
        0.4 + Math.random() * 0.6,
        'semantic'
      );
    }
    
    // Activate random nodes
    setTimeout(() => {
      const randomNode = nodes[Math.floor(Math.random() * nodes.length)];
      holographicMemory.activateConcept(randomNode.id, 0.8);
    }, 1000);
    
    setTimeout(() => {
      const randomNode = nodes[Math.floor(Math.random() * nodes.length)];
      holographicMemory.activateConcept(randomNode.id, 0.6);
    }, 2000);
  }
  
  function clearMemory() {
    if (holographicMemory) {
      holographicMemory.clear();
    }
  }
  
  function detectClusters() {
    if (holographicMemory) {
      const clusters = holographicMemory.detectEmergentClusters();
      console.log(`🌌 Detected ${clusters.length} emergent clusters`);
    }
  }
</script>

<div class="flex flex-col h-full bg-gray-50">
  <!-- Header -->
  <div class="flex-shrink-0 p-4 bg-white border-b border-gray-200">
    <div class="flex items-center justify-between">
      <div>
        <h3 class="text-lg font-semibold text-gray-900">Holographic Memory</h3>
        <p class="text-sm text-gray-600">3D Concept Visualization</p>
      </div>
      
      <div class="flex items-center space-x-2">
        <button
          on:click={() => showVisualization = !showVisualization}
          class="px-3 py-1 text-xs bg-gray-100 hover:bg-gray-200 text-gray-700 rounded transition-colors"
        >
          {showVisualization ? 'Hide' : 'Show'}
        </button>
      </div>
    </div>
  </div>
  
  <!-- Stats Panel -->
  <div class="flex-shrink-0 p-4 bg-white border-b border-gray-200">
    <div class="grid grid-cols-2 gap-4">
      <div class="text-center">
        <div class="text-2xl font-bold text-blue-600">{stats.nodes}</div>
        <div class="text-xs text-gray-600">Concept Nodes</div>
      </div>
      
      <div class="text-center">
        <div class="text-2xl font-bold text-purple-600">{stats.connections}</div>
        <div class="text-xs text-gray-600">Connections</div>
      </div>
      
      <div class="text-center">
        <div class="text-2xl font-bold text-green-600">{stats.clusters}</div>
        <div class="text-xs text-gray-600">Clusters</div>
      </div>
      
      <div class="text-center">
        <div class="text-2xl font-bold {stats.activationWave ? 'text-orange-500' : 'text-gray-400'}">
          {stats.activationWave ? '🌊' : '💤'}
        </div>
        <div class="text-xs text-gray-600">Activity</div>
      </div>
    </div>
  </div>
  
  <!-- Visualization Area -->
  {#if showVisualization}
    <div class="flex-1 p-4">
      <div class="h-full bg-black rounded-lg relative overflow-hidden">
        {#if mounted && holographicMemory}
          <HolographicVisualization />
        {:else}
          <div class="flex items-center justify-center h-full text-white">
            <div class="text-center">
              <div class="text-4xl mb-4">🌌</div>
              <div class="text-lg font-semibold mb-2">Holographic Memory Loading</div>
              <div class="text-sm text-gray-400">Initializing 3D consciousness space...</div>
            </div>
          </div>
        {/if}
      </div>
    </div>
  {:else}
    <div class="flex-1 flex items-center justify-center text-gray-500">
      <div class="text-center">
        <div class="text-4xl mb-4">👁️</div>
        <div class="text-lg font-semibold">Visualization Hidden</div>
        <div class="text-sm">Click "Show" to view the 3D holographic memory</div>
      </div>
    </div>
  {/if}
  
  <!-- Controls Panel -->
  <div class="flex-shrink-0 p-4 bg-white border-t border-gray-200">
    <div class="space-y-2">
      <div class="text-xs font-semibold text-gray-700 mb-2">Memory Controls</div>
      
      <div class="grid grid-cols-1 gap-2">
        <button
          on:click={createTestNetwork}
          class="px-3 py-2 bg-blue-50 hover:bg-blue-100 text-blue-700 rounded text-sm transition-colors"
          disabled={!holographicMemory}
        >
          🧪 Create Test Network
        </button>
        
        <button
          on:click={detectClusters}
          class="px-3 py-2 bg-green-50 hover:bg-green-100 text-green-700 rounded text-sm transition-colors"
          disabled={!holographicMemory || stats.nodes < 3}
        >
          🔍 Detect Clusters
        </button>
        
        <button
          on:click={clearMemory}
          class="px-3 py-2 bg-red-50 hover:bg-red-100 text-red-700 rounded text-sm transition-colors"
          disabled={!holographicMemory}
        >
          🗑️ Clear Memory
        </button>
      </div>
    </div>
  </div>
  
  <!-- Status Indicator -->
  <div class="flex-shrink-0 p-2 bg-gray-100">
    <div class="flex items-center justify-between text-xs text-gray-600">
      <div class="flex items-center space-x-2">
        <div class="w-2 h-2 rounded-full {holographicMemory ? 'bg-green-400' : 'bg-red-400'}"></div>
        <span>{holographicMemory ? 'Holographic Memory Active' : 'System Loading'}</span>
      </div>
      
      <div>
        3D Consciousness Space
      </div>
    </div>
  </div>
</div>
