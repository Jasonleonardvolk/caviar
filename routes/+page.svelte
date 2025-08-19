<!-- ENHANCED CHAT WITH AUTO-SCROLL FUNCTIONALITY -->
<script lang="ts">
  import { onMount, afterUpdate, tick } from 'svelte';
  import { conceptMesh, addConceptDiff } from '$lib/stores/conceptMesh';
  import ConceptDebugPanel from '$lib/components/ConceptDebugPanel.svelte';
  
  // STEP 2: Import Enhanced API Service
  import { enhancedApiService } from '$lib/services/enhancedApi';
  
  // STEP 1-4: Import ALL systems
  let braidMemory: any = null;
  let cognitiveEngine: any = null;
  let holographicMemory: any = null;
  let ghostCollective: any = null;
  
  // Get user data from server via layout
  export let data: { user: { name: string; role: 'admin' | 'user' } | null };
  
  let mounted = false;
  let messageInput = '';
  let isTyping = false;
  let showDebugPanel = false;
  let conversationHistory: Array<{
    id: string;
    role: 'user' | 'assistant';
    content: string;
    timestamp: Date;
    concepts?: string[];
    loopId?: string;
    braidStats?: any;
    processingMethod?: string;
    confidence?: number;
    systemInsights?: string[];
    activePersona?: any;
    holographicData?: any;
    conceptNodes?: any[];
  }> = [];
  
  // ✅ AUTO-SCROLL FUNCTIONALITY
  let scrollContainer: HTMLElement;
  let isUserScrolledUp = false;
  let showScrollToBottom = false;
  let lastMessageCount = 0;
  let shouldAutoScroll = true;
  
  // Track scroll position to detect manual scrolling
  function handleScroll() {
    if (!scrollContainer) return;
    
    const { scrollTop, scrollHeight, clientHeight } = scrollContainer;
    const isAtBottom = scrollHeight - scrollTop - clientHeight < 50; // 50px threshold
    
    isUserScrolledUp = !isAtBottom;
    showScrollToBottom = isUserScrolledUp && conversationHistory.length > 0;
    shouldAutoScroll = isAtBottom;
  }
  
  // Auto-scroll to bottom function
  function scrollToBottom(force = false) {
    if (!scrollContainer) return;
    
    if (force || shouldAutoScroll) {
      scrollContainer.scrollTop = scrollContainer.scrollHeight;
      isUserScrolledUp = false;
      showScrollToBottom = false;
      shouldAutoScroll = true;
    }
  }
  
  // Auto-scroll when new messages are added
  $: if (conversationHistory.length > lastMessageCount && mounted) {
    lastMessageCount = conversationHistory.length;
    // Use tick to ensure DOM is updated before scrolling
    tick().then(() => {
      if (shouldAutoScroll) {
        scrollToBottom(false);
      } else {
        // Show scroll to bottom button if user is scrolled up
        showScrollToBottom = true;
      }
    });
  }
  
  // Force scroll to bottom (for button click)
  function forceScrollToBottom() {
    scrollToBottom(true);
  }
  
  onMount(async () => {
    mounted = true;
    
    // STEP 1-4: Load ALL cognitive systems
    try {
      const cognitive = await import('$lib/cognitive');
      braidMemory = cognitive.braidMemory;
      cognitiveEngine = cognitive.cognitiveEngine;
      holographicMemory = cognitive.holographicMemory;
      ghostCollective = cognitive.ghostCollective;
      
      console.log('🧬 ALL SYSTEMS LOADED:', {
        braidMemory: !!braidMemory,
        cognitiveEngine: !!cognitiveEngine,
        holographicMemory: !!holographicMemory,
        ghostCollective: !!ghostCollective
      });
    } catch (error) {
      console.warn('⚠️ Some cognitive systems not available:', error);
    }
    
    // STEP 2-4: Initialize Enhanced API Service
    console.log('🚀 Enhanced API Service v4.0 initialized with full system integration');
    
    // Load conversation history from localStorage
    const saved = localStorage.getItem('tori-conversation-history');
    if (saved) {
      try {
        const loadedHistory = JSON.parse(saved).map((msg: any) => ({
          ...msg,
          timestamp: new Date(msg.timestamp)
        }));
        conversationHistory = loadedHistory;
        lastMessageCount = loadedHistory.length;
        
        // Auto-scroll to bottom after loading history
        tick().then(() => scrollToBottom(true));
      } catch (e) {
        console.warn('Failed to load conversation history:', e);
      }
    }
    
    console.log('🎯 TORI main page loaded with FULL SYSTEM INTEGRATION and auto-scroll');
  });
  
  // Auto-save conversation history
  $: if (mounted && conversationHistory.length > 0) {
    localStorage.setItem('tori-conversation-history', JSON.stringify(conversationHistory));
  }
  
  async function sendMessage() {
    if (!messageInput.trim() || isTyping || !data.user) return;
    
    const userMessage = {
      id: `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      role: 'user' as const,
      content: messageInput.trim(),
      timestamp: new Date()
    };
    
    conversationHistory = [...conversationHistory, userMessage];
    const currentMessage = messageInput;
    messageInput = '';
    isTyping = true;
    
    // Auto-scroll for user's message
    shouldAutoScroll = true;
    
    try {
      // STEP 2-4: Use Enhanced API Service for ULTIMATE processing
      const context = {
        userQuery: currentMessage,
        currentConcepts: [...new Set($conceptMesh.flatMap(d => d.concepts))],
        conversationHistory: conversationHistory.slice(-10), // Last 10 messages for context
        userProfile: data.user
      };
      
      console.log('🌌👻🧬🎯 ULTIMATE: All systems processing...');
      const enhancedResponse = await enhancedApiService.generateResponse(context);
      
      const assistantMessage = {
        id: `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        role: 'assistant' as const,
        content: enhancedResponse.response,
        timestamp: new Date(),
        concepts: enhancedResponse.newConcepts,
        loopId: enhancedResponse.loopId,
        braidStats: enhancedResponse.braidMetrics,
        processingMethod: enhancedResponse.processingMethod,
        confidence: enhancedResponse.confidence,
        systemInsights: enhancedResponse.systemInsights,
        activePersona: enhancedResponse.activePersona,
        holographicData: enhancedResponse.holographicData,
        conceptNodes: enhancedResponse.conceptNodes
      };
      
      conversationHistory = [...conversationHistory, assistantMessage];
      
      // Add to concept mesh with FULL system metadata
      if (enhancedResponse.newConcepts && enhancedResponse.newConcepts.length > 0) {
        addConceptDiff({
          type: 'chat',
          title: `Ultimate AI: ${currentMessage.substring(0, 50)}...`,
          concepts: enhancedResponse.newConcepts,
          summary: `Ultimate AI processing via ${enhancedResponse.processingMethod}. Confidence: ${Math.round(enhancedResponse.confidence * 100)}%${enhancedResponse.activePersona ? ` (${enhancedResponse.activePersona.name})` : ''}${enhancedResponse.conceptNodes ? ` | ${enhancedResponse.conceptNodes.length} 3D nodes` : ''}`,
          metadata: {
            messageCount: conversationHistory.length,
            userMessage: currentMessage,
            processingMethod: enhancedResponse.processingMethod,
            confidence: enhancedResponse.confidence,
            systemInsights: enhancedResponse.systemInsights,
            loopId: enhancedResponse.loopId,
            braidMetrics: enhancedResponse.braidMetrics,
            emergentConnections: enhancedResponse.emergentConnections,
            activePersona: enhancedResponse.activePersona,
            holographicData: enhancedResponse.holographicData,
            conceptNodes: enhancedResponse.conceptNodes,
            fullSystemIntegration: true,
            timestamp: new Date()
          }
        });
      }
      
    } catch (error) {
      console.error('Ultimate AI processing failed:', error);
      
      const errorMessage = {
        id: `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        role: 'assistant' as const,
        content: "I'm having trouble with my advanced processing systems right now. Let me try a simpler approach...",
        timestamp: new Date(),
        processingMethod: 'fallback'
      };
      
      conversationHistory = [...conversationHistory, errorMessage];
    } finally {
      isTyping = false;
    }
  }
  
  function handleKeyPress(event: KeyboardEvent) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  }
  
  function clearConversation() {
    if (confirm('Clear conversation? This will remove all messages but keep your memory intact.')) {
      conversationHistory = [];
      lastMessageCount = 0;
      localStorage.removeItem('tori-conversation-history');
      showScrollToBottom = false;
      isUserScrolledUp = false;
      shouldAutoScroll = true;
    }
  }
  
  function toggleDebugPanel() {
    showDebugPanel = !showDebugPanel;
  }
  
  // Get system stats for display
  function getSystemStats() {
    const stats = {
      braid: null,
      holographic: null,
      ghost: null
    };
    
    try {
      if (braidMemory) stats.braid = braidMemory.getStats();
      if (holographicMemory) stats.holographic = holographicMemory.getVisualizationData();
      if (ghostCollective) stats.ghost = ghostCollective.getDiagnostics();
    } catch (error) {
      console.warn('Failed to get system stats:', error);
    }
    
    return stats;
  }
  
  $: systemStats = getSystemStats();
  
  // STEP 3: Get processing method icon (enhanced)
  function getProcessingIcon(method: string): string {
    switch (method) {
      case 'revolutionary_synthesis': return '🌌';
      case 'holographic_synthesis': return '🎯';
      case 'ghost_collective': return '👻';
      case 'cognitive_engine': return '🧬';
      case 'braid_memory': return '🔗';
      case 'simple': return '⚡';
      default: return '🤖';
    }
  }
  
  // STEP 3: Get processing method name (enhanced)
  function getProcessingName(method: string): string {
    switch (method) {
      case 'revolutionary_synthesis': return 'Revolutionary';
      case 'holographic_synthesis': return 'Holographic';
      case 'ghost_collective': return 'Ghost Collective';
      case 'cognitive_engine': return 'Cognitive Engine';
      case 'braid_memory': return 'BraidMemory';
      case 'simple': return 'Enhanced';
      default: return 'Standard';
    }
  }
  
  // STEP 3: Get persona icon
  function getPersonaIcon(persona: any): string {
    if (!persona) return '';
    switch (persona.id) {
      case 'scholar': return '🧠';
      case 'creator': return '🎨';
      case 'explorer': return '🔍';
      case 'mentor': return '🌟';
      case 'synthesizer': return '🔮';
      default: return '👤';
    }
  }
</script>

<svelte:head>
  <title>TORI - Consciousness Interface</title>
</svelte:head>

<!-- Main conversation interface -->
<div class="flex flex-col h-full bg-white relative">
  
  <!-- ✅ CONVERSATION AREA WITH AUTO-SCROLL -->
  <div 
    class="flex-1 overflow-y-auto px-6 py-4" 
    bind:this={scrollContainer}
    on:scroll={handleScroll}
  >
    {#if conversationHistory.length === 0}
      <!-- Welcome state -->
      <div class="flex flex-col items-center justify-center h-full text-center">
        <div class="w-16 h-16 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-2xl flex items-center justify-center mb-6">
          <span class="text-white text-2xl font-bold">T</span>
        </div>
        
        <h2 class="text-2xl font-bold text-gray-900 mb-2">
          Welcome back, {data.user?.name.split(' ')[0] || 'User'}
        </h2>
        <p class="text-gray-600 mb-6 max-w-md">
          Your TORI consciousness interface is ready with <strong>FULL SYSTEM INTEGRATION</strong>: Revolutionary AI + Ghost Collective + BraidMemory + Holographic Memory. What would you like to explore today?
        </p>
        
        <!-- Enhanced system status overview -->
        <div class="grid grid-cols-4 gap-4 mb-8">
          <div class="text-center p-3 bg-gray-50 rounded-lg">
            <div class="text-lg font-semibold text-gray-900">{$conceptMesh.length}</div>
            <div class="text-xs text-gray-600">Memory Entries</div>
          </div>
          <div class="text-center p-3 bg-green-50 rounded-lg">
            <div class="text-lg font-semibold text-green-600">👻 Personas</div>
            <div class="text-xs text-gray-600">Ghost Collective</div>
          </div>
          <div class="text-center p-3 bg-purple-50 rounded-lg">
            <div class="text-lg font-semibold text-purple-600">
              {systemStats?.braid?.totalLoops || 0}
            </div>
            <div class="text-xs text-gray-600">Cognitive Loops</div>
          </div>
          <div class="text-center p-3 bg-blue-50 rounded-lg">
            <div class="text-lg font-semibold text-blue-600">
              🎯 {systemStats?.holographic?.nodes?.length || 0}
            </div>
            <div class="text-xs text-gray-600">3D Concepts</div>
          </div>
        </div>
        
        <!-- Enhanced quick start suggestions for all systems -->
        <div class="space-y-2">
          <p class="text-sm text-gray-500 mb-3">Experience ultimate AI processing with all systems:</p>
          <div class="flex flex-wrap gap-2 justify-center">
            <button 
              class="px-3 py-1 text-sm bg-blue-50 hover:bg-blue-100 text-blue-700 rounded-full transition-colors"
              on:click={() => messageInput = "How does machine learning work? Analyze it deeply with all your systems."}
            >
              🧠 Ultimate Scholar Mode
            </button>
            <button 
              class="px-3 py-1 text-sm bg-purple-50 hover:bg-purple-100 text-purple-700 rounded-full transition-colors"
              on:click={() => messageInput = "Create an innovative visualization of AI consciousness in 3D space"}
            >
              🎨 3D Creator Mode
            </button>
            <button 
              class="px-3 py-1 text-sm bg-green-50 hover:bg-green-100 text-green-700 rounded-full transition-colors"
              on:click={() => messageInput = "How do all these cognitive systems connect and work together?"}
            >
              🔍 System Explorer
            </button>
            <button 
              class="px-3 py-1 text-sm bg-orange-50 hover:bg-orange-100 text-orange-700 rounded-full transition-colors"
              on:click={() => messageInput = "Show me my holographic memory patterns and emergent clusters"}
            >
              🎯 Holographic Memory
            </button>
          </div>
        </div>
      </div>
    {:else}
      <!-- Conversation history -->
      <div class="space-y-4 pb-4">
        {#each conversationHistory as message}
          <div class="flex {message.role === 'user' ? 'justify-end' : 'justify-start'}">
            <div class="max-w-[85%] {message.role === 'user' ? 'order-2' : 'order-1'}">
              <!-- Message bubble -->
              <div class="px-4 py-3 rounded-2xl {
                message.role === 'user' 
                  ? 'bg-blue-600 text-white' 
                  : 'bg-gray-100 text-gray-900'
              }">
                <p class="text-sm whitespace-pre-wrap">{message.content}</p>
              </div>
              
              <!-- Enhanced message metadata with all systems -->
              <div class="flex items-center space-x-2 mt-1 px-2 {message.role === 'user' ? 'justify-end' : 'justify-start'}">
                <span class="text-xs text-gray-500">
                  {message.timestamp.toLocaleTimeString()}
                </span>
                
                <!-- STEP 4: Processing method indicator -->
                {#if message.processingMethod && message.role === 'assistant'}
                  <span class="text-xs text-purple-600 bg-purple-50 px-2 py-1 rounded-full flex items-center space-x-1">
                    <span class="text-base">{getProcessingIcon(message.processingMethod)}</span>
                    <span>{getProcessingName(message.processingMethod)}</span>
                  </span>
                {/if}
                
                <!-- STEP 3: Persona indicator -->
                {#if message.activePersona && message.role === 'assistant'}
                  <span class="text-xs text-green-600 bg-green-50 px-2 py-1 rounded-full flex items-center space-x-1">
                    <span class="text-base">{getPersonaIcon(message.activePersona)}</span>
                    <span>{message.activePersona.name}</span>
                  </span>
                {/if}
                
                <!-- STEP 4: Holographic nodes indicator -->
                {#if message.conceptNodes && message.conceptNodes.length > 0 && message.role === 'assistant'}
                  <span class="text-xs text-blue-600 bg-blue-50 px-2 py-1 rounded-full flex items-center space-x-1">
                    <span class="text-base">🎯</span>
                    <span>{message.conceptNodes.length} nodes</span>
                  </span>
                {/if}
                
                <!-- STEP 2: Confidence indicator -->
                {#if message.confidence && message.role === 'assistant'}
                  <span class="text-xs text-gray-400">
                    {Math.round(message.confidence * 100)}%
                  </span>
                {/if}
                
                <!-- STEP 1: Loop ID -->
                {#if message.loopId}
                  <span class="text-xs text-purple-600 bg-purple-50 px-1 rounded">
                    🧬 {message.loopId.substring(0, 8)}
                  </span>
                {/if}
                
                {#if message.concepts && message.concepts.length > 0}
                  <div class="flex space-x-1">
                    {#each message.concepts.slice(0, 2) as concept}
                      <span class="px-1.5 py-0.5 text-xs bg-blue-100 text-blue-700 rounded">
                        {concept}
                      </span>
                    {/each}
                    {#if message.concepts.length > 2}
                      <span class="text-xs text-gray-400">+{message.concepts.length - 2}</span>
                    {/if}
                  </div>
                {/if}
                
                <!-- STEP 1: Braid stats -->
                {#if message.braidStats}
                  <span class="text-xs text-gray-400">
                    {message.braidStats.crossings} crossings
                  </span>
                {/if}
              </div>
              
              <!-- STEP 2-4: System insights (expandable) -->
              {#if message.systemInsights && message.systemInsights.length > 0}
                <details class="mt-1 px-2">
                  <summary class="text-xs text-gray-400 cursor-pointer hover:text-gray-600">
                    System insights ({message.systemInsights.length})
                  </summary>
                  <div class="mt-1 text-xs text-gray-500 space-y-1">
                    {#each message.systemInsights as insight}
                      <div>• {insight}</div>
                    {/each}
                  </div>
                </details>
              {/if}
            </div>
            
            <!-- Avatar -->
            <div class="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 {
              message.role === 'user' 
                ? 'bg-blue-600 text-white order-1 ml-3' 
                : 'bg-gray-300 text-gray-600 order-2 mr-3'
            }">
              {#if message.role === 'user' && data.user}
                {data.user.name.charAt(0).toUpperCase()}
              {:else if message.role === 'user'}
                👤
              {:else}
                {#if message.activePersona}
                  {getPersonaIcon(message.activePersona)}
                {:else if message.processingMethod}
                  {getProcessingIcon(message.processingMethod)}
                {:else}
                  🤖
                {/if}
              {/if}
            </div>
          </div>
        {/each}
        
        <!-- Enhanced typing indicator -->
        {#if isTyping}
          <div class="flex justify-start">
            <div class="w-8 h-8 rounded-full bg-gradient-to-br from-purple-400 to-blue-500 text-white flex items-center justify-center flex-shrink-0 mr-3">
              🌌
            </div>
            <div class="bg-gray-100 px-4 py-3 rounded-2xl max-w-[80%]">
              <div class="flex space-x-1 items-center">
                <span class="text-xs text-gray-600 mr-2">
                  Ultimate AI processing - All systems active
                </span>
                <div class="w-2 h-2 bg-purple-400 rounded-full animate-bounce"></div>
                <div class="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style="animation-delay: 0.1s"></div>
                <div class="w-2 h-2 bg-green-400 rounded-full animate-bounce" style="animation-delay: 0.2s"></div>
                <div class="w-2 h-2 bg-orange-400 rounded-full animate-bounce" style="animation-delay: 0.3s"></div>
              </div>
            </div>
          </div>
        {/if}
      </div>
    {/if}
  </div>
  
  <!-- ✅ SCROLL TO BOTTOM BUTTON (appears when user scrolls up) -->
  {#if showScrollToBottom}
    <div class="absolute bottom-24 right-6 z-10">
      <button
        on:click={forceScrollToBottom}
        class="bg-blue-600 hover:bg-blue-700 text-white p-3 rounded-full shadow-lg transition-all duration-200 flex items-center space-x-2"
        title="Scroll to bottom"
      >
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 14l-7 7m0 0l-7-7m7 7V3" />
        </svg>
        <span class="text-xs">New messages</span>
      </button>
    </div>
  {/if}
  
  <!-- Input area -->
  <div class="border-t border-gray-200 bg-white px-6 py-4">
    <div class="flex items-end space-x-3">
      <!-- Message input -->
      <div class="flex-1">
        <textarea
          bind:value={messageInput}
          on:keydown={handleKeyPress}
          placeholder="Ask anything - Ultimate AI with all systems ready... (Enter to send)"
          class="w-full px-4 py-3 border border-gray-300 rounded-2xl resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          rows="1"
          style="min-height: 44px; max-height: 120px;"
          disabled={isTyping}
        ></textarea>
      </div>
      
      <!-- Send button -->
      <button
        on:click={sendMessage}
        disabled={!messageInput.trim() || isTyping}
        class="px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 disabled:bg-gray-300 disabled:cursor-not-allowed text-white rounded-2xl transition-all font-medium"
      >
        {#if isTyping}
          <div class="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
        {:else}
          Send
        {/if}
      </button>
      
      <!-- Debug toggle button -->
      <button
        on:click={toggleDebugPanel}
        class="px-4 py-3 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-2xl transition-colors"
        title="Toggle concept debug panel"
      >
        🧠
      </button>
      
      <!-- Clear conversation button -->
      {#if conversationHistory.length > 0}
        <button
          on:click={clearConversation}
          class="px-4 py-3 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-2xl transition-colors"
          title="Clear conversation"
        >
          🗑️
        </button>
      {/if}
    </div>
    
    <!-- Enhanced input hints -->
    <div class="flex items-center justify-between mt-2 px-1">
      <div class="text-xs text-gray-500">
        🌌 Revolutionary • 👻 Ghost Collective • 🧬 BraidMemory • 🎯 Holographic Memory • 🚀 Ultimate AI
        {#if isUserScrolledUp}
          • ⬆️ Scroll position preserved
        {:else}
          • ⬇️ Auto-scroll active
        {/if}
        {#if showDebugPanel}
          • 🧠 Debug panel active
        {/if}
      </div>
      
      <div class="text-xs text-gray-500">
        {conversationHistory.length} messages • {$conceptMesh.length} concepts
        {#if systemStats?.braid?.totalLoops}
          • {systemStats.braid.totalLoops} loops
        {/if}
        {#if systemStats?.holographic?.nodes?.length}
          • {systemStats.holographic.nodes.length} 3D nodes
        {/if}
      </div>
    </div>
  </div>
  
  <!-- 🧠 CONCEPT DEBUG PANEL (collapsible) -->
  {#if showDebugPanel}
    <div class="border-t border-gray-200 bg-gray-50 p-4 max-h-96 overflow-y-auto">
      <div class="flex items-center justify-between mb-3">
        <h3 class="text-sm font-semibold text-gray-700 flex items-center space-x-2">
          <span>🧠</span>
          <span>Concept Debug Panel</span>
          <span class="text-xs text-gray-500">({$conceptMesh.length} entries)</span>
        </h3>
        <button
          on:click={toggleDebugPanel}
          class="text-gray-400 hover:text-gray-600 text-sm"
        >
          ✕
        </button>
      </div>
      
      <ConceptDebugPanel />
    </div>
  {/if}
</div>

<style>
  textarea {
    field-sizing: content;
  }
  
  @keyframes bounce {
    0%, 80%, 100% {
      transform: scale(0);
    }
    40% {
      transform: scale(1);
    }
  }
  
  .animate-bounce {
    animation: bounce 1.4s infinite ease-in-out both;
  }
  
  details summary {
    list-style: none;
  }
  
  details summary::-webkit-details-marker {
    display: none;
  }
  
  /* ✅ Smooth scrolling behavior */
  .flex-1.overflow-y-auto {
    scroll-behavior: smooth;
  }
  
  /* ✅ Custom scrollbar styling */
  .flex-1.overflow-y-auto::-webkit-scrollbar {
    width: 6px;
  }
  
  .flex-1.overflow-y-auto::-webkit-scrollbar-track {
    background: #f1f5f9;
  }
  
  .flex-1.overflow-y-auto::-webkit-scrollbar-thumb {
    background: #cbd5e1;
    border-radius: 3px;
  }
  
  .flex-1.overflow-y-auto::-webkit-scrollbar-thumb:hover {
    background: #94a3b8;
  }
  
  /* ✅ Debug panel scrollbar styling */
  .max-h-96.overflow-y-auto::-webkit-scrollbar {
    width: 4px;
  }
  
  .max-h-96.overflow-y-auto::-webkit-scrollbar-track {
    background: #e5e7eb;
  }
  
  .max-h-96.overflow-y-auto::-webkit-scrollbar-thumb {
    background: #9ca3af;
    border-radius: 2px;
  }
  
  .max-h-96.overflow-y-auto::-webkit-scrollbar-thumb:hover {
    background: #6b7280;
  }
</style>