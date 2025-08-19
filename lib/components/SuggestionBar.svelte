<!-- components/SuggestionBar.svelte -->
<script lang="ts">
  import { conceptMesh } from '$lib/stores/conceptMesh';
  import { ghostPersona } from '$lib/stores/ghostPersona';
  import { userSession } from '$lib/stores/user';
  import { derived } from 'svelte/store';
  
  // Dynamic suggestions based on context
  const contextualSuggestions = derived(
    [conceptMesh, ghostPersona, userSession],
    ([$concepts, $ghost, $user]) => {
      const suggestions: Array<{text: string, action: string, icon: string}> = [];
      
      // Base suggestions always available
      suggestions.push(
        { text: "Help me understand", action: "help", icon: "💡" },
        { text: "What can you do?", action: "capabilities", icon: "🤖" }
      );
      
      // Document-based suggestions
      const documents = $concepts.filter(c => c.type === 'document');
      if (documents.length > 0) {
        suggestions.push(
          { text: "Summarize documents", action: "summarize", icon: "📄" },
          { text: "Find connections", action: "connect", icon: "🔗" }
        );
      }
      
      // Concept-based suggestions
      const allConcepts = $concepts.flatMap(c => c.concepts);
      const uniqueConcepts = [...new Set(allConcepts)];
      
      if (uniqueConcepts.length > 3) {
        suggestions.push(
          { text: "Explore concept map", action: "conceptmap", icon: "🗺️" },
          { text: "Analyze patterns", action: "patterns", icon: "📊" }
        );
      }
      
      // Mood-based suggestions
      if ($ghost.mood === 'helpful') {
        suggestions.push(
          { text: "Guide me through this", action: "guide", icon: "🧭" }
        );
      } else if ($ghost.mood === 'curious') {
        suggestions.push(
          { text: "What questions should I ask?", action: "questions", icon: "❓" }
        );
      }
      
      // User activity-based suggestions
      if ($user?.isAuthenticated) {
        const stats = $user.user?.stats;
        if (stats && stats.documentsUploaded > 0 && stats.conversationsCount === 0) {
          suggestions.push(
            { text: "Let's discuss your documents", action: "discuss", icon: "💬" }
          );
        }
      }
      
      // Recent concept suggestions
      if ($concepts.length > 0) {
        const recentConcept = $concepts[$concepts.length - 1];
        if (recentConcept.concepts.length > 0) {
          const concept = recentConcept.concepts[0];
          suggestions.push(
            { text: `Tell me more about ${concept}`, action: `explain:${concept}`, icon: "🔍" }
          );
        }
      }
      
      return suggestions.slice(0, 6); // Limit to 6 suggestions
    }
  );
  
  let showAllSuggestions = false;
  let visibleCount = 4;
  
  function handleSuggestionClick(suggestion: {text: string, action: string, icon: string}) {
    const action = suggestion.action;
    let message = "";
    
    switch (action) {
      case "help":
        message = "I need help understanding something. Can you guide me?";
        break;
      case "capabilities":
        message = "What can you do to help me?";
        break;
      case "summarize":
        message = "Can you summarize the documents I've uploaded?";
        break;
      case "connect":
        message = "Show me connections between the concepts in my documents.";
        break;
      case "conceptmap":
        message = "I'd like to explore the concept map of my knowledge.";
        break;
      case "patterns":
        message = "Help me analyze patterns in my data and concepts.";
        break;
      case "guide":
        message = "Please guide me through this step by step.";
        break;
      case "questions":
        message = "What questions should I be asking about this topic?";
        break;
      case "discuss":
        message = "Let's discuss the documents I've uploaded.";
        break;
      default:
        if (action.startsWith("explain:")) {
          const concept = action.split(":")[1];
          message = `Tell me more about ${concept} and how it relates to my other concepts.`;
        } else {
          message = suggestion.text;
        }
    }
    
    // Dispatch custom event to parent (ChatPanel can listen for this)
    const event = new CustomEvent('suggestion-selected', {
      detail: { message, action, suggestion },
      bubbles: true
    });
    
    document.dispatchEvent(event);
  }
  
  function toggleShowAll() {
    showAllSuggestions = !showAllSuggestions;
    visibleCount = showAllSuggestions ? $contextualSuggestions.length : 4;
  }
</script>

<div class="bg-white border-t border-gray-200 px-4 py-3">
  <!-- Header -->
  <div class="flex items-center justify-between mb-2">
    <h3 class="text-xs font-semibold text-gray-600 uppercase tracking-wide">
      💡 Intelligent Suggestions
    </h3>
    <div class="flex items-center space-x-2 text-xs text-gray-500">
      <span>Ghost Mode: {$ghostPersona.mood}</span>
      {#if $contextualSuggestions.length > 4}
        <button 
          class="text-blue-600 hover:text-blue-800 font-medium"
          on:click={toggleShowAll}
        >
          {showAllSuggestions ? 'Show Less' : `+${$contextualSuggestions.length - 4} More`}
        </button>
      {/if}
    </div>
  </div>
  
  <!-- Suggestions grid -->
  <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2">
    {#each $contextualSuggestions.slice(0, visibleCount) as suggestion, i}
      <button 
        class="flex items-center space-x-2 text-left text-xs bg-gray-50 hover:bg-blue-50 
               border border-gray-200 hover:border-blue-300 rounded-lg px-3 py-2 
               transition-all duration-200 group"
        on:click={() => handleSuggestionClick(suggestion)}
      >
        <span class="text-base group-hover:scale-110 transition-transform duration-200">
          {suggestion.icon}
        </span>
        <span class="text-gray-700 group-hover:text-blue-700 font-medium flex-1">
          {suggestion.text}
        </span>
      </button>
    {/each}
  </div>
  
  <!-- Context indicators -->
  <div class="mt-3 flex items-center justify-between text-xs text-gray-400">
    <div class="flex items-center space-x-4">
      <span class="flex items-center space-x-1">
        <span class="w-2 h-2 bg-blue-400 rounded-full"></span>
        <span>{$conceptMesh.length} concepts</span>
      </span>
      <span class="flex items-center space-x-1">
        <span class="w-2 h-2 bg-green-400 rounded-full"></span>
        <span>{$conceptMesh.filter(c => c.type === 'document').length} documents</span>
      </span>
      <span class="flex items-center space-x-1">
        <span class="w-2 h-2 bg-purple-400 rounded-full"></span>
        <span>{$conceptMesh.filter(c => c.type === 'chat').length} conversations</span>
      </span>
    </div>
    
    <!-- System status -->
    <div class="flex items-center space-x-2">
      <span>ELFIN++ Ready</span>
      <div class="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
    </div>
  </div>
</div>

<style>
  /* Enhanced hover effects */
  button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  }
  
  /* Staggered animation for suggestions */
  button {
    animation: suggestion-appear 0.3s ease-out;
    animation-delay: calc(var(--index, 0) * 0.1s);
    animation-fill-mode: both;
  }
  
  @keyframes suggestion-appear {
    from {
      opacity: 0;
      transform: translateY(10px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }
</style>