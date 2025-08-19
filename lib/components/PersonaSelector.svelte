<!-- 4D Persona Selector - Revolutionary cognitive dimension interface -->
<script lang="ts">
  import { ghostPersona, setPersona, setMood } from '$lib/stores/ghostPersona';
  import { userSession } from '$lib/stores/user';
  import { fade, scale } from 'svelte/transition';
  
  // 4D Persona definitions with cognitive coordinates
  const personas = [
    {
      name: 'Mentor',
      ψ: 'socratic',        // Cognitive mode: questioning and guiding
      ε: [0.7, 0.8, 0.2],   // Emotional palette: [calm, warm, urgent]
      τ: 0.6,              // Temporal bias: balanced between recent and timeless
      φ: 42,               // Phase seed: stable foundational frequency
      color: '#4f46e5',    // Indigo
      description: 'Guides through questions and frameworks',
      mood: 'contemplative'
    },
    {
      name: 'Scholar',
      ψ: 'analytical',     // Cognitive mode: systematic analysis
      ε: [0.9, 0.3, 0.4],   // Emotional palette: [calm, analytical focus, moderate urgency]
      τ: 0.3,              // Temporal bias: favors timeless knowledge
      φ: 137,              // Phase seed: golden ratio frequency
      color: '#059669',    // Emerald
      description: 'Deep research and systematic analysis',
      mood: 'focused'
    },
    {
      name: 'Explorer',
      ψ: 'lateral',        // Cognitive mode: lateral thinking
      ε: [0.2, 0.9, 0.7],   // Emotional palette: [adventurous, curious, excited]
      τ: 0.8,              // Temporal bias: loves cutting-edge and novel
      φ: 89,               // Phase seed: prime number for uniqueness
      color: '#dc2626',    // Red
      description: 'Discovers connections and new territories',
      mood: 'curious'
    },
    {
      name: 'Architect',
      ψ: 'structural',     // Cognitive mode: systems thinking
      ε: [0.8, 0.2, 0.5],   // Emotional palette: [methodical, precise, determined]
      τ: 0.4,              // Temporal bias: long-term planning focus
      φ: 21,               // Phase seed: Fibonacci number for structure
      color: '#7c3aed',    // Violet
      description: 'Builds frameworks and systematic structures',
      mood: 'methodical'
    },
    {
      name: 'Creator',
      ψ: 'intuitive',      // Cognitive mode: intuitive leaps
      ε: [0.3, 0.8, 0.9],   // Emotional palette: [inspired, imaginative, passionate]
      τ: 0.7,              // Temporal bias: creates for the future
      φ: 73,               // Phase seed: creative prime frequency
      color: '#ea580c',    // Orange
      description: 'Synthesizes novel ideas and innovations',
      mood: 'inspired'
    }
  ];
  
  let selectedPersona = $ghostPersona.activePersona || 'Mentor';
  let showDetails = false;
  
  // Get current persona object
  $: currentPersona = personas.find(p => p.name === selectedPersona) || personas[0];
  
  function selectPersona(personaName: string) {
    selectedPersona = personaName;
    const persona = personas.find(p => p.name === personaName);
    if (persona) {
      setPersona(personaName);
      setMood(persona.mood);
      console.log(`🧠 4D Persona activated: ${personaName} [ψ:${persona.ψ}, ε:[${persona.ε.join(',')}], τ:${persona.τ}, φ:${persona.φ}]`);
    }
  }
  
  function formatCoordinates(persona: any): string {
    return `ψ:${persona.ψ} ε:[${persona.ε.map(v => v.toFixed(1)).join(',')}] τ:${persona.τ} φ:${persona.φ}`;
  }
  
  // Calculate phase indicator position (visual representation of φ)
  function getPhaseIndicatorStyle(φ: number): string {
    const angle = (φ / 180) * Math.PI;
    const x = 50 + 40 * Math.cos(angle);
    const y = 50 + 40 * Math.sin(angle);
    return `left: ${x}%; top: ${y}%;`;
  }
  
  // Get emotional palette gradient
  function getEmotionalGradient(ε: number[]): string {
    const [calm, warm, urgent] = ε;
    return `linear-gradient(45deg, 
      rgba(59, 130, 246, ${calm}) 0%, 
      rgba(34, 197, 94, ${warm}) 50%, 
      rgba(239, 68, 68, ${urgent}) 100%)`;
  }
</script>

<div class="bg-white border-b border-gray-200 p-4">
  <div class="flex items-center justify-between mb-3">
    <h3 class="text-sm font-semibold text-gray-800">4D Personas</h3>
    <button 
      class="text-xs px-2 py-1 rounded bg-purple-100 hover:bg-purple-200 text-purple-700 transition-colors"
      on:click={() => showDetails = !showDetails}
    >
      {showDetails ? 'Hide' : 'Coords'}
    </button>
  </div>
  
  <!-- Active Persona Display -->
  <div class="mb-4 p-3 rounded-lg border-2 border-gray-100 bg-gray-50">
    <div class="flex items-center space-x-3">
      <!-- Persona orb with phase visualization -->
      <div class="relative w-12 h-12 rounded-full border-2 flex items-center justify-center text-white text-sm font-bold"
           style="background-color: {currentPersona.color}; border-color: {currentPersona.color}">
        {currentPersona.name.charAt(0)}
        
        <!-- Phase indicator -->
        <div class="absolute w-2 h-2 bg-white rounded-full opacity-80"
             style="{getPhaseIndicatorStyle(currentPersona.φ)}">
        </div>
      </div>
      
      <div class="flex-1 min-w-0">
        <div class="flex items-center space-x-2">
          <h4 class="text-sm font-medium text-gray-900">{currentPersona.name}</h4>
          <span class="text-xs px-2 py-0.5 rounded-full bg-blue-100 text-blue-700">
            {$ghostPersona.mood || currentPersona.mood}
          </span>
        </div>
        <p class="text-xs text-gray-600 mt-1">{currentPersona.description}</p>
        
        <!-- 4D Coordinates (when details shown) -->
        {#if showDetails}
          <div class="text-xs text-gray-500 mt-1 font-mono" transition:fade={{duration: 200}}>
            {formatCoordinates(currentPersona)}
          </div>
        {/if}
      </div>
      
      <!-- Emotional palette indicator -->
      <div class="w-3 h-8 rounded-full border border-gray-300"
           style="background: {getEmotionalGradient(currentPersona.ε)}">
      </div>
    </div>
  </div>
  
  <!-- Persona Grid Selector -->
  <div class="grid grid-cols-5 gap-2">
    {#each personas as persona}
      <button
        class="relative p-2 rounded-lg border-2 transition-all hover:scale-105 {
          selectedPersona === persona.name 
            ? 'border-current ring-2 ring-purple-200' 
            : 'border-gray-200 hover:border-gray-300'
        }"
        style="color: {persona.color}"
        on:click={() => selectPersona(persona.name)}
        title="{persona.name}: {persona.description}"
      >
        <!-- Persona avatar -->
        <div class="w-6 h-6 rounded-full mx-auto mb-1 flex items-center justify-center text-white text-xs font-bold"
             style="background-color: {persona.color}">
          {persona.name.charAt(0)}
        </div>
        
        <!-- Name -->
        <div class="text-xs text-center text-gray-700 truncate">
          {persona.name}
        </div>
        
        <!-- Active indicator -->
        {#if selectedPersona === persona.name}
          <div class="absolute -top-1 -right-1 w-3 h-3 bg-purple-500 rounded-full border-2 border-white"
               transition:scale={{duration: 200}}>
          </div>
        {/if}
        
        <!-- Subtle phase pulse for active persona -->
        {#if selectedPersona === persona.name}
          <div class="absolute inset-0 rounded-lg border-2 border-purple-300 animate-pulse opacity-50">
          </div>
        {/if}
      </button>
    {/each}
  </div>
  
  <!-- 4D Visualization Panel (when details shown) -->
  {#if showDetails}
    <div class="mt-4 p-3 bg-gray-50 rounded-lg border border-gray-200" transition:fade={{duration: 300}}>
      <h5 class="text-xs font-medium text-gray-700 mb-2">4D Cognitive Coordinates</h5>
      
      <div class="grid grid-cols-2 gap-3 text-xs">
        <!-- Cognitive Mode (ψ) -->
        <div>
          <span class="text-gray-600">ψ (Thinking):</span>
          <span class="font-mono text-blue-700 ml-1">{currentPersona.ψ}</span>
        </div>
        
        <!-- Emotional Palette (ε) -->
        <div>
          <span class="text-gray-600">ε (Emotion):</span>
          <span class="font-mono text-green-700 ml-1">[{currentPersona.ε.map(v => v.toFixed(1)).join(',')}]</span>
        </div>
        
        <!-- Temporal Bias (τ) -->
        <div>
          <span class="text-gray-600">τ (Time):</span>
          <span class="font-mono text-orange-700 ml-1">{currentPersona.τ}</span>
        </div>
        
        <!-- Phase Seed (φ) -->
        <div>
          <span class="text-gray-600">φ (Phase):</span>
          <span class="font-mono text-purple-700 ml-1">{currentPersona.φ}</span>
        </div>
      </div>
      
      <!-- Phase visualization -->
      <div class="mt-3">
        <div class="text-xs text-gray-600 mb-1">Phase Oscillator:</div>
        <div class="relative w-full h-8 bg-gray-200 rounded-lg overflow-hidden">
          <div class="absolute inset-0 flex items-center justify-center">
            <div class="w-2 h-2 rounded-full animate-pulse"
                 style="background-color: {currentPersona.color}; animation-duration: {1000/currentPersona.φ}ms;">
            </div>
          </div>
        </div>
      </div>
    </div>
  {/if}
  
  <!-- Authentication prompt (if not signed in) -->
  {#if !$userSession.isAuthenticated}
    <div class="mt-3 p-2 bg-blue-50 border border-blue-200 rounded-lg text-center">
      <p class="text-xs text-blue-700">Sign in to save persona preferences</p>
    </div>
  {/if}
</div>

<style>
  @keyframes phase-pulse {
    0%, 100% { opacity: 0.6; transform: scale(1); }
    50% { opacity: 1; transform: scale(1.2); }
  }
  
  .animate-phase {
    animation: phase-pulse 2s ease-in-out infinite;
  }
</style>