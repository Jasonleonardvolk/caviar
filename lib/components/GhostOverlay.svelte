<!-- components/GhostOverlay.svelte -->
<script lang="ts">
  import { ghostPersona } from '$lib/stores/ghostPersona';
  import { derived } from 'svelte/store';
  import { onMount } from 'svelte';
  
  // Derive visual properties from ghostPersona
  const glowColor = derived(ghostPersona, $gp => {
    // Choose a glow color based on mood (enhanced for light theme)
    switch ($gp.mood) {
      case 'calm': return 'rgba(79, 70, 229, 0.15)';      // soft indigo
      case 'helpful': return 'rgba(16, 185, 129, 0.15)';   // emerald
      case 'focused': return 'rgba(245, 158, 11, 0.15)';   // amber
      case 'pleased': return 'rgba(236, 72, 153, 0.15)';   // pink
      case 'alert': return 'rgba(239, 68, 68, 0.15)';      // red
      case 'curious': return 'rgba(168, 85, 247, 0.15)';   // purple
      default: return 'rgba(79, 70, 229, 0.1)';            // default indigo
    }
  });
  
  const pulseSpeed = derived(ghostPersona, $gp => {
    // Pulse speed based on stability
    const baseSpeed = 3;
    const speedMultiplier = 1 + (1 - $gp.stability);
    return baseSpeed * speedMultiplier;
  });
  
  let avatarElement: HTMLDivElement;
  let showTooltip = false;
  
  onMount(() => {
    // Subtle entrance animation
    if (avatarElement) {
      avatarElement.style.transform = 'scale(0) rotate(180deg)';
      avatarElement.style.opacity = '0';
      
      setTimeout(() => {
        avatarElement.style.transition = 'all 0.8s cubic-bezier(0.34, 1.56, 0.64, 1)';
        avatarElement.style.transform = 'scale(1) rotate(0deg)';
        avatarElement.style.opacity = '0.9';
      }, 500);
    }
  });
  
  function getPersonaEmoji(persona: string): string {
    switch (persona.toLowerCase()) {
      case 'mentor': return '🧙‍♂️';
      case 'scholar': return '📚';
      case 'debugger': return '🔧';
      case 'explorer': return '🔍';
      case 'creator': return '🎨';
      case 'analyst': return '📊';
      default: return '👻';
    }
  }
  
  function getPersonaInitial(persona: string): string {
    return persona.charAt(0).toUpperCase();
  }
  
  function handleAvatarClick() {
    showTooltip = !showTooltip;
    setTimeout(() => showTooltip = false, 3000);
  }
</script>

<!-- Full-screen ghost overlay (non-interactive except for avatar) -->
<div class="pointer-events-none fixed inset-0 z-10">
  <!-- Subtle aura background that pulses with ghost state -->
  <div 
    class="absolute inset-0 transition-all duration-1000"
    style="background: radial-gradient(circle at 75% 25%, {$glowColor}, transparent 60%);
           animation: ghost-pulse {$pulseSpeed}s ease-in-out infinite;">
  </div>
  
  <!-- Secondary aura for depth -->
  <div 
    class="absolute inset-0 transition-all duration-1000"
    style="background: radial-gradient(circle at 25% 75%, {$glowColor}, transparent 70%);
           animation: ghost-pulse-offset {$pulseSpeed * 1.3}s ease-in-out infinite;">
  </div>
  
  <!-- Ghost avatar in bottom-right corner -->
  <div class="absolute bottom-6 right-6 pointer-events-auto">
    <!-- Tooltip -->
    {#if showTooltip}
      <div class="absolute bottom-full right-0 mb-2 bg-gray-900 text-white text-xs 
                  rounded-lg px-3 py-2 whitespace-nowrap animate-fade-in">
        <div class="font-medium">{$ghostPersona.persona} Ghost</div>
        <div class="text-gray-300">Mood: {$ghostPersona.mood}</div>
        <div class="text-gray-300">Stability: {($ghostPersona.stability * 100).toFixed(0)}%</div>
        <!-- Tooltip arrow -->
        <div class="absolute top-full right-4 w-0 h-0 border-l-4 border-r-4 border-t-4 
                    border-transparent border-t-gray-900"></div>
      </div>
    {/if}
    
    <!-- Avatar container with enhanced styling -->
    <div 
      bind:this={avatarElement}
      class="relative w-14 h-14 rounded-full cursor-pointer
             bg-gradient-to-br from-white to-gray-100
             border-2 border-gray-200 shadow-lg
             hover:shadow-xl hover:scale-105
             transition-all duration-300"
      on:click={handleAvatarClick}
      on:keydown={(e) => e.key === 'Enter' && handleAvatarClick()}
      role="button"
      tabindex="0"
      title="Ghost AI Status">
      
      <!-- Stability ring -->
      <div class="absolute inset-0 rounded-full">
        <svg class="w-full h-full -rotate-90" viewBox="0 0 36 36">
          <!-- Background circle -->
          <path class="text-gray-300" stroke="currentColor" stroke-width="2" fill="none"
                d="M 18,18 m -16,0 a 16,16 0 1,1 0,32 a 16,16 0 1,1 0,-32"></path>
          <!-- Progress circle -->
          <path class="text-blue-500 transition-all duration-500" stroke="currentColor" 
                stroke-width="2" fill="none" stroke-linecap="round"
                stroke-dasharray="{$ghostPersona.stability * 100}, 100"
                d="M 18,18 m -16,0 a 16,16 0 1,1 0,32 a 16,16 0 1,1 0,-32"></path>
        </svg>
      </div>
      
      <!-- Persona avatar -->
      <div class="absolute inset-2 rounded-full bg-gradient-to-br from-blue-50 to-indigo-100
                  flex items-center justify-center text-lg border border-blue-200">
        {#if $ghostPersona.persona}
          <span class="filter drop-shadow-sm">
            {getPersonaEmoji($ghostPersona.persona)}
          </span>
        {:else}
          <span class="text-blue-600 font-bold text-sm">
            {getPersonaInitial($ghostPersona.persona || 'G')}
          </span>
        {/if}
      </div>
      
      <!-- Mood indicator dot -->
      <div class="absolute -top-1 -right-1 w-4 h-4 rounded-full border-2 border-white
                  {$ghostPersona.mood === 'calm' ? 'bg-blue-400' : 
                   $ghostPersona.mood === 'helpful' ? 'bg-green-400' :
                   $ghostPersona.mood === 'focused' ? 'bg-yellow-400' :
                   $ghostPersona.mood === 'pleased' ? 'bg-pink-400' :
                   $ghostPersona.mood === 'alert' ? 'bg-red-400' :
                   $ghostPersona.mood === 'curious' ? 'bg-purple-400' : 'bg-gray-400'}
                  animate-pulse"></div>
    </div>
    
    <!-- Activity indicator -->
    <div class="absolute -bottom-1 -left-1 w-3 h-3 bg-green-400 rounded-full border-2 border-white
                animate-pulse shadow-sm"></div>
  </div>
</div>

<style>
  @keyframes ghost-pulse {
    0%, 100% { opacity: 0.3; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(1.02); }
  }
  
  @keyframes ghost-pulse-offset {
    0%, 100% { opacity: 0.2; transform: scale(1.01); }
    50% { opacity: 0.4; transform: scale(1.03); }
  }
  
  /* Ensure smooth transitions for all ghost states */
  .absolute {
    transition: all 0.3s ease-in-out;
  }
</style>