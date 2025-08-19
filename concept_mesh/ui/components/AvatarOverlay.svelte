<script>
  import { createEventDispatcher } from 'svelte';
  
  // Props
  export let user = null;
  export let personaId = null;
  export let size = 'medium'; // small, medium, large
  export let position = 'bottom-right'; // top-left, top-right, bottom-left, bottom-right
  export let showTooltip = true;
  export let interactive = true;
  
  // Optional props
  export let timestamp = null;
  export let conceptId = null;
  
  // Local state
  let hover = false;
  
  // Event dispatcher
  const dispatch = createEventDispatcher();
  
  // Computed sizes
  $: avatarSize = size === 'small' ? 24 : (size === 'large' ? 40 : 32);
  $: fontSize = size === 'small' ? '0.7rem' : (size === 'large' ? '1rem' : '0.85rem');
  $: borderWidth = size === 'small' ? 1 : (size === 'large' ? 3 : 2);
  
  // Computed position
  $: positionClass = `position-${position}`;
  
  // Click handler
  function handleClick(event) {
    if (!interactive) return;
    
    dispatch('click', {
      user,
      personaId,
      conceptId,
      timestamp,
      originalEvent: event
    });
    
    event.stopPropagation();
  }
  
  // Get persona display name
  function getPersonaName(personaId) {
    if (!personaId) return 'Unknown';
    
    switch (personaId) {
      case 'creative_agent': return 'Creative Agent';
      case 'glyphsmith': return 'Glyphsmith';
      case 'memory_pruner': return 'Memory Pruner';
      case 'researcher': return 'Researcher';
      default: return personaId.charAt(0).toUpperCase() + personaId.slice(1).replace('_', ' ');
    }
  }
  
  // Get avatar initials
  function getInitials(name) {
    if (!name) return '?';
    return name.split(' ').map(n => n[0]).join('').toUpperCase();
  }
  
  // Get persona color
  function getPersonaColor(personaId) {
    if (!personaId) return 'var(--avatar-bg, #4a9eff)';
    
    switch (personaId) {
      case 'creative_agent': return 'var(--creative-color, #9c27b0)';
      case 'glyphsmith': return 'var(--glyphsmith-color, #3f51b5)';
      case 'memory_pruner': return 'var(--pruner-color, #009688)';
      case 'researcher': return 'var(--researcher-color, #ff5722)';
      default: return 'var(--avatar-bg, #4a9eff)';
    }
  }
  
  // Format timestamp
  function formatTimestamp(timestamp) {
    if (!timestamp) return '';
    
    const date = new Date(timestamp);
    return date.toLocaleString();
  }
</script>

<div 
  class="avatar-overlay {positionClass} {interactive ? 'interactive' : ''}"
  style="--avatar-size: {avatarSize}px; --font-size: {fontSize}; --border-width: {borderWidth}px; --persona-color: {getPersonaColor(personaId)}"
  on:click={handleClick}
  on:mouseenter={() => hover = true}
  on:mouseleave={() => hover = false}
>
  <div class="avatar" style="background-color: {getPersonaColor(personaId)}">
    {#if user && user.avatar_url}
      <img src={user.avatar_url} alt={user.name || 'User'} />
    {:else}
      <div class="initials">
        {user ? getInitials(user.name || 'U') : '?'}
      </div>
    {/if}
  </div>
  
  {#if showTooltip && hover}
    <div class="tooltip">
      <div class="tooltip-content">
        <div class="tooltip-header">
          <span class="name">{user ? (user.name || 'Unknown User') : 'Unknown User'}</span>
          <span class="persona">{getPersonaName(personaId)}</span>
        </div>
        
        {#if timestamp}
          <div class="timestamp">{formatTimestamp(timestamp)}</div>
        {/if}
        
        {#if conceptId}
          <div class="concept-id">{conceptId}</div>
        {/if}
      </div>
    </div>
  {/if}
</div>

<style>
  .avatar-overlay {
    position: absolute;
    z-index: 10;
    width: var(--avatar-size);
    height: var(--avatar-size);
  }
  
  .position-top-left {
    top: calc(-1 * var(--avatar-size) / 2);
    left: calc(-1 * var(--avatar-size) / 2);
  }
  
  .position-top-right {
    top: calc(-1 * var(--avatar-size) / 2);
    right: calc(-1 * var(--avatar-size) / 2);
  }
  
  .position-bottom-left {
    bottom: calc(-1 * var(--avatar-size) / 2);
    left: calc(-1 * var(--avatar-size) / 2);
  }
  
  .position-bottom-right {
    bottom: calc(-1 * var(--avatar-size) / 2);
    right: calc(-1 * var(--avatar-size) / 2);
  }
  
  .avatar {
    width: 100%;
    height: 100%;
    border-radius: 50%;
    border: var(--border-width) solid var(--persona-color);
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
    background-color: var(--persona-color);
    color: white;
    font-weight: bold;
    font-size: var(--font-size);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
  }
  
  .interactive .avatar:hover {
    transform: scale(1.1);
    box-shadow: 0 3px 6px rgba(0, 0, 0, 0.3);
    cursor: pointer;
  }
  
  .avatar img {
    width: 100%;
    height: 100%;
    object-fit: cover;
  }
  
  .initials {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 100%;
    height: 100%;
    font-size: calc(var(--avatar-size) * 0.4);
  }
  
  .tooltip {
    position: absolute;
    bottom: calc(var(--avatar-size) + 8px);
    left: 50%;
    transform: translateX(-50%);
    background-color: var(--tooltip-bg, #333);
    color: var(--tooltip-text, #fff);
    border-radius: 4px;
    padding: 8px;
    min-width: 150px;
    box-shadow: 0 3px 6px rgba(0, 0, 0, 0.3);
    z-index: 20;
    pointer-events: none;
    font-size: var(--font-size);
    white-space: nowrap;
  }
  
  .tooltip::after {
    content: '';
    position: absolute;
    bottom: -8px;
    left: 50%;
    transform: translateX(-50%);
    border-width: 8px 8px 0;
    border-style: solid;
    border-color: var(--tooltip-bg, #333) transparent transparent;
  }
  
  .tooltip-content {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  
  .tooltip-header {
    display: flex;
    flex-direction: column;
  }
  
  .name {
    font-weight: bold;
  }
  
  .persona {
    font-size: 0.85em;
    opacity: 0.8;
    color: var(--persona-color);
  }
  
  .timestamp, .concept-id {
    font-size: 0.8em;
    opacity: 0.7;
  }
  
  .concept-id {
    max-width: 200px;
    overflow: hidden;
    text-overflow: ellipsis;
  }
</style>
