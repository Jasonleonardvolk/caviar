<!-- Enhanced Navigation Panel with Persona Management -->
<script lang="ts">
  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  import { ghostPersona, setPersona } from '$lib/stores/ghostPersona';
  import PersonaPanel from './PersonaPanel.svelte';
  
  export let activeView = 'chat'; // 'chat', 'upload', 'gallery', 'settings'
  
  // Persona management state
  let showPersonaPanel = false;
  let showCreatePersona = false;
  
  // Get current persona from store, default to ENOLA
  let currentPersona = {
    id: 'enola',
    name: 'Enola',
    description: 'Systematic investigation and discovery',
    psi: 'investigative',
    epsilon: 'focused',
    tau: 'methodical',
    phi: '2.718',
    color: { r: 0.15, g: 0.39, b: 0.92 }
  };
  
  // Available personas
  let availablePersonas = [
    {
      id: 'enola',
      name: 'Enola',
      description: 'Systematic investigation and discovery',
      psi: 'investigative',
      epsilon: 'focused',
      tau: 'methodical',
      phi: '2.718',
      color: { r: 0.15, g: 0.39, b: 0.92 }
    },
    {
      id: 'mentor',
      name: 'Mentor',
      description: 'Guides through questions and frameworks',
      psi: 'socratic',
      epsilon: 'contemplative',
      tau: 'balanced',
      phi: '42',
      color: { r: 0.31, g: 0.27, b: 0.9 }
    },
    {
      id: 'scholar',
      name: 'Scholar',
      description: 'Deep research and systematic analysis',
      psi: 'analytical',
      epsilon: 'focused',
      tau: 'timeless',
      phi: '137',
      color: { r: 0.02, g: 0.59, b: 0.41 }
    },
    {
      id: 'explorer',
      name: 'Explorer',
      description: 'Discovers connections and new territories',
      psi: 'lateral',
      epsilon: 'curious',
      tau: 'novel',
      phi: '89',
      color: { r: 0.86, g: 0.15, b: 0.15 }
    },
    {
      id: 'architect',
      name: 'Architect',
      description: 'Builds frameworks and systematic structures',
      psi: 'structural',
      epsilon: 'methodical',
      tau: 'long-term',
      phi: '21',
      color: { r: 0.49, g: 0.23, b: 0.93 }
    },
    {
      id: 'creator',
      name: 'Creator',
      description: 'Synthesizes novel ideas and innovations',
      psi: 'intuitive',
      epsilon: 'inspired',
      tau: 'future',
      phi: '73',
      color: { r: 0.92, g: 0.35, b: 0.05 }
    }
  ];
  
  // Form for new persona
  let newPersonaForm = {
    name: '',
    description: '',
    psi: 'analytical',
    epsilon: 'calm',
    tau: 'present',
    phi: '1',
    color: { r: 0.5, g: 0.5, b: 0.5 }
  };
  
  // Load saved persona on mount
  onMount(() => {
    if (typeof window !== 'undefined') {
      // Load custom personas from localStorage
      const customPersonas = JSON.parse(localStorage.getItem('tori-custom-personas') || '[]');
      if (customPersonas.length > 0) {
        availablePersonas = [...availablePersonas, ...customPersonas];
      }
      
      // Load last selected persona from localStorage
      const savedPersonaId = localStorage.getItem('tori-last-persona') || 'enola';
      const savedPersona = availablePersonas.find(p => p.id === savedPersonaId);
      
      if (savedPersona) {
        currentPersona = savedPersona;
        setPersona(savedPersona.name);
      } else {
        // Default to ENOLA
        currentPersona = availablePersonas[0]; // ENOLA is first
        setPersona('Enola');
      }
    }
  });
  
  // Navigation items with icons and routes
  const navItems = [
    { id: 'select', label: 'Change Persona', icon: '🔄', action: 'selectPersona' },
    { id: 'create', label: 'New Persona', icon: '✨', action: 'createPersona' },
    { id: 'vault', label: 'Vault', icon: '🗄️', route: '/vault' },
    { id: 'history', label: 'History', icon: '👻', route: '/ghost-history' }
  ];
  
  function handleNavClick(item) {
    if (item.action === 'selectPersona') {
      showPersonaPanel = true;
      showCreatePersona = false;
    } else if (item.action === 'createPersona') {
      showPersonaPanel = true;
      showCreatePersona = true;
    } else if (item.route && typeof window !== 'undefined') {
      window.location.href = item.route;
    }
  }
  
  function switchPersona(persona) {
    currentPersona = persona;
    setPersona(persona.name);
    
    // Save to localStorage
    if (typeof window !== 'undefined') {
      localStorage.setItem('tori-last-persona', persona.id);
    }
    
    console.log(`🎭 Switched to persona: ${persona.name}`);
    showPersonaPanel = false;
  }
  
  function createPersona() {
    if (!newPersonaForm.name.trim()) return;
    
    const newPersona = {
      id: newPersonaForm.name.toLowerCase().replace(/\s+/g, '-'),
      ...newPersonaForm,
      name: newPersonaForm.name.trim()
    };
    
    availablePersonas = [...availablePersonas, newPersona];
    
    // Save to localStorage
    if (typeof window !== 'undefined') {
      const savedPersonas = JSON.parse(localStorage.getItem('tori-custom-personas') || '[]');
      savedPersonas.push(newPersona);
      localStorage.setItem('tori-custom-personas', JSON.stringify(savedPersonas));
    }
    
    // Switch to new persona
    switchPersona(newPersona);
    
    // Reset form
    newPersonaForm = {
      name: '',
      description: '',
      psi: 'analytical',
      epsilon: 'calm',
      tau: 'present',
      phi: '1',
      color: { r: 0.5, g: 0.5, b: 0.5 }
    };
    
    showCreatePersona = false;
  }
  
  function toggleCreatePersona() {
    showCreatePersona = !showCreatePersona;
  }
  
  // Update active view based on current route
  $: if ($page) {
    const currentPath = $page.url.pathname;
    const activeItem = navItems.find(item => item.route === currentPath);
    if (activeItem) {
      activeView = activeItem.id;
    }
  }
</script>

<nav class="navigation-panel">
    <div class="nav-header">
    <div class="persona-section">
      <h3 class="nav-title">
        <span class="title-icon">🎭</span>
        Current Persona
      </h3>
      {#if currentPersona}
        <div class="current-persona-display">
          <div class="persona-badge" style="background-color: rgb({currentPersona.color.r * 255}, {currentPersona.color.g * 255}, {currentPersona.color.b * 255})">
            {currentPersona.name.charAt(0)}
          </div>
          <div class="persona-info">
            <span class="persona-name">{currentPersona.name}</span>
            <span class="persona-description">{currentPersona.description}</span>
          </div>
        </div>
      {/if}
    </div>
  </div>
  
  <div class="nav-items">
    {#each navItems as item}
      <button 
        class="nav-item {activeView === item.id ? 'active' : ''}"
        on:click={() => handleNavClick(item)}
      >
        <span class="nav-icon">{item.icon}</span>
        <span class="nav-label">{item.label}</span>
      </button>
    {/each}
  </div>
  
  <!-- Quick Actions -->
  <div class="quick-actions">
    <h4 class="actions-title">Quick Actions</h4>
    <button class="action-button upload-button" on:click={() => window.location.href = '/upload'}>
      <span class="action-icon">⬆️</span>
      <span>Upload Document</span>
    </button>
    <button class="action-button photo-button" on:click={() => window.location.href = '/upload?mode=photo'}>
      <span class="action-icon">📷</span>
      <span>Upload Photo</span>
    </button>
    <button class="action-button video-button" on:click={() => window.location.href = '/upload?mode=video'}>
      <span class="action-icon">🎥</span>
      <span>Upload Video</span>
    </button>
  </div>
</nav>

<!-- Persona Management Panel -->
<PersonaPanel 
  bind:showPersonaPanel={showPersonaPanel}
  bind:showCreatePersona={showCreatePersona}
  bind:currentPersona={currentPersona}
  bind:availablePersonas={availablePersonas}
  bind:newPersonaForm={newPersonaForm}
  {switchPersona}
  {createPersona}
  {toggleCreatePersona}
/>

<style>
  .navigation-panel {
    background: var(--color-secondary);
    border-radius: var(--border-radius);
    padding: var(--space-3);
    margin-bottom: var(--space-3);
    box-shadow: var(--shadow-sm);
  }
  
  :global(.dark) .navigation-panel {
    background: #1a1a1a;
  }
  
  .nav-header {
    margin-bottom: var(--space-3);
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  
  .nav-title {
    font-size: var(--text-xl);
    font-weight: 600;
    color: var(--color-text-primary);
    margin: 0;
  }
  
  .persona-badge {
    width: 24px;
    height: 24px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: var(--text-xs);
    font-weight: 600;
  }
  
  .persona-name {
    font-size: var(--text-sm);
    font-weight: 500;
    color: var(--color-text-primary);
  }
  
  :global(.dark) .nav-title {
    color: #e0e0e0;
  }
  
  .nav-items {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    margin-bottom: var(--space-4);
  }
  
  .nav-item {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    padding: var(--space-2) var(--space-2);
    background: var(--color-base);
    border: var(--border-width) solid var(--color-border);
    border-radius: var(--border-radius-sm);
    color: var(--color-text-primary);
    font-size: var(--text-sm);
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s ease;
    width: 100%;
    text-align: left;
  }
  
  :global(.dark) .nav-item {
    background: rgba(219, 234, 254, 0.1);
    border: 1px solid rgba(147, 197, 253, 0.2);
    color: #93c5fd;
  }
  
  .nav-item:hover {
    background: var(--color-active);
    border-color: var(--color-accent);
    transform: translateY(-1px);
    box-shadow: var(--shadow-sm);
  }
  
  :global(.dark) .nav-item:hover {
    background: rgba(191, 219, 254, 0.15);
    border-color: rgba(96, 165, 250, 0.3);
    color: #dbeafe;
  }
  
  .nav-item.active {
    background: var(--color-active);
    border-color: var(--color-accent);
    color: var(--color-accent);
    font-weight: 600;
  }
  
  :global(.dark) .nav-item.active {
    background: rgba(219, 234, 254, 0.15);
    border-color: rgba(96, 165, 250, 0.4);
    color: #dbeafe;
  }
  
  .nav-icon {
    font-size: var(--text-xl);
    flex-shrink: 0;
  }
  
  .nav-label {
    font-weight: 500;
  }
  
  .quick-actions {
    padding-top: var(--space-3);
    border-top: var(--border-width) solid var(--color-border);
  }
  
  .actions-title {
    font-size: var(--text-xs);
    font-weight: 600;
    color: var(--color-text-secondary);
    margin: 0 0 var(--space-2) 0;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  
  .action-button {
    display: flex;
    align-items: center;
    gap: var(--space-1);
    width: 100%;
    padding: calc(var(--space-1) * 1.5) var(--space-2);
    margin-bottom: var(--space-1);
    background: var(--color-base);
    border: var(--border-width) solid var(--color-border);
    border-radius: var(--border-radius-sm);
    color: var(--color-text-primary);
    font-size: var(--text-sm);
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s ease;
  }
  
  .action-button:hover {
    transform: translateY(-1px);
    box-shadow: var(--shadow-sm);
    background: var(--color-active);
    border-color: var(--color-accent);
    color: var(--color-accent);
  }
  
  :global(.dark) .action-button:hover {
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
  }
  
  .action-icon {
    font-size: var(--text-lg);
  }
  
  
  .persona-section {
    width: 100%;
  }
  
  .nav-title {
    display: flex;
    align-items: center;
    gap: var(--space-1);
    font-size: var(--text-sm);
    font-weight: 600;
    color: var(--color-text-secondary);
    margin: 0 0 var(--space-2) 0;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  
  .title-icon {
    font-size: var(--text-lg);
  }
  
  .current-persona-display {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    padding: var(--space-2);
    background: var(--color-base);
    border: 2px solid var(--color-border);
    border-radius: var(--border-radius);
    transition: all 0.2s ease;
  }
  
  .current-persona-display:hover {
    border-color: var(--color-accent);
    transform: translateY(-1px);
    box-shadow: var(--shadow-md);
  }
  
  :global(.dark) .current-persona-display {
    background: rgba(255, 255, 255, 0.05);
    border-color: rgba(255, 255, 255, 0.1);
  }
  
  :global(.dark) .current-persona-display:hover {
    border-color: rgba(138, 43, 226, 0.3);
  }
  
  .persona-badge {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: var(--text-lg);
    font-weight: 600;
    box-shadow: var(--shadow-sm);
    flex-shrink: 0;
  }
  
  .persona-info {
    display: flex;
    flex-direction: column;
    gap: 2px;
    flex: 1;
    min-width: 0;
  }
  
  .persona-name {
    font-size: var(--text-base);
    font-weight: 600;
    color: var(--color-text-primary);
  }
  
  .persona-description {
    font-size: var(--text-xs);
    color: var(--color-text-secondary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  
  :global(.dark) .persona-name {
    color: #e0e0e0;
  }
  
  :global(.dark) .persona-description {
    color: #999;
  }

  /* Responsive adjustments */
  @media (max-width: 1440px) {
    .navigation-panel {
      padding: var(--space-2);
    }
    
    .nav-items {
      gap: calc(var(--space-1) * 1.5);
      margin-bottom: var(--space-3);
    }
    
    .quick-actions {
      padding-top: var(--space-2);
    }
  }
</style>