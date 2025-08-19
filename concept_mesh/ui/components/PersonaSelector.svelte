<script>
  import { invoke } from '@tauri-apps/api/tauri';
  import { createEventDispatcher } from 'svelte';
  
  const dispatch = createEventDispatcher();
  
  export let user = null;
  
  let loading = true;
  let error = '';
  let personaModes = [];
  let selectedPersona = null;
  
  // Load persona modes when component mounts
  $: if (user) {
    loadPersonaModes();
  }
  
  async function loadPersonaModes() {
    loading = true;
    error = '';
    
    try {
      const response = await invoke('get_persona_modes');
      personaModes = response.modes;
      
      if (personaModes.length > 0) {
        selectedPersona = personaModes[0].id;
      }
      
      loading = false;
    } catch (err) {
      error = err.toString();
      loading = false;
    }
  }
  
  async function selectPersona() {
    if (!selectedPersona) {
      error = 'Please select a persona to continue';
      return;
    }
    
    loading = true;
    error = '';
    
    try {
      const response = await invoke('select_persona', {
        userId: user.concept_id,
        personaMode: selectedPersona
      });
      
      if (response.success) {
        dispatch('personaSelected', { session: response.session });
      } else {
        error = response.error || 'Failed to select persona';
      }
    } catch (err) {
      error = err.toString();
    } finally {
      loading = false;
    }
  }
</script>

<div class="persona-selector">
  <div class="persona-container">
    <div class="header">
      <h1>Choose Your Persona</h1>
      {#if user}
        <div class="user-info">
          <div class="avatar">
            {#if user.avatar_url}
              <img src={user.avatar_url} alt="Avatar" />
            {:else}
              <div class="avatar-placeholder">{user.name ? user.name[0].toUpperCase() : 'U'}</div>
            {/if}
          </div>
          <div class="user-details">
            <h2>{user.name || 'User'}</h2>
            {#if user.email}
              <p>{user.email}</p>
            {/if}
          </div>
        </div>
      {/if}
    </div>
    
    <div class="persona-options">
      {#if loading && personaModes.length === 0}
        <div class="loading">Loading personas...</div>
      {:else}
        <p class="instruction">Select a persona to define how you'll interact with the Concept Mesh</p>
        
        <div class="options-list">
          {#each personaModes as mode}
            <label class="persona-option">
              <input 
                type="radio" 
                name="persona" 
                value={mode.id} 
                bind:group={selectedPersona}
              />
              <div class="option-content">
                <h3>{mode.name}</h3>
                <p>{mode.description}</p>
              </div>
            </label>
          {/each}
        </div>
      {/if}
    </div>
    
    {#if error}
      <div class="error-message">
        {error}
      </div>
    {/if}
    
    <div class="actions">
      <button 
        class="continue-button" 
        on:click={selectPersona} 
        disabled={loading || !selectedPersona}>
        {loading ? 'Loading...' : 'Continue'}
      </button>
    </div>
  </div>
</div>

<style>
  .persona-selector {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 100vh;
    background-color: var(--bg-color, #1a1a1a);
    color: var(--text-color, #ffffff);
  }
  
  .persona-container {
    width: 600px;
    padding: 2rem;
    border-radius: 8px;
    background-color: var(--card-bg, #2a2a2a);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
  }
  
  .header {
    margin-bottom: 2rem;
  }
  
  .header h1 {
    font-size: 2rem;
    margin-bottom: 1rem;
    color: var(--primary-color, #4a9eff);
    text-align: center;
  }
  
  .user-info {
    display: flex;
    align-items: center;
    padding: 1rem;
    border-radius: 8px;
    background-color: var(--user-info-bg, rgba(255, 255, 255, 0.05));
  }
  
  .avatar {
    width: 60px;
    height: 60px;
    border-radius: 50%;
    overflow: hidden;
    margin-right: 1rem;
  }
  
  .avatar img {
    width: 100%;
    height: 100%;
    object-fit: cover;
  }
  
  .avatar-placeholder {
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    font-weight: bold;
    background-color: var(--avatar-bg, #4a9eff);
    color: white;
  }
  
  .user-details h2 {
    font-size: 1.2rem;
    margin: 0 0 0.25rem 0;
  }
  
  .user-details p {
    font-size: 0.9rem;
    margin: 0;
    opacity: 0.8;
  }
  
  .instruction {
    margin-bottom: 1.5rem;
    text-align: center;
    font-size: 1rem;
  }
  
  .options-list {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }
  
  .persona-option {
    display: flex;
    align-items: flex-start;
    padding: 1rem;
    border: 1px solid var(--border-color, #444);
    border-radius: 6px;
    background-color: var(--option-bg, #333);
    cursor: pointer;
    transition: all 0.2s ease;
  }
  
  .persona-option:hover {
    background-color: var(--option-hover-bg, #444);
  }
  
  .persona-option input {
    margin-right: 1rem;
    margin-top: 0.25rem;
  }
  
  .option-content h3 {
    font-size: 1.1rem;
    margin: 0 0 0.25rem 0;
  }
  
  .option-content p {
    font-size: 0.9rem;
    margin: 0;
    opacity: 0.8;
  }
  
  .loading {
    text-align: center;
    padding: 2rem 0;
    font-style: italic;
    opacity: 0.7;
  }
  
  .error-message {
    margin-top: 1rem;
    padding: 0.75rem;
    border-radius: 4px;
    background-color: var(--error-bg, rgba(255, 0, 0, 0.1));
    color: var(--error-text, #ff6b6b);
    text-align: center;
  }
  
  .actions {
    margin-top: 2rem;
    display: flex;
    justify-content: center;
  }
  
  .continue-button {
    padding: 0.75rem 2rem;
    font-size: 1rem;
    font-weight: bold;
    border: none;
    border-radius: 4px;
    background-color: var(--primary-color, #4a9eff);
    color: white;
    cursor: pointer;
    transition: all 0.2s ease;
  }
  
  .continue-button:hover {
    background-color: var(--primary-color-hover, #3a8eef);
    transform: translateY(-2px);
  }
  
  .continue-button:disabled {
    opacity: 0.7;
    cursor: not-allowed;
    transform: none;
  }
</style>
