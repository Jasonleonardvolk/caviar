<script>
  import { invoke } from '@tauri-apps/api/tauri';
  import { createEventDispatcher } from 'svelte';
  import { shell } from '@tauri-apps/api';
  
  const dispatch = createEventDispatcher();
  
  let loading = false;
  let error = '';
  let currentProvider = '';
  
  // Provider configuration
  const providers = [
    {
      id: 'github',
      name: 'GitHub',
      icon: 'github',
      description: 'Connect using your GitHub account'
    },
    {
      id: 'google',
      name: 'Google',
      icon: 'google',
      description: 'Connect using your Google account'
    },
    {
      id: 'apple',
      name: 'Apple',
      icon: 'apple',
      description: 'Connect using your Apple ID'
    },
    {
      id: 'discord',
      name: 'Discord',
      icon: 'discord',
      description: 'Connect using your Discord account'
    },
    {
      id: 'auth0',
      name: 'Auth0',
      icon: 'auth0',
      description: 'Connect using Auth0'
    }
  ];
  
  // Start OAuth flow for provider
  async function login(providerId) {
    loading = true;
    currentProvider = providerId;
    error = '';
    
    try {
      const response = await invoke('login_oauth', { provider: providerId });
      
      if (!response.success) {
        error = response.error || 'Login failed';
        loading = false;
        return;
      }
      
      if (response.auth_url) {
        // Open browser with the auth URL
        await shell.open(response.auth_url);
        
        // Tauri app will handle the callback via URL scheme
        // but we'll show a message in case the user needs to manually
        // complete the process
        setTimeout(() => {
          loading = false;
        }, 5000);
      }
    } catch (err) {
      error = err.toString();
      loading = false;
    }
  }
  
  // Handle successful login (called from the main app when OAuth callback received)
  export function handleLoginSuccess(user) {
    dispatch('loginSuccess', { user });
  }
</script>

<div class="login-screen">
  <div class="login-container">
    <div class="login-header">
      <h1>Welcome to Concept Mesh</h1>
      <p>Choose an identity provider to get started</p>
    </div>
    
    <div class="provider-list">
      {#each providers as provider}
        <button 
          class="provider-button" 
          on:click={() => login(provider.id)}
          disabled={loading}>
          <div class="provider-icon">
            <i class="icon-{provider.icon}"></i>
          </div>
          <div class="provider-info">
            <h2>{provider.name}</h2>
            <p>{provider.description}</p>
          </div>
          {#if loading && currentProvider === provider.id}
            <div class="spinner"></div>
          {/if}
        </button>
      {/each}
    </div>
    
    {#if error}
      <div class="error-message">
        {error}
      </div>
    {/if}
    
    {#if loading}
      <div class="status-message">
        Connecting to {currentProvider}...
        <p class="small">Your browser will open to complete authentication.</p>
      </div>
    {/if}
  </div>
</div>

<style>
  .login-screen {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 100vh;
    background-color: var(--bg-color, #1a1a1a);
    color: var(--text-color, #ffffff);
  }
  
  .login-container {
    width: 600px;
    padding: 2rem;
    border-radius: 8px;
    background-color: var(--card-bg, #2a2a2a);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
  }
  
  .login-header {
    text-align: center;
    margin-bottom: 2rem;
  }
  
  .login-header h1 {
    font-size: 2rem;
    margin-bottom: 0.5rem;
    color: var(--primary-color, #4a9eff);
  }
  
  .provider-list {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }
  
  .provider-button {
    display: flex;
    align-items: center;
    padding: 1rem;
    border: 1px solid var(--border-color, #444);
    border-radius: 6px;
    background-color: var(--button-bg, #333);
    color: var(--button-text, #fff);
    cursor: pointer;
    transition: all 0.2s ease;
    text-align: left;
  }
  
  .provider-button:hover {
    background-color: var(--button-hover-bg, #444);
    transform: translateY(-2px);
  }
  
  .provider-button:disabled {
    opacity: 0.7;
    cursor: not-allowed;
  }
  
  .provider-icon {
    width: 40px;
    height: 40px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-right: 1rem;
    font-size: 1.5rem;
  }
  
  .provider-info {
    flex: 1;
  }
  
  .provider-info h2 {
    font-size: 1.1rem;
    margin: 0 0 0.25rem 0;
  }
  
  .provider-info p {
    font-size: 0.9rem;
    margin: 0;
    opacity: 0.8;
  }
  
  .error-message {
    margin-top: 1rem;
    padding: 0.75rem;
    border-radius: 4px;
    background-color: var(--error-bg, rgba(255, 0, 0, 0.1));
    color: var(--error-text, #ff6b6b);
    text-align: center;
  }
  
  .status-message {
    margin-top: 1rem;
    text-align: center;
    font-size: 0.9rem;
    opacity: 0.8;
  }
  
  .small {
    font-size: 0.8rem;
    opacity: 0.7;
  }
  
  .spinner {
    width: 20px;
    height: 20px;
    border: 2px solid rgba(255,255,255,0.3);
    border-radius: 50%;
    border-top-color: #fff;
    animation: spin 1s ease-in-out infinite;
  }
  
  @keyframes spin {
    to { transform: rotate(360deg); }
  }
  
  /* Icon specific styles */
  .icon-github::before { content: ""; }
  .icon-google::before { content: ""; }
  .icon-apple::before { content: ""; }
  .icon-discord::before { content: ""; }
  .icon-auth0::before { content: ""; }
  
  /* In a real implementation, use actual icon fonts or SVGs */
</style>
