<script lang="ts">
  import { onMount } from 'svelte';
  import { browser } from '$app/environment';
  
  let isHealthy = false;
  let isChecking = true;
  let retryCount = 0;
  const maxRetries = 10;
  const retryDelay = 1000; // 1 second
  
  async function checkHealth() {
    if (!browser) return;
    
    try {
      const response = await fetch('/api/health', {
        method: 'GET',
        headers: {
          'Accept': 'application/json'
        }
      });
      
      if (response.ok) {
        isHealthy = true;
        isChecking = false;
        console.log('✅ API is healthy, proceeding with app initialization');
      } else {
        throw new Error(`Health check failed: ${response.status}`);
      }
    } catch (error) {
  const msg = error instanceof Error ? (error instanceof Error ? error.message : String(error)) : String(error);
      retryCount++;
      console.warn(`⏳ API health check failed (attempt ${retryCount
}/${maxRetries}):`, (error instanceof Error ? error.message : String(error)));
      
      if (retryCount < maxRetries) {
        setTimeout(checkHealth, retryDelay);
      } else {
        console.error('❌ API health check failed after maximum retries');
        isChecking = false;
      }
    }
  }
  
  onMount(() => {
    checkHealth();
  });
</script>

{#if isHealthy}
  <slot />
{:else if isChecking}
  <div class="health-gate">
    <div class="health-gate-content">
      <h2>Initializing TORI System</h2>
      <p>Waiting for API services to start...</p>
      <div class="spinner"></div>
      <p class="attempt">Attempt {retryCount} of {maxRetries}</p>
    </div>
  </div>
{:else}
  <div class="health-gate error">
    <div class="health-gate-content">
      <h2>Connection Error</h2>
      <p>Unable to connect to TORI API services.</p>
      <p>Please ensure the backend is running and try refreshing the page.</p>
      <button on:click={() => window.location.reload()}>
        Refresh Page
      </button>
    </div>
  </div>
{/if}

<style>
  .health-gate {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: var(--color-background, #1a1a1a);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 9999;
  }
  
  .health-gate-content {
    text-align: center;
    padding: 2rem;
  }
  
  h2 {
    color: var(--color-text, #ffffff);
    margin-bottom: 1rem;
  }
  
  p {
    color: var(--color-text-secondary, #888888);
    margin-bottom: 1rem;
  }
  
  .attempt {
    font-size: 0.875rem;
    margin-top: 1rem;
  }
  
  .spinner {
    width: 40px;
    height: 40px;
    margin: 2rem auto;
    border: 3px solid var(--color-primary-light, #4a4a4a);
    border-top-color: var(--color-primary, #00ff88);
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }
  
  @keyframes spin {
    to { transform: rotate(360deg); }
  }
  
  .error .health-gate-content {
    color: var(--color-error, #ff4444);
  }
  
  button {
    background: var(--color-primary, #00ff88);
    color: var(--color-background, #1a1a1a);
    border: none;
    padding: 0.75rem 1.5rem;
    border-radius: 4px;
    font-size: 1rem;
    cursor: pointer;
    margin-top: 1rem;
    transition: opacity 0.2s;
  }
  
  button:hover {
    opacity: 0.8;
  }
</style>
