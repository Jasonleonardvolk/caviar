<script lang="ts">
  import { enhance } from '$app/forms';
  import { fade, fly } from 'svelte/transition';
  
  export let form;
  
  let loading = false;
</script>

<svelte:head>
  <title>TORI - Sign In</title>
</svelte:head>

<!-- Login page with TORI's signature style -->
<div class="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-purple-50 flex items-center justify-center p-4">
  <div class="bg-white rounded-2xl shadow-2xl max-w-md w-full overflow-hidden" transition:fly={{y: 20, duration: 300}}>
    
    <!-- Header -->
    <div class="px-8 pt-8 pb-6 text-center">
      <div class="w-16 h-16 bg-gradient-to-br from-tori-primary to-tori-secondary rounded-full 
                  flex items-center justify-center mx-auto mb-4 shadow-lg">
        <span class="text-white text-2xl font-bold">T</span>
      </div>
      <h1 class="text-2xl font-bold text-gray-800 mb-2">Welcome to TORI</h1>
      <p class="text-gray-600 text-sm">Your revolutionary AI consciousness interface</p>
    </div>
    
    <!-- Login Form -->
    <form 
      method="POST" 
      use:enhance={() => {
        loading = true;
        return async ({ update }) => {
          loading = false;
          await update();
        };
      }}
      class="px-8 pb-8"
    >
      <!-- Error message -->
      {#if form?.error}
        <div class="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg mb-4 text-sm"
             transition:fade={{duration: 200}}>
          {form.error}
        </div>
      {/if}
      
      <!-- Success message for new users -->
      {#if form?.success}
        <div class="bg-green-50 border border-green-200 text-green-700 px-4 py-3 rounded-lg mb-4 text-sm"
             transition:fade={{duration: 200}}>
          {form.success}
        </div>
      {/if}
      
      <div class="space-y-4">
        <!-- Username field -->
        <div>
          <label for="username" class="block text-sm font-medium text-gray-700 mb-2">
            Username
          </label>
          <input 
            type="text" 
            id="username"
            name="username" 
            required
            placeholder="Enter your username"
            class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
            autofocus
          />
          <p class="text-xs text-gray-500 mt-1">Use 'admin' for administrative access</p>
        </div>
        
        <!-- Password field -->
        <div>
          <label for="password" class="block text-sm font-medium text-gray-700 mb-2">
            Password
          </label>
          <input 
            type="password" 
            id="password"
            name="password" 
            required
            placeholder="Enter your password"
            class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
          />
        </div>
        
        <!-- Submit button -->
        <button 
          type="submit" 
          disabled={loading}
          class="w-full py-3 bg-tori-primary hover:bg-tori-secondary disabled:bg-gray-300 disabled:cursor-not-allowed 
                 text-white font-medium rounded-lg transition-colors duration-200 flex items-center justify-center"
        >
          {#if loading}
            <div class="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin mr-2"></div>
            Signing in...
          {:else}
            Sign In
          {/if}
        </button>
      </div>
      
      <!-- Privacy note -->
      <p class="text-xs text-gray-500 mt-6 text-center">
        🔒 Your data stays local. TORI respects your privacy.
      </p>
    </form>
  </div>
</div>

<style>
  /* Custom focus states for accessibility */
  input:focus {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15);
  }
  
  button:active {
    transform: scale(0.98);
  }
</style>
