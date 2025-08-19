<!-- Fixed UserAuth - works properly without auto-showing modal -->
<script lang="ts">
  import { userSession, loginUser, registerUser, logoutUser } from '$lib/stores/user';
  import { onMount } from 'svelte';
  import { fade, fly } from 'svelte/transition';
  
  let showAuth = false;
  let isLogin = true;
  let email = '';
  let name = '';
  let loading = false;
  let error = '';
  let isFirstTime = false;
  let authStep = 'initial'; // 'initial', 'email', 'name', 'complete'
  
  // Don't auto-show auth modal - only show when user clicks
  onMount(() => {
    // Just check if user is already logged in, don't show modal
    console.log('UserAuth component mounted');
  });
  
  async function handleSubmit() {
    if (loading) return;
    
    error = '';
    loading = true;
    
    try {
      if (!email.trim()) {
        throw new Error('Email is required');
      }
      
      if (!isLogin && !name.trim()) {
        throw new Error('Name is required');
      }
      
      let success = false;
      
      if (isLogin) {
        success = loginUser(email.trim());
        if (!success) {
          // New user - switch to registration flow
          isLogin = false;
          authStep = 'name';
          isFirstTime = true;
          error = '';
          loading = false;
          return;
        }
      } else {
        success = registerUser(email.trim(), name.trim());
        if (!success) {
          error = 'Something went wrong. Please try again.';
        } else {
          isFirstTime = true;
          authStep = 'complete';
        }
      }
      
      if (success && isLogin) {
        // Smooth transition for returning users
        authStep = 'complete';
        setTimeout(() => {
          showAuth = false;
          resetForm();
        }, 800);
      }
    } catch (err) {
      error = err instanceof Error ? err.message : 'An error occurred';
    } finally {
      if (authStep !== 'name' && authStep !== 'complete') {
        loading = false;
      }
    }
  }
  
  function resetForm() {
    email = '';
    name = '';
    error = '';
    authStep = 'initial';
    isFirstTime = false;
    isLogin = true;
  }
  
  function handleEmailStep() {
    if (!email.trim()) {
      error = 'Please enter your email';
      return;
    }
    error = '';
    authStep = 'email';
    handleSubmit();
  }
  
  function handleKeydown(event: KeyboardEvent) {
    if (event.key === 'Enter') {
      event.preventDefault();
      if (authStep === 'initial') {
        handleEmailStep();
      } else if (authStep === 'name') {
        handleSubmit();
      }
    }
  }
  
  function handleLogout() {
    logoutUser();
    resetForm();
  }
  
  function closeWelcome() {
    showAuth = false;
    resetForm();
  }
  
  function openAuth() {
    showAuth = true;
    resetForm();
  }
</script>

<!-- User status in header (when authenticated) - Clean and minimal -->
{#if $userSession.isAuthenticated && $userSession.user}
  <div class="flex items-center space-x-3">
    <div class="flex items-center space-x-2">
      <!-- Subtle user indicator -->
      <div class="w-6 h-6 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 
                  flex items-center justify-center text-white text-xs font-medium">
        {$userSession.user.name.charAt(0).toUpperCase()}
      </div>
      <div class="hidden sm:block">
        <div class="text-sm text-gray-700">
          {$userSession.user.name.split(' ')[0]}
        </div>
      </div>
    </div>
    
    <!-- Unobtrusive logout -->
    <button 
      class="text-xs text-gray-400 hover:text-gray-600 transition-colors p-1 rounded"
      on:click={handleLogout}
      title="Sign out"
    >
      ⏻
    </button>
  </div>
{:else}
  <!-- Clean sign in button when not authenticated -->
  <button 
    class="text-sm text-gray-600 hover:text-gray-800 transition-colors px-3 py-1 rounded-md border border-gray-200 hover:border-gray-300"
    on:click={openAuth}
  >
    Sign in
  </button>
{/if}

<!-- Authentication Modal - Only shows when user clicks -->
{#if showAuth}
  <div class="fixed inset-0 bg-black/20 backdrop-blur-sm flex items-center justify-center z-50 p-4"
       transition:fade={{duration: 200}}>
    <div class="bg-white rounded-xl shadow-2xl max-w-md w-full overflow-hidden"
         transition:fly={{y: 20, duration: 300}}>
      
      {#if authStep === 'initial'}
        <!-- Welcome Screen -->
        <div class="p-8 text-center">
          <div class="mb-6">
            <div class="w-16 h-16 bg-gradient-to-br from-tori-primary to-tori-secondary rounded-full 
                        flex items-center justify-center mx-auto mb-4 shadow-lg">
              <span class="text-white text-2xl font-bold">T</span>
            </div>
            <h2 class="text-2xl font-bold text-gray-800 mb-2">Welcome to TORI</h2>
            <p class="text-gray-600 text-sm">Your revolutionary AI consciousness interface</p>
          </div>
          
          {#if error}
            <div class="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg mb-4 text-sm">
              {error}
            </div>
          {/if}
          
          <!-- Email-first approach -->
          <div class="space-y-4">
            <div>
              <input 
                type="email"
                bind:value={email}
                on:keydown={handleKeydown}
                placeholder="Enter your email"
                class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                autocomplete="email"
                autofocus
              />
            </div>
            
            <button 
              class="w-full py-3 bg-tori-primary hover:bg-tori-secondary text-white font-medium rounded-lg transition-colors disabled:opacity-50"
              on:click={handleEmailStep}
              disabled={loading || !email.trim()}
            >
              {loading ? 'Checking...' : 'Continue'}
            </button>
            
            <!-- Privacy note -->
            <p class="text-xs text-gray-500 mt-4">
              🔒 Your data stays local. We respect your privacy.
            </p>
          </div>
        </div>
        
      {:else if authStep === 'name'}
        <!-- Name Collection -->
        <div class="p-8" transition:fly={{x: 100, duration: 300}}>
          <div class="mb-6 text-center">
            <h3 class="text-xl font-semibold text-gray-800 mb-2">
              Nice to meet you!
            </h3>
            <p class="text-gray-600 text-sm">
              What should TORI call you?
            </p>
          </div>
          
          <div class="space-y-4">
            <div>
              <input 
                type="text"
                bind:value={name}
                on:keydown={handleKeydown}
                placeholder="Your first name"
                class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                autocomplete="given-name"
                autofocus
              />
              <p class="text-xs text-gray-500 mt-2">
                Just your first name is fine - we keep things friendly here
              </p>
            </div>
            
            <button 
              class="w-full py-3 bg-tori-primary hover:bg-tori-secondary text-white font-medium rounded-lg transition-colors disabled:opacity-50"
              on:click={handleSubmit}
              disabled={loading || !name.trim()}
            >
              {loading ? 'Setting up...' : 'Get Started'}
            </button>
          </div>
        </div>
        
      {:else if authStep === 'complete'}
        <!-- Welcome Complete -->
        <div class="p-8 text-center" transition:fade={{duration: 300}}>
          <div class="mb-6">
            {#if isFirstTime}
              <div class="text-4xl mb-4">🎉</div>
              <h3 class="text-xl font-semibold text-gray-800 mb-2">
                Welcome to TORI, {$userSession.user?.name}!
              </h3>
              <p class="text-gray-600 text-sm">
                Your revolutionary AI consciousness is ready
              </p>
            {:else}
              <div class="text-3xl mb-4">👋</div>
              <h3 class="text-xl font-semibold text-gray-800 mb-2">
                Welcome back, {$userSession.user?.name}!
              </h3>
              <p class="text-gray-600 text-sm">
                Your revolutionary AI is ready...
              </p>
            {/if}
          </div>
          
          <!-- Quick tips for new users -->
          {#if isFirstTime}
            <div class="bg-blue-50 rounded-lg p-4 mb-6 text-left">
              <h4 class="font-medium text-blue-900 mb-2 text-sm">Revolutionary Features:</h4>
              <ul class="space-y-1 text-xs text-blue-800">
                <li>• Ghost AI collective intelligence</li>
                <li>• ELFIN++ meta-cognitive scripts</li>
                <li>• Holographic memory visualization</li>
                <li>• Your data stays completely private</li>
              </ul>
            </div>
          {/if}
          
          <button 
            class="px-6 py-2 bg-tori-primary hover:bg-tori-secondary text-white rounded-lg transition-colors text-sm font-medium"
            on:click={closeWelcome}
          >
            {isFirstTime ? 'Start Exploring' : 'Continue'}
          </button>
        </div>
      {/if}
      
      <!-- Close button -->
      {#if authStep !== 'complete'}
        <button 
          class="absolute top-4 right-4 text-gray-400 hover:text-gray-600 transition-colors"
          on:click={() => showAuth = false}
          title="Close"
        >
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      {/if}
    </div>
  </div>
{/if}

<style>
  /* Smooth transitions */
  input {
    transition: all 0.2s ease;
  }
  
  input:focus {
    transform: translateY(-1px);
  }
  
  button {
    transition: all 0.2s ease;
  }
  
  button:active {
    transform: scale(0.98);
  }
</style>
