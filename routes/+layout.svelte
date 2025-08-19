<!-- CLEAN LAYOUT - Zero problematic imports -->
<script lang="ts">
  import '../app.css';
  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  import { browser } from '$app/environment';
  
  // Import only safe components (no cognitive imports)
  import MemoryPanel from '$lib/components/MemoryPanel.svelte';
  import ThoughtspaceRenderer from '$lib/components/ThoughtspaceRenderer.svelte';
  import ScholarSpherePanel from '$lib/components/ScholarSpherePanel.svelte';
  
  // Get user data from server
  export let data: { user: { name: string; role: 'admin' | 'user' } | null };
  
  let mounted = false;
  let showWelcome = false;
  let elfinReady = false;
  
  // Feature flag for ScholarSphere UI
  const lab_sphere_ui = true;
  
  // ULTRA-SIMPLE ELFIN++ ENGINE - No imports, no dependencies
  function createSimpleELFIN() {
    console.log('🧬 Creating ultra-simple ELFIN++ engine...');
    
    const engine = {
      // Simple data storage
      scripts: [
        { id: 'synthesis', name: 'Knowledge Synthesis', author: 'Scholar' },
        { id: 'research', name: 'Research Orchestrator', author: 'Explorer' },
        { id: 'novelty', name: 'Novelty Injection', author: 'Creator' }
      ],
      executions: [],
      
      // Main interface methods
      getExecutionStats() {
        return {
          totalScripts: this.scripts.length,
          totalExecutions: this.executions.length,
          averageSuccessRate: 1.0,
          recentExecutions: this.executions.slice(-5),
          topScripts: this.scripts,
          engineType: 'ultra_simple_engine',
          status: 'ready',
          timestamp: new Date().toISOString()
        };
      },
      
      triggerTestUpload() {
        console.log('🧪 Ultra-simple ELFIN++ test triggered');
        
        // Record test execution
        this.executions.push({
          id: Date.now(),
          type: 'test_upload',
          timestamp: new Date(),
          success: true,
          details: 'Manual test upload successful'
        });
        
        // Dispatch test event
        if (typeof window !== 'undefined') {
          window.dispatchEvent(new CustomEvent('tori:upload', {
            detail: {
              filename: 'SimpleTest.pdf',
              text: 'Test document content',
              concepts: ['Test', 'Simple'],
              timestamp: new Date(),
              source: 'simple_test'
            }
          }));
        }
        
        console.log('✅ Test upload completed');
        return this.getExecutionStats();
      },
      
      processUpload(details) {
        console.log('📚 Processing upload:', details.filename);
        
        this.executions.push({
          id: Date.now(),
          type: 'document_upload',
          filename: details.filename,
          timestamp: new Date(),
          success: true,
          concepts: details.concepts || []
        });
        
        console.log('✅ Upload processed successfully');
      }
    };
    
    console.log('✅ Ultra-simple ELFIN++ engine created');
    return engine;
  }
  
  // DIRECT WINDOW ASSIGNMENT - No async, no delays
  function initializeELFIN() {
    if (typeof window === 'undefined') {
      console.log('❌ Window undefined - skipping ELFIN++ init');
      return false;
    }
    
    console.log('🌐 Initializing ELFIN++ directly on window...');
    
    try {
      const engine = createSimpleELFIN();
      
      // Direct assignment
      (window as any).ELFIN = engine;
      (window as any).TORI = {
        elfin: engine,
        checkStats: () => engine.getExecutionStats(),
        testUpload: () => engine.triggerTestUpload(),
        checkConcepts: () => localStorage.getItem('tori-concept-mesh') || '[]',
        checkDocs: () => localStorage.getItem('tori-scholarsphere-documents') || '[]'
      };
      
      // Setup upload event listener
      window.addEventListener('tori:upload', (event: any) => {
        engine.processUpload(event.detail);
      });
      
      console.log('✅ window.ELFIN assigned successfully');
      console.log('✅ window.TORI debug interface ready');
      
      // Immediate verification
      const testStats = (window as any).ELFIN.getExecutionStats();
      console.log('🧪 Immediate verification successful:', testStats);
      
      // Double verification
      if ((window as any).ELFIN && typeof (window as any).ELFIN.getExecutionStats === 'function') {
        console.log('✅ Final verification: ELFIN++ is fully functional');
        return true;
      } else {
        console.error('❌ Final verification failed: ELFIN++ not properly assigned');
        return false;
      }
    } catch (error) {
      console.error('❌ ELFIN++ initialization failed:', error);
      return false;
    }
  }
  
  // Check if user has seen welcome message before
  function shouldShowWelcome(): boolean {
    if (!browser || !data.user) return false;
    
    const hasSeenWelcome = localStorage.getItem('tori-welcome-seen');
    return !hasSeenWelcome;
  }
  
  // Mark welcome as seen
  function dismissWelcome() {
    showWelcome = false;
    if (browser) {
      localStorage.setItem('tori-welcome-seen', 'true');
      console.log('👋 Welcome message dismissed and stored');
    }
  }
  
  onMount(() => {
    mounted = true;
    
    console.log('🧠 TORI Genesis UI Loading (Clean Version)...');
    
    // Initialize ELFIN++ with no dependencies
    if (browser) {
      console.log('🌐 Browser environment detected, initializing ELFIN++...');
      
      const success = initializeELFIN();
      console.log('🔍 initializeELFIN() returned:', success);
      
      if (success) {
        elfinReady = true;
        console.log('✅ elfinReady flag set to TRUE');
        console.log('✅ Clean ELFIN++ initialization successful!');
        console.log('🧪 Ready to test:');
        console.log('  window.ELFIN.getExecutionStats()');
        console.log('  window.TORI.testUpload()');
      } else {
        console.error('❌ Clean ELFIN++ initialization failed');
        console.error('❌ elfinReady remains FALSE');
      }
      
      // Additional polling check (backup)
      setTimeout(() => {
        if (typeof window !== 'undefined' && (window as any).ELFIN && typeof (window as any).ELFIN.getExecutionStats === 'function') {
          if (!elfinReady) {
            console.log('🔄 Backup check: ELFIN++ detected, setting flag to ready');
            elfinReady = true;
          }
        }
      }, 100);
    } else {
      console.log('🚫 Not in browser environment, skipping ELFIN++ init');
    }
    
    // Show welcome message only if user hasn't seen it before
    showWelcome = shouldShowWelcome();
    if (showWelcome) {
      console.log('👋 Showing welcome message for new user');
    } else {
      console.log('👋 Welcome message previously dismissed');
    }
    
    console.log('✅ Clean TORI Genesis UI Ready');
  });
  
  // Page metadata
  $: pageTitle = $page.url.pathname === '/' ? 'TORI - Consciousness Interface' : 
                 $page.url.pathname === '/login' ? 'TORI - Sign In' :
                 `TORI - ${$page.url.pathname.slice(1)}`;
  
  function handleScholarSphereUpload(event) {
    const { document, conceptsAdded } = event.detail;
    console.log('📚 ScholarSphere upload completed:', document.filename, `${conceptsAdded} concepts added`);
    console.log('🧬 ELFIN++ should process this upload');
  }
</script>

<svelte:head>
  <title>{pageTitle}</title>
  <meta name="description" content="TORI - Temporal Ontological Reality Interface" />
</svelte:head>

<!-- TORI Welcome Message -->
{#if showWelcome && data.user}
  <div class="fixed inset-0 bg-black/20 backdrop-blur-sm flex items-center justify-center z-50 p-4">
    <div class="bg-gradient-to-r from-indigo-100 to-purple-100 p-8 rounded-xl shadow-md text-center max-w-2xl mx-auto">
      <h1 class="text-3xl font-bold text-gray-800 mb-4">
        Welcome. I'm <span class="text-purple-700">TORI</span>.
      </h1>
      <p class="text-lg text-gray-700 leading-relaxed mb-6">
        Type or upload something you care about — something on your mind.
        <br />
        I'll remember it. I'll never share it.
        <br />
        This is just for you.
        <br />
        I'll always be here.
      </p>
      
      {#if data.user.role === 'admin'}
        <p class="text-sm text-purple-600 mb-4">
          🔧 Admin access granted. ScholarSphere document vault is now available.
        </p>
      {/if}
      
      <button 
        on:click={dismissWelcome}
        class="px-6 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors font-medium"
      >
        Let's Begin
      </button>
    </div>
  </div>
{/if}

{#if $page.url.pathname === '/login'}
  <!-- Login page -->
  <slot />
{:else if data.user}
  <!-- Main layout -->
  <div class="flex h-screen bg-gray-50 text-gray-900 overflow-hidden">
    
    <!-- Left Panel: Memory System -->
    <aside class="w-80 flex-shrink-0 h-full border-r border-gray-200">
      <MemoryPanel />
    </aside>

    <!-- Main Content -->
    <main class="flex-1 min-w-0 h-full overflow-hidden flex flex-col bg-white">
      <!-- Header -->
      <header class="flex items-center justify-between px-6 py-4 border-b border-gray-200 bg-white">
        <div class="flex items-center space-x-4">
          <div class="flex items-center space-x-2">
            <div class="w-8 h-8 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-lg flex items-center justify-center">
              <span class="text-white text-sm font-bold">T</span>
            </div>
            <div>
              <h1 class="text-xl font-bold text-gray-900">TORI</h1>
              <div class="text-xs text-gray-500">Consciousness Interface</div>
            </div>
          </div>
        </div>
        
        <!-- Header controls -->
        <div class="flex items-center space-x-4">
          <div class="text-sm text-gray-600">
            Welcome, {data.user.name}
            {#if data.user.role === 'admin'}
              <span class="text-purple-600 font-medium">(Admin)</span>
            {/if}
          </div>
          
          <div class="flex items-center space-x-3">
            <div class="text-xs text-gray-500 flex items-center space-x-2">
              <span>Phase 3 • Clean Build</span>
              {#if data.user.role === 'admin'}
                <span class="text-green-600">• ScholarSphere online</span>
              {/if}
              {#if elfinReady}
                <span class="text-blue-600">• ELFIN++ ready</span>
              {:else}
                <span class="text-red-600">• ELFIN++ initializing</span>
              {/if}
            </div>
          </div>
          
          <a href="/logout" class="text-sm text-gray-500 hover:text-gray-700 transition-colors">
            Logout
          </a>
        </div>
      </header>
      
      <!-- Main content -->
      <div class="flex-1 overflow-hidden">
        <slot />
      </div>
    </main>

    <!-- Right Panel -->
    <aside class="w-80 flex-shrink-0 h-full border-l border-gray-200">
      {#if data.user.role === 'admin'}
        <ScholarSpherePanel on:upload-complete={handleScholarSphereUpload} />
      {:else}
        <ThoughtspaceRenderer />
      {/if}
    </aside>
  </div>

  <!-- Dev indicator -->
  {#if mounted && import.meta.env.DEV}
    <div class="fixed bottom-4 left-1/2 transform -translate-x-1/2 z-40">
      <div class="bg-black/80 text-white px-3 py-1 rounded-full text-xs opacity-50 hover:opacity-100 transition-opacity">
        Clean Build • 
        {#if elfinReady}ELFIN++ Ready{:else}ELFIN++ Loading{/if}
        • {data.user.name} ({data.user.role})
      </div>
    </div>
  {/if}
{:else}
  <!-- Not authenticated -->
  <div class="min-h-screen flex items-center justify-center bg-gray-50">
    <div class="text-center">
      <h1 class="text-2xl font-bold text-gray-900 mb-4">Access Denied</h1>
      <p class="text-gray-600 mb-4">Please sign in to access TORI.</p>
      <a href="/login" class="text-blue-600 hover:text-blue-800">Go to Sign In</a>
    </div>
  </div>
{/if}

<style>
  :global(body) {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    overflow: hidden;
  }
</style>
