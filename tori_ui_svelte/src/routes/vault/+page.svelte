<!-- Enhanced Vault page with Memory Vault Dashboard integration -->
<script lang="ts">
  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  import MemoryVaultDashboard from '$lib/components/vault/MemoryVaultDashboard.svelte';
  
  // Get user info from page data
  $: userId = $page.data?.user?.id || '';
  $: userName = $page.data?.user?.name || 'User';
  
  let mounted = false;
  
  onMount(() => {
    mounted = true;
    console.log('🏛️ Vault page mounted with enhanced Memory Vault Dashboard');
  });
</script>

<svelte:head>
  <title>TORI - Memory Vault</title>
</svelte:head>

<!-- Use the enhanced Memory Vault Dashboard component -->
<div class="h-full">
  {#if mounted}
    <MemoryVaultDashboard {userId} {userName} />
  {:else}
    <!-- Initial loading state -->
    <div class="h-full flex items-center justify-center bg-gray-50">
      <div class="text-center">
        <div class="animate-spin text-4xl mb-4">⟳</div>
        <div class="text-gray-600">Initializing Memory Vault...</div>
      </div>
    </div>
  {/if}
</div>
