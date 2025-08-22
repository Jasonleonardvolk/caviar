<script lang="ts">
  import { onMount } from 'svelte';
  let HUD: any = null;
  let loadError: string | null = null;

  onMount(async () => {
    try {
      // Preferred canonical path (most repos): src/lib/components/HolographicDisplay.svelte
      HUD = (await import('$lib/components/HolographicDisplay.svelte')).default;
    } catch (e1) {
      try {
        // Fallback #1 seen in some layouts: src/lib/renderer/HolographicDisplay.svelte
        HUD = (await import('$lib/renderer/HolographicDisplay.svelte')).default;
      } catch (e2) {
        try {
          // Fallback #2: src/lib/HolographicDisplay.svelte
          HUD = (await import('$lib/HolographicDisplay.svelte')).default;
        } catch (e3) {
          loadError = String(e3);
        }
      }
    }
  });
</script>

{#if HUD}
  <svelte:component this={HUD} />
{:else}
  <div style="padding:16px;font-family:ui-monospace,Consolas,monospace">
    <h3>iRis HUD route is up, but the HUD component was not found.</h3>
    <p>Expected one of:</p>
    <ul>
      <li><code>tori_ui_svelte\src\lib\components\HolographicDisplay.svelte</code></li>
      <li><code>tori_ui_svelte\src\lib\renderer\HolographicDisplay.svelte</code></li>
      <li><code>tori_ui_svelte\src\lib\HolographicDisplay.svelte</code></li>
    </ul>
    {#if loadError}<pre>{loadError}</pre>{/if}
  </div>
{/if}