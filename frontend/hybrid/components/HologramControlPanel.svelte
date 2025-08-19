<script>
  /**
   * HologramControlPanel.svelte
   * UI panel for controlling holographic rendering parameters and viewing diagnostics.
   * 
   * Exposed controls:
   *  - Blend Ratio: slider to adjust blend between two rendering states (e.g., physical vs computational).
   *  - Phase Mode: dropdown to select phase rendering mode (e.g., Kerr or Soliton).
   *  - Persona State: selection of persona/emotion state for the hologram or AI.
   *  - View Mode: select output view configuration (quilt, stereo, depth layers).
   *  - Diagnostics: shows real-time diagnostic info (FPS, memory, etc.) if provided.
   * 
   * The component exports bindable props for each control, and emits an 'update' event with all values whenever any control changes.
   */
  import { createEventDispatcher } from 'svelte';
  export let blendRatio = 0.5;
  export let phaseMode = "Kerr";
  export let personaState = "neutral";
  export let viewMode = "quilt";
  export let diagnostics = {};
  const dispatch = createEventDispatcher();
  // Dispatch an update event whenever any setting changes
  $: dispatch('update', { blendRatio, phaseMode, personaState, viewMode });
</script>

<div class="control-panel">
  <!-- Blend Ratio Control -->
  <div class="control-group">
    <label for="blendRatio">Blend Ratio: {blendRatio.toFixed(2)}</label>
    <input id="blendRatio" type="range" min="0" max="1" step="0.01" bind:value={blendRatio} />
  </div>
  <!-- Phase Mode Control -->
  <div class="control-group">
    <label for="phaseMode">Phase Mode:</label>
    <select id="phaseMode" bind:value={phaseMode}>
      <option value="Kerr">Kerr</option>
      <option value="Soliton">Soliton</option>
    </select>
  </div>
  <!-- Persona State Control -->
  <div class="control-group">
    <label for="personaState">Persona State:</label>
    <select id="personaState" bind:value={personaState}>
      <option value="neutral">Neutral</option>
      <option value="happy">Happy</option>
      <option value="sad">Sad</option>
      <option value="angry">Angry</option>
    </select>
  </div>
  <!-- View Mode Control -->
  <div class="control-group">
    <label for="viewMode">View Mode:</label>
    <select id="viewMode" bind:value={viewMode}>
      <option value="quilt">Quilt (Tiled Views)</option>
      <option value="stereo">Stereo (Side-by-Side)</option>
      <option value="depth">Depth Layers</option>
    </select>
  </div>
  <!-- Diagnostics Panel -->
  <div class="control-group">
    <details>
      <summary>Diagnostics</summary>
      {#if diagnostics && Object.keys(diagnostics).length > 0}
        <ul>
        {#each Object.entries(diagnostics) as [key, value]}
          <li><strong>{key}:</strong> {value}</li>
        {/each}
        </ul>
      {:else}
        <p>No diagnostics available.</p>
      {/if}
    </details>
  </div>
</div>

<style>
  .control-panel {
    background: rgba(0, 0, 0, 0.6);
    color: #fff;
    padding: 10px;
    border-radius: 5px;
    max-width: 280px;
    font-family: sans-serif;
    font-size: 14px;
  }
  .control-group {
    margin-bottom: 8px;
  }
  .control-group label {
    display: block;
    margin-bottom: 4px;
    font-weight: 500;
  }
  .control-group input[type="range"] {
    width: 100%;
  }
  .control-group select, .control-group input[type="range"] {
    padding: 2px 4px;
    font-size: 14px;
    width: 100%;
    box-sizing: border-box;
  }
  details summary {
    cursor: pointer;
    outline: none;
    font-weight: 500;
  }
  details ul {
    list-style: none;
    padding-left: 0;
    margin: 4px 0;
    font-size: 13px;
  }
  details li {
    margin: 2px 0;
  }
  details p {
    margin: 4px 0;
    font-size: 13px;
  }
</style>