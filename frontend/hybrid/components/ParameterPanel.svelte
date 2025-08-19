<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  
  export let state = {
    phaseMode: 'Kerr',
    personaState: 'neutral',
    viewMode: 'quilt',
    blendRatio: 0.5
  };
  
  const dispatch = createEventDispatcher();
  
  const phaseModes = ['Kerr', 'Soliton', 'Tensor', 'Phase-Coherent'];
  const personaStates = ['neutral', 'happy', 'sad', 'excited'];
  const viewModes = ['quilt', 'stereo', 'depth', 'normal'];
  
  function updateParameter(key: string, value: any) {
    state[key] = value;
    dispatch('update', { [key]: value });
  }
</script>

<div class="parameter-panel control-panel">
  <h3>Control Parameters</h3>
  
  <div class="control-group">
    <label for="phaseMode">Phase Mode</label>
    <select 
      id="phaseMode" 
      value={state.phaseMode}
      on:change={(e) => updateParameter('phaseMode', e.target.value)}
    >
      {#each phaseModes as mode}
        <option value={mode}>{mode}</option>
      {/each}
    </select>
  </div>
  
  <div class="control-group">
    <label for="personaState">Persona State</label>
    <select 
      id="personaState"
      value={state.personaState}
      on:change={(e) => updateParameter('personaState', e.target.value)}
    >
      {#each personaStates as persona}
        <option value={persona}>{persona}</option>
      {/each}
    </select>
  </div>
  
  <div class="control-group">
    <label for="viewMode">View Mode</label>
    <select 
      id="viewMode"
      value={state.viewMode}
      on:change={(e) => updateParameter('viewMode', e.target.value)}
    >
      {#each viewModes as mode}
        <option value={mode}>{mode}</option>
      {/each}
    </select>
  </div>
  
  <div class="control-group">
    <label for="blendRatio">
      Blend Ratio: {(state.blendRatio * 100).toFixed(0)}%
    </label>
    <input 
      type="range"
      id="blendRatio"
      min="0"
      max="1"
      step="0.01"
      value={state.blendRatio}
      on:input={(e) => updateParameter('blendRatio', parseFloat(e.target.value))}
    />
  </div>
</div>

<style>
  .parameter-panel {
    background: #1e293b;
    border-radius: 8px;
    padding: 1rem;
  }
  
  h3 {
    margin: 0 0 1rem;
    font-size: 1rem;
    color: #e2e8f0;
  }
  
  .control-group {
    margin-bottom: 1rem;
  }
  
  label {
    display: block;
    margin-bottom: 0.25rem;
    font-size: 0.875rem;
    color: #94a3b8;
  }
  
  select,
  input[type="range"] {
    width: 100%;
    padding: 0.5rem;
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 4px;
    color: #e2e8f0;
    font-size: 0.875rem;
  }
  
  select:focus,
  input[type="range"]:focus {
    outline: none;
    border-color: #3b82f6;
  }
  
  input[type="range"] {
    padding: 0.25rem;
  }
</style>
