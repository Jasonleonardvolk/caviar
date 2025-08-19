<script lang="ts">
  export let status = { audio: false, video: false };
  
  $: audioIcon = status.audio ? '🔊' : '🔇';
  $: videoIcon = status.video ? '📹' : '📷';
  $: audioClass = status.audio ? 'active' : 'inactive';
  $: videoClass = status.video ? 'active' : 'inactive';
</script>

<div class="av-status-panel">
  <h3>AV Pipeline</h3>
  
  <div class="av-indicators">
    <div class="indicator {audioClass}">
      <span class="icon">{audioIcon}</span>
      <span class="label">Audio</span>
      <span class="status">{status.audio ? 'Active' : 'Inactive'}</span>
    </div>
    
    <div class="indicator {videoClass}">
      <span class="icon">{videoIcon}</span>
      <span class="label">Video</span>
      <span class="status">{status.video ? 'Active' : 'Inactive'}</span>
    </div>
  </div>
  
  {#if status.audio || status.video}
    <div class="av-details">
      {#if status.audioLevel !== undefined}
        <div class="detail">
          <span>Audio Level</span>
          <progress value={status.audioLevel} max="100"></progress>
        </div>
      {/if}
      
      {#if status.fps !== undefined}
        <div class="detail">
          <span>Video FPS</span>
          <span class="value">{status.fps}</span>
        </div>
      {/if}
      
      {#if status.codec}
        <div class="detail">
          <span>Codec</span>
          <span class="value">{status.codec}</span>
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
  .av-status-panel {
    background: #1e293b;
    border-radius: 8px;
    padding: 1rem;
    margin-top: 1rem;
  }
  
  h3 {
    margin: 0 0 1rem;
    font-size: 1rem;
    color: #e2e8f0;
  }
  
  .av-indicators {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.75rem;
  }
  
  .indicator {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 0.75rem;
    background: #0f172a;
    border-radius: 4px;
    border: 1px solid #334155;
    transition: all 0.3s;
  }
  
  .indicator.active {
    border-color: #10b981;
    background: rgba(16, 185, 129, 0.1);
  }
  
  .indicator.inactive {
    opacity: 0.6;
  }
  
  .indicator .icon {
    font-size: 1.5rem;
    margin-bottom: 0.25rem;
  }
  
  .indicator .label {
    font-size: 0.875rem;
    color: #94a3b8;
    margin-bottom: 0.25rem;
  }
  
  .indicator .status {
    font-size: 0.75rem;
    font-weight: 600;
    color: #e2e8f0;
  }
  
  .indicator.active .status {
    color: #10b981;
  }
  
  .av-details {
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid #334155;
  }
  
  .detail {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
    font-size: 0.875rem;
    color: #94a3b8;
  }
  
  .detail .value {
    color: #e2e8f0;
    font-family: monospace;
  }
  
  progress {
    flex: 1;
    margin-left: 1rem;
    height: 8px;
    border-radius: 4px;
    overflow: hidden;
  }
  
  progress::-webkit-progress-bar {
    background: #0f172a;
    border-radius: 4px;
  }
  
  progress::-webkit-progress-value {
    background: #3b82f6;
    border-radius: 4px;
  }
</style>
