<script lang="ts">
  import { telemetry } from '$lib/stores/telemetry';
  
  // Color coding for assist states
  function getAssistColor(state: string) {
    switch(state) {
      case 'local': return '#0f0';
      case 'remote': return '#ff0';
      case 'degraded': return '#f00';
      default: return '#fff';
    }
  }
</script>

<style>
  .hud {
    position: fixed;
    top: 0.5rem;
    right: 0.5rem;
    padding: 0.5rem;
    background: #000c;
    color: #fff;
    font: 12px/1.2 monospace;
    border-radius: 0.5rem;
    min-width: 280px;
  }
  .assist-indicator {
    display: inline-block;
    margin-left: 0.5rem;
  }
</style>

<div class="hud">
  N={$telemetry.N} | z={$telemetry.zModes} | EMA={$telemetry.frameMsEMA.toFixed(2)}ms
  <span class="assist-indicator" style="color: {getAssistColor($telemetry.assist)}">
    | assist={$telemetry.assist}
  </span>
</div>