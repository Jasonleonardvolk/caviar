<script lang="ts">
  import { plan, limits } from '$lib/stores/userPlan';
  import { recordCanvasToVideo } from '$lib/utils/exportVideo';
  import { onMount, onDestroy } from 'svelte';

  export let hologramCanvasSelector: string = '#hologram-canvas';
  export let fps: number = 30;
  export let defaultWatermark = 'Created with CAVIAR';
  export let recording = false;

  let targetCanvas: HTMLCanvasElement | null = null;
  let countdown = 0;
  let elapsed = 0;
  let url: string | null = null;
  let timer: any;

  function mmss(ms: number) {
    const s = Math.max(0, Math.floor(ms / 1000));
    const m = Math.floor(s / 60);
    const r = s % 60;
    return `${String(m).padStart(2, '0')}:${String(r).padStart(2, '0')}`;
  }

  async function start() {
    if (!targetCanvas) return;
    recording = true;
    countdown = $limits.maxMs;
    elapsed = 0;

    clearInterval(timer);
    const t0 = performance.now();
    timer = setInterval(() => {
      elapsed = performance.now() - t0;
      countdown = Math.max(0, $limits.maxMs - elapsed);
      if (countdown <= 0) clearInterval(timer);
    }, 200);

    const { blob, url: outUrl } = await recordCanvasToVideo({
      sourceCanvas: targetCanvas,
      fps,
      durationMs: $limits.maxMs,
      includeMic: true,
      watermark: $limits.watermark ? { text: defaultWatermark, alpha: 0.35 } : undefined
    });

    url = outUrl;
    const a = document.createElement('a');
    const stamp = new Date().toISOString().replace(/[:.]/g, '-');
    a.href = url; a.download = `caviar-${$plan}-${stamp}.mp4`;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => url && URL.revokeObjectURL(url), 10_000);
    recording = false;
  }

  function stop() { clearInterval(timer); recording = false; }

  onMount(() => {
    targetCanvas = document.querySelector(hologramCanvasSelector) as HTMLCanvasElement | null;
  });
  onDestroy(() => clearInterval(timer));
</script>

<style>
  .recorder { display: grid; gap: .75rem; grid-template-columns: 1fr auto auto; align-items: center; }
  .capsule { border: 1px solid #2b2b2b; border-radius: 12px; padding: .5rem .75rem; }
  button { border-radius: 10px; padding: .5rem .9rem; border: 1px solid #444; background: #111; color: #fff; }
  button[disabled] { opacity: .45; cursor: not-allowed; }
  .badge { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; padding: .25rem .5rem; border-radius: 8px; background:#0a0a0a; border: 1px solid #333;}
</style>

<div class="recorder capsule">
  <div>
    <div class="badge">Plan: {$plan.toUpperCase()}</div>
    <div class="badge">Limit: {mmss($limits.maxMs)} { $limits.watermark ? '• Watermark' : '' }</div>
  </div>
  <button on:click={start} disabled={recording || !targetCanvas}>🎬 Start</button>
  <button on:click={stop}  disabled={!recording}>⏹ Stop</button>
</div>

{#if recording}
  <div class="badge">Recording… time left {mmss(countdown)}</div>
{:else if url}
  <div class="badge">Saved! Check your downloads.</div>
{/if}