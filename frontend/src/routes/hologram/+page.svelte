<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import HologramRecorder from '$lib/components/HologramRecorder.svelte';
  import { detectCapabilities, prefersWebGPUHint } from '$lib/device/capabilities';
  import { initHologram } from '$lib/hologram/engineShim';

  let caps: Awaited<ReturnType<typeof detectCapabilities>> | null = null;
  let cleanup: (() => void) | null = null;
  let status = 'init';

  onMount(async () => {
    status = 'probing';
    caps = await detectCapabilities();
    status = prefersWebGPUHint(caps) ? 'webgpu' : (caps.webgl2 ? 'webgl2' : 'cpu');

    // Kick off engine (real if present, fallback if not)
    const canvas = document.getElementById('hologram-canvas') as HTMLCanvasElement | null;
    if (canvas) cleanup = await initHologram(canvas);
  });

  onDestroy(() => { try { cleanup?.(); } catch {} });
</script>

<style>
  .wrap { display:grid; gap:1rem; grid-template-columns: 1fr; }
  .bar  { display:flex; gap:.5rem; align-items:center; border:1px solid #2b2b2b; border-radius:12px; padding:.5rem .75rem; background:#0d0d0d; color:#fff; }
  .pill { padding:.2rem .5rem; border-radius:999px; border:1px solid #333; background:#111; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
  canvas { width: 100%; max-width: 540px; height:auto; aspect-ratio: 9 / 16; border-radius: 16px; display:block; background:#000; }
  .grid { display:flex; gap:.75rem; flex-wrap:wrap; }
  a.btn, button.btn { border:1px solid #444; border-radius:10px; padding:.5rem .9rem; background:#111; color:#fff; text-decoration:none; }
</style>

<div class="wrap">
  <div class="bar">
    <div class="pill">/hologram</div>
    {#if caps}
      <div class="pill">{caps.iosLike ? 'iOS-like' : 'Desktop'}</div>
      <div class="pill">{status}</div>
      {#if caps.reason}<div class="pill" title={caps.reason}>note</div>{/if}
    {:else}
      <div class="pill">probing…</div>
    {/if}
    <div class="grid" style="margin-left:auto">
      <a class="btn" href="/pricing">Pricing</a>
      <a class="btn" href="/templates">Templates</a>
      <a class="btn" href="/account/manage">Billing</a>
    </div>
  </div>

  <!-- Your render target -->
  <canvas id="hologram-canvas" width={1080} height={1920}></canvas>

  <!-- Recorder with plan-gated limits and watermarking -->
  <HologramRecorder hologramCanvasSelector="#hologram-canvas" />
</div>