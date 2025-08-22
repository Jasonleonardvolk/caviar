<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { maxDuration, needsWatermark, getPlan } from '$lib/stores/userPlan';

  export let sourceCanvasId: string = 'holo-canvas';
  export let hudTheme: 'auto'|'light'|'dark' = 'auto';
  export let hudPos: 'bl'|'bc'|'br' = 'bc'; // bottom-left/center/right by default

  let recording = false;
  let status = 'idle';
  let countdown = 0;
  let chunks: BlobPart[] = [];
  let mediaRecorder: MediaRecorder | null = null;
  let composite: HTMLCanvasElement;
  let rafId: number | null = null;
  let showToast = false;
  let toastMessage = '';
  let showGuide = true;
  let firstRecord = true;

  let fileName = '';
  let mime = '';
  let durationLimit = 10;
  let currentTheme: 'light'|'dark' = 'dark';

  // Luminance sampling for auto-theme
  function luminanceSample(canvas: HTMLCanvasElement): number {
    const s = document.createElement('canvas');
    const w = 64, h = 64; 
    s.width = w; 
    s.height = h;
    const sctx = s.getContext('2d')!;
    // Sample a band across the bottom where the HUD sits
    const sx = 0;
    const sy = Math.max(0, canvas.height - Math.round(canvas.height * 0.25));
    const sw = canvas.width;
    const sh = Math.round(canvas.height * 0.25);
    sctx.drawImage(canvas, sx, sy, sw, sh, 0, 0, w, h);
    const data = sctx.getImageData(0,0,w,h).data;
    let sum = 0;
    for (let i=0;i<data.length;i+=4){
      const r=data[i], g=data[i+1], b=data[i+2];
      sum += 0.2126*r + 0.7152*g + 0.0722*b;
    }
    return sum / (data.length/4) / 255; // 0..1
  }

  function updateThemeAuto(comp?: HTMLCanvasElement){
    if (hudTheme !== 'auto' || !comp) return;
    const L = luminanceSample(comp);
    currentTheme = (L > 0.45) ? 'dark' : 'light';
  }

  function pickMime(): string {
    const candidates = [
      'video/webm;codecs=vp9',
      'video/webm;codecs=vp8',
      'video/webm',
      'video/mp4'
    ];
    for (const m of candidates) {
      if (MediaRecorder.isTypeSupported(m)) return m;
    }
    return 'video/webm';
  }

  function drawWatermark(ctx: CanvasRenderingContext2D, w: number, h: number) {
    if (!needsWatermark()) return;
    ctx.save();
    ctx.globalAlpha = 0.35;
    ctx.font = `${Math.round(h * 0.045)}px Inter, Arial, sans-serif`;
    ctx.textAlign = 'right';
    ctx.fillStyle = '#ffffff';
    const line1 = 'iRis • hologram studio';
    const line2 = 'Created with CAVIAR';
    ctx.fillText(line1, w - 24, h - 56);
    ctx.fillText(line2, w - 24, h - 20);
    ctx.restore();
  }

  function startCompositeLoop(src: HTMLCanvasElement, comp: HTMLCanvasElement, dpr = 1) {
    const ctx = comp.getContext('2d')!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    function loop() {
      ctx.clearRect(0, 0, comp.width / dpr, comp.height / dpr);
      ctx.drawImage(src, 0, 0, comp.width / dpr, comp.height / dpr);
      drawWatermark(ctx, comp.width / dpr, comp.height / dpr);
      updateThemeAuto(comp);
      rafId = requestAnimationFrame(loop);
    }
    rafId = requestAnimationFrame(loop);
  }

  async function startRecording() {
    const src = document.getElementById(sourceCanvasId) as HTMLCanvasElement | null;
    if (!src) { 
      showNotification('Source canvas not found');
      return;
    }

    if (firstRecord) {
      showGuide = false;
      firstRecord = false;
    }

    durationLimit = maxDuration();
    composite = document.createElement('canvas');
    composite.width = src.width;
    composite.height = src.height;

    const dpr = window.devicePixelRatio || 1;
    startCompositeLoop(src, composite, dpr);

    mime = pickMime();
    const stream = composite.captureStream(30);
    chunks = [];
    mediaRecorder = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 6_000_000 });

    mediaRecorder.ondataavailable = (ev) => { 
      if (ev.data?.size) chunks.push(ev.data); 
    };
    
    mediaRecorder.onstop = () => {
      const blob = new Blob(chunks, { type: mime });
      fileName = `iris_${Date.now()}.${mime.includes('mp4') ? 'mp4' : 'webm'}`;
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; 
      a.download = fileName; 
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 10_000);
      stopCompositeLoop();
      recording = false; 
      status = 'saved';
      showNotification(`Saved → Downloads/${fileName}`);
    };

    mediaRecorder.start(1000);
    status = 'recording';
    recording = true;

    // Countdown + auto-stop
    countdown = durationLimit;
    const iv = setInterval(() => {
      countdown--;
      if (countdown <= 0) {
        clearInterval(iv);
        stopRecording();
      }
    }, 1000);
  }

  function stopCompositeLoop() {
    if (rafId) cancelAnimationFrame(rafId);
    rafId = null;
  }

  function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
  }

  function showNotification(msg: string) {
    toastMessage = msg;
    showToast = true;
    setTimeout(() => showToast = false, 3000);
  }

  // Keyboard shortcuts
  function handleKeydown(e: KeyboardEvent) {
    switch(e.key) {
      case ' ':
        e.preventDefault();
        if (recording) stopRecording();
        else startRecording();
        break;
      case 'f':
      case 'F':
        if (document.fullscreenElement) {
          document.exitFullscreen();
        } else {
          document.documentElement.requestFullscreen();
        }
        break;
    }
  }

  onMount(() => {
    window.addEventListener('keydown', handleKeydown);
  });

  onDestroy(() => {
    stopCompositeLoop();
    window.removeEventListener('keydown', handleKeydown);
  });

  // Format time display
  function formatTime(seconds: number): string {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
  }
</script>

<style>
  .recorder-wrap {
    position: absolute; 
    left: 0; 
    right: 0; 
    bottom: 12px; 
    pointer-events: none;
    display: flex; 
    justify-content: flex-end; 
    padding: 0 16px;
    z-index: 100;
  }
  
  .recorder-wrap.bc { 
    justify-content: center; 
  }
  
  .recorder-wrap.bl { 
    justify-content: flex-start; 
  }

  .recorder-bar {
    pointer-events: auto;
    display: flex; 
    gap: 10px; 
    align-items: center;
    padding: 10px 12px; 
    border-radius: 14px;
    backdrop-filter: blur(10px) saturate(120%);
    border: 1px solid rgba(255,255,255,0.12);
    box-shadow: 0 6px 24px rgba(0,0,0,0.35);
    transition: all 0.3s ease;
  }

  /* Glass themes with contrast */
  .recorder-bar.dark {
    background: linear-gradient(180deg, rgba(15,15,20,0.85), rgba(10,10,14,0.75));
    color: #f2f6ff;
  }
  
  .recorder-bar.light {
    background: linear-gradient(180deg, rgba(255,255,255,0.85), rgba(245,247,255,0.78));
    color: #0b0f19;
    border-color: rgba(0,0,0,0.15);
    text-shadow: 0 1px 0 rgba(255,255,255,0.4);
  }

  .pill {
    font: 600 12px/1.2 Inter, system-ui, sans-serif;
    padding: 6px 10px; 
    border-radius: 999px;
    background: rgba(0,0,0,0.25);
    border: 1px solid rgba(255,255,255,0.15);
  }
  
  .recorder-bar.light .pill { 
    background: rgba(255,255,255,0.65); 
    border-color: rgba(0,0,0,0.1); 
  }

  .pill.plan {
    background: linear-gradient(135deg, #10b981, #00ff88);
    color: #000;
    font-weight: 700;
  }

  .btn {
    padding: 8px 14px; 
    border-radius: 12px; 
    border: 0; 
    cursor: pointer; 
    font-weight: 800;
    box-shadow: 0 2px 10px rgba(0,0,0,0.25);
    transition: all 0.2s ease;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  
  .btn:hover {
    transform: scale(1.05);
  }
  
  .start { 
    background: #10B981; 
    color: #041307; 
  }
  
  .stop { 
    background: #EF4444; 
    color: #fff; 
  }

  .countdown-ring {
    position: relative;
    display: inline-flex;
  }

  .countdown-text {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    font-size: 18px;
    font-weight: bold;
  }

  /* Toast notification */
  .toast {
    position: fixed;
    bottom: 80px;
    left: 16px;
    background: rgba(0,0,0,0.9);
    color: white;
    padding: 12px 20px;
    border-radius: 8px;
    font-size: 14px;
    animation: slideIn 0.3s ease;
    z-index: 200;
  }

  @keyframes slideIn {
    from { transform: translateY(100%); opacity: 0; }
    to { transform: translateY(0); opacity: 1; }
  }

  /* Guide overlay */
  .guide {
    position: absolute;
    top: 20px;
    left: 50%;
    transform: translateX(-50%);
    background: rgba(0,0,0,0.8);
    color: white;
    padding: 12px 24px;
    border-radius: 12px;
    font-size: 14px;
    z-index: 150;
    cursor: pointer;
    animation: pulse 2s infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.8; }
  }

  .fullscreen-hint {
    position: absolute;
    top: 20px;
    right: 20px;
    background: rgba(0,0,0,0.7);
    color: white;
    padding: 8px 12px;
    border-radius: 8px;
    font-size: 12px;
    cursor: pointer;
  }

  /* Responsive */
  @media (max-width: 768px) {
    .recorder-bar {
      flex-direction: column;
      gap: 8px;
    }
    
    .guide {
      font-size: 12px;
      padding: 8px 16px;
    }
  }
</style>

<!-- Guide for first-time users -->
{#if showGuide && firstRecord}
  <div class="guide" on:click={() => showGuide = false}>
    ① Choose Look (top-right) ② Press Start (bottom) ③ Clip saves to Downloads
  </div>
{/if}

<!-- Fullscreen hint -->
{#if !document.fullscreenElement}
  <div class="fullscreen-hint" on:click={() => document.documentElement.requestFullscreen()}>
    ↗ Try Fullscreen
  </div>
{/if}

<!-- Main recorder HUD -->
<div class="recorder-wrap {hudPos}">
  <div class="recorder-bar {hudTheme === 'auto' ? currentTheme : hudTheme}">
    <div class="pill plan">
      Plan: {getPlan().name.toUpperCase()}
    </div>
    
    <div class="pill limit">
      Limit: {formatTime(maxDuration())}
      • {needsWatermark() ? 'Watermark' : 'No Watermark'}
    </div>

    {#if status === 'recording'}
      <div class="countdown-ring">
        <button class="btn stop" on:click={stopRecording}>
          ● Stop
        </button>
        <span class="countdown-text">{countdown}s</span>
      </div>
    {:else}
      <button class="btn start" on:click={startRecording}>
        ▸ Start
      </button>
    {/if}
  </div>
</div>

<!-- Toast notification -->
{#if showToast}
  <div class="toast">
    {toastMessage}
  </div>
{/if}

<!-- Keyboard shortcuts info -->
<div style="position: absolute; bottom: 4px; right: 16px; color: rgba(255,255,255,0.3); font-size: 10px;">
  Space: Start/Stop • F: Fullscreen
</div>