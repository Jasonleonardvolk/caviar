<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { exportVideo } from '$lib/utils/exportVideo';
  import { userPlan } from '$lib/stores/userPlan';
  import { get } from 'svelte/store';
  
  export let canvas: HTMLCanvasElement | undefined = undefined;
  
  let isRecording = false;
  let recordingTime = 0;
  let recordingTimer: number | null = null;
  let mediaRecorder: MediaRecorder | null = null;
  let chunks: Blob[] = [];
  
  $: currentPlan = $userPlan;
  $: maxRecordTime = currentPlan?.features?.maxVideoLengthSec || 10;
  $: hasWatermark = currentPlan?.features?.watermark ?? true;
  $: canExportKit = currentPlan?.features?.exportARkit ?? false;
  
  async function startRecording() {
    if (!canvas) {
      console.error('No canvas element provided');
      return;
    }
    
    try {
      // Reset state
      chunks = [];
      recordingTime = 0;
      
      // Get canvas stream at 30/60 fps
      const stream = canvas.captureStream(currentPlan?.tier >= 1 ? 60 : 30);
      
      // Setup MediaRecorder with H.264 for iOS compatibility
      const options: MediaRecorderOptions = {
        mimeType: 'video/webm;codecs=h264',
        videoBitsPerSecond: currentPlan?.tier >= 2 ? 12_000_000 : 8_000_000
      };
      
      // Fallback for Safari
      if (!MediaRecorder.isTypeSupported(options.mimeType)) {
        options.mimeType = 'video/mp4';
      }
      
      mediaRecorder = new MediaRecorder(stream, options);
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunks.push(event.data);
        }
      };
      
      mediaRecorder.onstop = async () => {
        const blob = new Blob(chunks, { type: 'video/mp4' });
        await processRecording(blob);
      };
      
      // Start recording
      mediaRecorder.start(1000); // Capture in 1-second chunks
      isRecording = true;
      
      // Start timer
      recordingTimer = window.setInterval(() => {
        recordingTime++;
        
        // Auto-stop at max time
        if (recordingTime >= maxRecordTime) {
          stopRecording();
        }
      }, 1000);
      
    } catch (error) {
      console.error('Failed to start recording:', error);
      alert('Recording failed. Please check permissions.');
    }
  }
  
  function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
      mediaRecorder.stop();
    }
    
    if (recordingTimer) {
      clearInterval(recordingTimer);
      recordingTimer = null;
    }
    
    isRecording = false;
  }
  
  async function processRecording(blob: Blob) {
    try {
      // Process with exportVideo utility
      const processedBlob = await exportVideo(blob, {
        watermark: hasWatermark,
        resolution: currentPlan?.tier >= 1 ? '1920x1080' : '1080x1920',
        format: 'mp4'
      });
      
      // Create download link
      const url = URL.createObjectURL(processedBlob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `iris_hologram_${Date.now()}.mp4`;
      a.click();
      
      // Cleanup
      setTimeout(() => URL.revokeObjectURL(url), 100);
      
      // Track event
      if (window.psiTelemetry) {
        window.psiTelemetry.track('video_exported', {
          duration: recordingTime,
          plan: currentPlan?.id,
          watermark: hasWatermark
        });
      }
      
    } catch (error) {
      console.error('Failed to process recording:', error);
      alert('Failed to process video. Please try again.');
    }
  }
  
  function formatTime(seconds: number): string {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  }
  
  onDestroy(() => {
    if (isRecording) {
      stopRecording();
    }
  });
</script>

<div class="hologram-recorder">
  <div class="recorder-controls">
    {#if !isRecording}
      <button 
        class="record-button"
        on:click={startRecording}
        disabled={!canvas}
      >
        <span class="record-icon">🔴</span>
        Record
      </button>
      
      {#if currentPlan?.tier === 0}
        <div class="upgrade-hint">
          <small>
            {maxRecordTime}s limit • Watermark
            <a href="/pricing">Upgrade for more</a>
          </small>
        </div>
      {/if}
    {:else}
      <button 
        class="stop-button"
        on:click={stopRecording}
      >
        <span class="stop-icon">⏹️</span>
        Stop Recording
      </button>
      
      <div class="recording-status">
        <span class="recording-indicator">🔴</span>
        <span class="recording-time">
          {formatTime(recordingTime)} / {formatTime(maxRecordTime)}
        </span>
      </div>
    {/if}
    
    {#if canExportKit}
      <button class="export-kit-button">
        <span class="export-icon">📦</span>
        Export AR Kit
      </button>
    {/if}
  </div>
  
  {#if hasWatermark && !isRecording}
    <div class="watermark-notice">
      <small>Videos will include iRis watermark • <a href="/pricing">Remove with Plus</a></small>
    </div>
  {/if}
</div>

<style>
  .hologram-recorder {
    position: relative;
    padding: 1rem;
    background: rgba(0, 0, 0, 0.8);
    border-radius: 8px;
    backdrop-filter: blur(10px);
  }
  
  .recorder-controls {
    display: flex;
    align-items: center;
    gap: 1rem;
  }
  
  .record-button,
  .stop-button,
  .export-kit-button {
    padding: 0.75rem 1.5rem;
    border: none;
    border-radius: 4px;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    transition: all 0.2s;
  }
  
  .record-button {
    background: linear-gradient(135deg, #ff4458, #ff6b6b);
    color: white;
  }
  
  .record-button:hover:not(:disabled) {
    transform: scale(1.05);
    box-shadow: 0 4px 20px rgba(255, 68, 88, 0.4);
  }
  
  .record-button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  
  .stop-button {
    background: #333;
    color: white;
  }
  
  .export-kit-button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
  }
  
  .recording-status {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: white;
    font-family: 'SF Mono', monospace;
  }
  
  .recording-indicator {
    animation: pulse 1s infinite;
  }
  
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }
  
  .recording-time {
    font-size: 1.1rem;
    font-weight: 600;
  }
  
  .upgrade-hint,
  .watermark-notice {
    margin-top: 0.5rem;
    color: #aaa;
    font-size: 0.875rem;
  }
  
  .upgrade-hint a,
  .watermark-notice a {
    color: #667eea;
    text-decoration: none;
  }
  
  .upgrade-hint a:hover,
  .watermark-notice a:hover {
    text-decoration: underline;
  }
  
  @media (max-width: 768px) {
    .recorder-controls {
      flex-direction: column;
      align-items: stretch;
    }
    
    .record-button,
    .stop-button,
    .export-kit-button {
      width: 100%;
      justify-content: center;
    }
  }
</style>
