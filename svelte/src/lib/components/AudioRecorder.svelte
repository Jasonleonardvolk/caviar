<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { createEventDispatcher } from 'svelte';
  import { fade, scale } from 'svelte/transition';
  import { tweened } from 'svelte/motion';
  import { cubicOut } from 'svelte/easing';
  import { psiMemoryStore } from '../stores/psiMemory';
  import { audioAnalysisStore } from '../stores/audioAnalysis';
  import { 
    connectAudioWS, 
    sendAudioChunk, 
    finalizeStream,
    disconnectStream,
    transcript as transcriptStore,
    spectral as spectralStore,
    emotion as emotionStore,
    hologramHint as hologramHintStore,
    streamState
  } from '../stores/audio';
  
  const dispatch = createEventDispatcher();
  
  // Recording state
  let isRecording = false;
  let isPaused = false;
  let isProcessing = false;
  let recordingTime = 0;
  let audioBlob: Blob | null = null;
  let errorMessage = '';
  
  // Audio components
  let mediaRecorder: MediaRecorder | null = null;
  let audioContext: AudioContext | null = null;
  let analyser: AnalyserNode | null = null;
  let source: MediaStreamAudioSourceNode | null = null;
  let stream: MediaStream | null = null;
  let audioChunks: Blob[] = [];
  
  // Streaming state
  let isStreaming = false;
  let streamProcessor: ScriptProcessorNode | null = null;
  
  // Real-time analysis
  let realtimeAnalyzer: Worker | null = null;
  let visualizationCanvas: HTMLCanvasElement;
  let canvasContext: CanvasRenderingContext2D | null = null;
  let animationId: number | null = null;
  
  // Audio levels
  const audioLevel = tweened(0, { duration: 100, easing: cubicOut });
  const peakLevel = tweened(0, { duration: 2000, easing: cubicOut });
  let clipping = false;
  
  // Configuration
  export let maxDuration = 300; // 5 minutes
  export let sampleRate = 48000;
  export let enableNoiseSuppression = true;
  export let enableEchoCancellation = true;
  export let enableAutoGainControl = true;
  export let visualizationType: 'waveform' | 'spectrum' | 'psi' = 'psi';
  export let sessionId = 'recording_' + Date.now();
  export let quality: 'low' | 'medium' | 'high' = 'high';
  export let enableStreaming = true;
  
  // Quality presets
  const qualityPresets = {
    low: { bitRate: 64000, sampleRate: 16000 },
    medium: { bitRate: 128000, sampleRate: 24000 },
    high: { bitRate: 256000, sampleRate: 48000 }
  };
  
  // Timer
  let timerInterval: number | null = null;
  
  // ψ-state visualization
  let psiPhase = 0;
  let phaseCoherence = 0;
  let emotionalResonance = {
    excitement: 0,
    calmness: 0.5,
    energy: 0,
    clarity: 0
  };
  
  const config = {
    maxDuration,
    sampleRate,
    enableNoiseSuppression,
    enableEchoCancellation,
    enableAutoGainControl,
    visualizationType,
    sessionId,
    quality,
    enableStreaming
  };
  
  onMount(async () => {
    await initializeAudio();
    setupRealtimeAnalyzer();
    connectAudioWS(); // Connect WebSocket
    
    if (visualizationCanvas) {
      canvasContext = visualizationCanvas.getContext('2d');
      setupCanvas();
    }
  });
  
  onDestroy(() => {
    cleanup();
    disconnectStream(); // Disconnect WebSocket
  });
  
  async function initializeAudio() {
    try {
      // Initialize AudioContext
      audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: qualityPresets[quality].sampleRate
      });
      
      // Create analyser node
      analyser = audioContext.createAnalyser();
      analyser.fftSize = 2048;
      analyser.smoothingTimeConstant = 0.8;
      
    } catch (error) {
      console.error('Failed to initialize audio:', error);
      errorMessage = 'Failed to initialize audio system';
    }
  }
  
  function setupRealtimeAnalyzer() {
    // Initialize Web Worker for real-time analysis
    realtimeAnalyzer = new Worker(new URL('../workers/audioAnalyzer.worker.ts', import.meta.url), {
      type: 'module'
    });
    
    realtimeAnalyzer.onmessage = (event) => {
      const { type, data } = event.data;
      
      switch (type) {
        case 'analysis':
          handleRealtimeAnalysis(data);
          break;
        case 'psi_update':
          updatePsiState(data);
          break;
        case 'emotion_change':
          handleEmotionChange(data);
          break;
        case 'error':
          console.error('Worker error:', data);
          break;
      }
    };
    
    // Configure worker
    realtimeAnalyzer.postMessage({
      type: 'configure',
      config: {
        sampleRate: qualityPresets[quality].sampleRate,
        sessionId,
        quality
      }
    });
  }
  
  async function startRecording() {
    try {
      errorMessage = '';
      
      // Get user media with constraints
      const constraints = {
        audio: {
          echoCancellation: enableEchoCancellation,
          noiseSuppression: enableNoiseSuppression,
          autoGainControl: enableAutoGainControl,
          sampleRate: qualityPresets[quality].sampleRate,
          channelCount: 1
        }
      };
      
      stream = await navigator.mediaDevices.getUserMedia(constraints);
      
      // Connect to audio context
      if (audioContext && analyser) {
        source = audioContext.createMediaStreamSource(stream);
        source.connect(analyser);
        
        // Start visualization
        startVisualization();
      }
      
      // Set up streaming processor
      if (audioContext && config.enableStreaming) {
        setupStreamingProcessor();
        isStreaming = true;
      }
      
      // Setup MediaRecorder
      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus') 
        ? 'audio/webm;codecs=opus' 
        : 'audio/webm';
        
      mediaRecorder = new MediaRecorder(stream, {
        mimeType,
        audioBitsPerSecond: qualityPresets[quality].bitRate
      });
      
      audioChunks = [];
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunks.push(event.data);
          
          // Send chunk to worker for real-time analysis
          if (realtimeAnalyzer && event.data.size > 0) {
            event.data.arrayBuffer().then(buffer => {
              realtimeAnalyzer?.postMessage({
                type: 'analyze_chunk',
                buffer
              }, [buffer]);
            });
          }
        }
      };
      
      mediaRecorder.onstop = handleRecordingStop;
      
      mediaRecorder.onerror = (event) => {
        console.error('MediaRecorder error:', event);
        errorMessage = 'Recording error occurred';
        stopRecording();
      };
      
      // Start recording
      mediaRecorder.start(100); // Collect data every 100ms
      isRecording = true;
      isPaused = false;
      
      // Start timer
      startTimer();
      
      // Dispatch event
      dispatch('recordingstart', { sessionId });
      
    } catch (error) {
      console.error('Failed to start recording:', error);
      errorMessage = error.message || 'Failed to access microphone';
      isRecording = false;
    }
  }
  
  function setupStreamingProcessor() {
    if (!audioContext || !source) return;
    
    // Create script processor for streaming
    const bufferSize = 4096;
    streamProcessor = audioContext.createScriptProcessor(bufferSize, 1, 1);
    
    streamProcessor.onaudioprocess = (event) => {
      const inputData = event.inputBuffer.getChannelData(0);
      
      // Convert to 16-bit PCM
      const pcmData = new Int16Array(inputData.length);
      for (let i = 0; i < inputData.length; i++) {
        pcmData[i] = Math.max(-32768, Math.min(32767, inputData[i] * 32768));
      }
      
      // Send to WebSocket
      sendAudioChunk(pcmData.buffer);
      
      // Pass through audio
      event.outputBuffer.copyToChannel(inputData, 0);
    };
    
    // Connect nodes
    source.connect(streamProcessor);
    streamProcessor.connect(audioContext.destination);
  }
  
  function pauseRecording() {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
      mediaRecorder.pause();
      isPaused = true;
      stopTimer();
      
      dispatch('recordingpause');
    }
  }
  
  function resumeRecording() {
    if (mediaRecorder && mediaRecorder.state === 'paused') {
      mediaRecorder.resume();
      isPaused = false;
      startTimer();
      
      dispatch('recordingresume');
    }
  }
  
  function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
      mediaRecorder.stop();
      isRecording = false;
      isPaused = false;
      
      // Finalize streaming
      if (isStreaming) {
        finalizeStream();
        isStreaming = false;
      }
      
      // Disconnect streaming processor
      if (streamProcessor) {
        streamProcessor.disconnect();
        streamProcessor = null;
      }
      
      // Stop all tracks
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
      
      // Stop visualization
      stopVisualization();
      stopTimer();
      
      dispatch('recordingstop');
    }
  }
  
  async function handleRecordingStop() {
    // Create blob from chunks
    const mimeType = audioChunks[0]?.type || 'audio/webm';
    audioBlob = new Blob(audioChunks, { type: mimeType });
    
    // Process the recording
    await processRecording();
  }
  
  async function processRecording() {
    if (!audioBlob) return;
    
    isProcessing = true;
    
    try {
      // Create FormData
      const formData = new FormData();
      formData.append('file', audioBlob, `recording_${Date.now()}.webm`);
      formData.append('session_id', sessionId);
      formData.append('enhance_quality', String(quality !== 'low'));
      
      // Upload and process
      const response = await fetch('/api/v1/audio/ingest', {
        method: 'POST',
        body: formData,
        headers: {
          'client-id': sessionId
        }
      });
      
      if (!response.ok) {
        throw new Error(`Processing failed: ${response.statusText}`);
      }
      
      const result = await response.json();
      
      // Update stores
      psiMemoryStore.updateFromIngestion(result);
      audioAnalysisStore.set(result);
      
      // Dispatch success event
      dispatch('processingsuccess', {
        result,
        audioBlob,
        duration: recordingTime
      });
      
    } catch (error) {
      console.error('Processing error:', error);
      errorMessage = 'Failed to process recording';
      
      dispatch('processingerror', { error });
    } finally {
      isProcessing = false;
    }
  }
  
  function startTimer() {
    stopTimer();
    
    const startTime = Date.now() - (recordingTime * 1000);
    
    timerInterval = window.setInterval(() => {
      recordingTime = Math.floor((Date.now() - startTime) / 1000);
      
      // Check max duration
      if (recordingTime >= maxDuration) {
        stopRecording();
      }
    }, 100);
  }
  
  function stopTimer() {
    if (timerInterval) {
      clearInterval(timerInterval);
      timerInterval = null;
    }
  }
  
  function resetRecording() {
    audioBlob = null;
    audioChunks = [];
    recordingTime = 0;
    errorMessage = '';
    
    // Reset visualizations
    audioLevel.set(0);
    peakLevel.set(0);
    clipping = false;
  }
  
  // Visualization functions
  function startVisualization() {
    if (!analyser || !canvasContext) return;
    
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    const floatArray = new Float32Array(bufferLength);
    
    function draw() {
      animationId = requestAnimationFrame(draw);
      
      if (!analyser || !canvasContext) return;
      
      // Get audio data
      if (visualizationType === 'waveform') {
        analyser.getByteTimeDomainData(dataArray);
        drawWaveform(dataArray);
      } else if (visualizationType === 'spectrum') {
        analyser.getByteFrequencyData(dataArray);
        drawSpectrum(dataArray);
      } else if (visualizationType === 'psi') {
        analyser.getFloatTimeDomainData(floatArray);
        drawPsiVisualization(floatArray);
      }
      
      // Update audio levels
      updateAudioLevels(dataArray);
    }
    
    draw();
  }
  
  function stopVisualization() {
    if (animationId) {
      cancelAnimationFrame(animationId);
      animationId = null;
    }
    
    // Clear canvas
    if (canvasContext && visualizationCanvas) {
      canvasContext.clearRect(0, 0, visualizationCanvas.width, visualizationCanvas.height);
    }
  }
  
  function setupCanvas() {
    if (!visualizationCanvas || !canvasContext) return;
    
    // Set canvas size
    const dpr = window.devicePixelRatio || 1;
    const rect = visualizationCanvas.getBoundingClientRect();
    
    visualizationCanvas.width = rect.width * dpr;
    visualizationCanvas.height = rect.height * dpr;
    
    canvasContext.scale(dpr, dpr);
    
    // Set styles
    canvasContext.lineWidth = 2;
    canvasContext.lineCap = 'round';
    canvasContext.lineJoin = 'round';
  }
  
  function drawWaveform(dataArray: Uint8Array) {
    if (!canvasContext || !visualizationCanvas) return;
    
    const width = visualizationCanvas.width / window.devicePixelRatio;
    const height = visualizationCanvas.height / window.devicePixelRatio;
    
    // Clear canvas
    canvasContext.fillStyle = 'rgba(0, 0, 0, 0.1)';
    canvasContext.fillRect(0, 0, width, height);
    
    // Draw waveform
    canvasContext.beginPath();
    canvasContext.strokeStyle = `hsl(${psiPhase * 180 / Math.PI}, 70%, 50%)`;
    
    const sliceWidth = width / dataArray.length;
    let x = 0;
    
    for (let i = 0; i < dataArray.length; i++) {
      const v = dataArray[i] / 128.0;
      const y = v * height / 2;
      
      if (i === 0) {
        canvasContext.moveTo(x, y);
      } else {
        canvasContext.lineTo(x, y);
      }
      
      x += sliceWidth;
    }
    
    canvasContext.stroke();
  }
  
  function drawSpectrum(dataArray: Uint8Array) {
    if (!canvasContext || !visualizationCanvas) return;
    
    const width = visualizationCanvas.width / window.devicePixelRatio;
    const height = visualizationCanvas.height / window.devicePixelRatio;
    
    // Clear canvas
    canvasContext.fillStyle = 'rgba(0, 0, 0, 0.1)';
    canvasContext.fillRect(0, 0, width, height);
    
    // Draw spectrum bars
    const barWidth = width / dataArray.length * 2.5;
    let x = 0;
    
    for (let i = 0; i < dataArray.length; i++) {
      const barHeight = (dataArray[i] / 255) * height;
      
      // Color based on frequency and ψ-state
      const hue = (i / dataArray.length) * 120 + psiPhase * 180 / Math.PI;
      const saturation = 50 + phaseCoherence * 50;
      const lightness = 30 + (dataArray[i] / 255) * 40;
      
      canvasContext.fillStyle = `hsl(${hue}, ${saturation}%, ${lightness}%)`;
      canvasContext.fillRect(x, height - barHeight, barWidth, barHeight);
      
      x += barWidth + 1;
    }
  }
  
  function drawPsiVisualization(floatArray: Float32Array) {
    if (!canvasContext || !visualizationCanvas) return;
    
    const width = visualizationCanvas.width / window.devicePixelRatio;
    const height = visualizationCanvas.height / window.devicePixelRatio;
    const centerX = width / 2;
    const centerY = height / 2;
    
    // Fade effect
    canvasContext.fillStyle = 'rgba(0, 0, 0, 0.05)';
    canvasContext.fillRect(0, 0, width, height);
    
    // Calculate RMS for radius
    let sum = 0;
    for (let i = 0; i < floatArray.length; i++) {
      sum += floatArray[i] * floatArray[i];
    }
    const rms = Math.sqrt(sum / floatArray.length);
    const baseRadius = Math.min(width, height) * 0.3;
    const radius = baseRadius + rms * baseRadius * 2;
    
    // Draw ψ-oscillator visualization
    canvasContext.beginPath();
    
    for (let i = 0; i < 360; i += 2) {
      const angle = (i * Math.PI / 180) + psiPhase;
      const index = Math.floor(i / 360 * floatArray.length);
      const amplitude = Math.abs(floatArray[index]);
      
      // Modulate radius with waveform
      const r = radius * (1 + amplitude * 0.5 * phaseCoherence);
      
      const x = centerX + Math.cos(angle) * r;
      const y = centerY + Math.sin(angle) * r;
      
      if (i === 0) {
        canvasContext.moveTo(x, y);
      } else {
        canvasContext.lineTo(x, y);
      }
    }
    
    canvasContext.closePath();
    
    // Color based on emotional resonance
    const hue = psiPhase * 180 / Math.PI;
    const saturation = 50 + emotionalResonance.energy * 50;
    const lightness = 30 + emotionalResonance.clarity * 40;
    
    canvasContext.strokeStyle = `hsl(${hue}, ${saturation}%, ${lightness}%)`;
    canvasContext.stroke();
    
    // Inner glow effect
    const gradient = canvasContext.createRadialGradient(
      centerX, centerY, 0,
      centerX, centerY, radius
    );
    gradient.addColorStop(0, `hsla(${hue}, ${saturation}%, ${lightness}%, 0.3)`);
    gradient.addColorStop(1, `hsla(${hue}, ${saturation}%, ${lightness}%, 0)`);
    
    canvasContext.fillStyle = gradient;
    canvasContext.fill();
    
    // Draw phase indicator
    const phaseX = centerX + Math.cos(psiPhase) * radius * 0.8;
    const phaseY = centerY + Math.sin(psiPhase) * radius * 0.8;
    
    canvasContext.beginPath();
    canvasContext.arc(phaseX, phaseY, 5, 0, Math.PI * 2);
    canvasContext.fillStyle = '#fff';
    canvasContext.fill();
  }
  
  function updateAudioLevels(dataArray: Uint8Array) {
    // Calculate RMS
    let sum = 0;
    let max = 0;
    
    for (let i = 0; i < dataArray.length; i++) {
      const value = (dataArray[i] - 128) / 128;
      sum += value * value;
      max = Math.max(max, Math.abs(value));
    }
    
    const rms = Math.sqrt(sum / dataArray.length);
    
    // Update audio level
    audioLevel.set(rms);
    
    // Update peak level
    if (rms > $peakLevel) {
      peakLevel.set(rms);
    }
    
    // Check for clipping
    clipping = max > 0.95;
  }
  
  function handleRealtimeAnalysis(data: any) {
    // Update audio analysis store with real-time data
    audioAnalysisStore.updateRealtime(data);
    
    // Check for significant events
    if (data.spectral_centroid > 2000 && data.rms > 0.3) {
      dispatch('audioEvent', { type: 'high_energy', data });
    }
  }
  
  function updatePsiState(data: any) {
    psiPhase = data.psi_phase || 0;
    phaseCoherence = data.phase_coherence || 0;
    
    if (data.emotional_resonance) {
      emotionalResonance = { ...data.emotional_resonance };
    }
    
    // Update psi memory store
    psiMemoryStore.updateRealtime(data);
  }
  
  function handleEmotionChange(data: any) {
    dispatch('emotionChange', data);
  }
  
  function cleanup() {
    stopRecording();
    stopVisualization();
    stopTimer();
    
    if (realtimeAnalyzer) {
      realtimeAnalyzer.terminate();
      realtimeAnalyzer = null;
    }
    
    if (audioContext) {
      audioContext.close();
      audioContext = null;
    }
    
    if (source) {
      source.disconnect();
      source = null;
    }
    
    analyser = null;
    mediaRecorder = null;
    stream = null;
  }
  
  function formatTime(seconds: number): string {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  }
  
  function downloadRecording() {
    if (!audioBlob) return;
    
    const url = URL.createObjectURL(audioBlob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `recording_${sessionId}_${Date.now()}.webm`;
    a.click();
    URL.revokeObjectURL(url);
  }
  
  // Keyboard shortcuts
  function handleKeydown(event: KeyboardEvent) {
    if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) {
      return;
    }
    
    switch (event.key) {
      case ' ':
        event.preventDefault();
        if (!isRecording) {
          startRecording();
        } else if (isPaused) {
          resumeRecording();
        } else {
          pauseRecording();
        }
        break;
      case 'Escape':
        if (isRecording) {
          stopRecording();
        }
        break;
      case 'r':
        if (!isRecording && audioBlob) {
          resetRecording();
        }
        break;
      case 'v':
        // Cycle visualization types
        const types = ['waveform', 'spectrum', 'psi'];
        const currentIndex = types.indexOf(visualizationType);
        visualizationType = types[(currentIndex + 1) % types.length];
        break;
    }
  }
</script>

<svelte:window on:keydown={handleKeydown} />

<div class="audio-recorder" class:recording={isRecording}>
  <!-- Visualization Canvas -->
  <div class="visualization-container">
    <canvas
      bind:this={visualizationCanvas}
      class="visualization-canvas"
      class:clipping
    />
    
    <!-- Overlay Controls -->
    <div class="visualization-controls">
      <button
        class="viz-button"
        on:click={() => visualizationType = 'waveform'}
        class:active={visualizationType === 'waveform'}
        title="Waveform (V)"
      >
        <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
          <path d="M22 12l-4 4V8l4 4M2 12l4-4v8l-4-4M12 2v20"/>
        </svg>
      </button>
      <button
        class="viz-button"
        on:click={() => visualizationType = 'spectrum'}
        class:active={visualizationType === 'spectrum'}
        title="Spectrum (V)"
      >
        <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
          <rect x="2" y="10" width="4" height="12"/>
          <rect x="8" y="6" width="4" height="16"/>
          <rect x="14" y="2" width="4" height="20"/>
          <rect x="20" y="8" width="4" height="14"/>
        </svg>
      </button>
      <button
        class="viz-button"
        on:click={() => visualizationType = 'psi'}
        class:active={visualizationType === 'psi'}
        title="ψ-Oscillator (V)"
      >
        <span class="psi-symbol">ψ</span>
      </button>
    </div>
    
    <!-- ψ-State Indicators -->
    {#if visualizationType === 'psi'}
      <div class="psi-indicators" transition:fade={{ duration: 300 }}>
        <div class="indicator">
          <span class="label">Phase</span>
          <span class="value">{(psiPhase * 180 / Math.PI).toFixed(0)}°</span>
        </div>
        <div class="indicator">
          <span class="label">Coherence</span>
          <span class="value">{(phaseCoherence * 100).toFixed(0)}%</span>
        </div>
      </div>
    {/if}
  </div>
  
  <!-- Audio Level Meters -->
  <div class="level-meters">
    <div class="meter">
      <div class="meter-label">Level</div>
      <div class="meter-bar">
        <div 
          class="meter-fill"
          style="width: {$audioLevel * 100}%"
        />
        <div 
          class="meter-peak"
          style="left: {$peakLevel * 100}%"
        />
      </div>
    </div>
    
    {#if clipping}
      <div class="clipping-warning" transition:scale>
        CLIPPING
      </div>
    {/if}
  </div>
  
  <!-- Controls -->
  <div class="controls">
    <div class="timer">
      {formatTime(recordingTime)} / {formatTime(maxDuration)}
    </div>
    
    <div class="main-controls">
      {#if !isRecording && !audioBlob}
        <button
          class="record-button"
          on:click={startRecording}
          disabled={isProcessing}
          title="Start Recording (Space)"
        >
          <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
            <circle cx="12" cy="12" r="8"/>
          </svg>
          <span>Record</span>
        </button>
      {:else if isRecording}
        <button
          class="pause-button"
          on:click={isPaused ? resumeRecording : pauseRecording}
          title="{isPaused ? 'Resume' : 'Pause'} (Space)"
        >
          {#if isPaused}
            <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
              <polygon points="5 3 19 12 5 21 5 3"/>
            </svg>
            <span>Resume</span>
          {:else}
            <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
              <rect x="6" y="4" width="4" height="16"/>
              <rect x="14" y="4" width="4" height="16"/>
            </svg>
            <span>Pause</span>
          {/if}
        </button>
        
        <button
          class="stop-button"
          on:click={stopRecording}
          title="Stop Recording (Escape)"
        >
          <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
            <rect x="6" y="6" width="12" height="12"/>
          </svg>
          <span>Stop</span>
        </button>
      {:else if audioBlob && !isProcessing}
        <button
          class="reset-button"
          on:click={resetRecording}
          title="New Recording (R)"
        >
          <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
            <path d="M17.65 6.35C16.2 4.9 14.21 4 12 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08c-.82 2.33-3.04 4-5.65 4-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z"/>
          </svg>
          <span>New</span>
        </button>
        
        <button
          class="download-button"
          on:click={downloadRecording}
        >
          <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
            <path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z"/>
          </svg>
          <span>Download</span>
        </button>
      {/if}
    </div>
    
    {#if isProcessing}
      <div class="processing-indicator" transition:fade>
        <div class="spinner"></div>
        <span>Processing...</span>
      </div>
    {/if}
  </div>
  
  <!-- Live Streaming Feedback -->
  {#if $streamState.connected}
    <div class="streaming-feedback" transition:fade>
      <div class="stream-status">
        <span class="status-indicator" class:active={$streamState.streaming}></span>
        <span>{$streamState.streaming ? 'Streaming' : 'Connected'}</span>
      </div>
      
      {#if $streamState.error}
        <div class="stream-error">
          {$streamState.error}
        </div>
      {/if}
      
      <div class="live-transcript">
        <h4>Live Transcript</h4>
        <p>{$transcriptStore || 'Start speaking...'}</p>
      </div>
      
      <div class="live-analysis">
        <div class="metric">
          <span class="label">Frequency</span>
          <span class="value">{$spectralStore.centroid.toFixed(0)} Hz</span>
        </div>
        <div class="metric">
          <span class="label">Emotion</span>
          <span class="value emotion-{$emotionStore.label}">
            {$emotionStore.label} ({($emotionStore.confidence * 100).toFixed(0)}%)
          </span>
        </div>
        <div class="metric">
          <span class="label">Hologram</span>
          <div class="hologram-preview" 
            style="background: hsl({$hologramHintStore.hue}, 70%, {30 + $hologramHintStore.intensity * 40}%)"
          ></div>
        </div>
      </div>
    </div>
  {/if}
  
  <!-- Error Message -->
  {#if errorMessage}
    <div class="error-message" transition:fade>
      <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/>
      </svg>
      {errorMessage}
    </div>
  {/if}
  
  <!-- Settings Panel -->
  <details class="settings-panel">
    <summary>
      <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
        <path d="M19.14,12.94c0.04-0.3,0.06-0.61,0.06-0.94c0-0.32-0.02-0.64-0.07-0.94l2.03-1.58c0.18-0.14,0.23-0.41,0.12-0.61 l-1.92-3.32c-0.12-0.22-0.37-0.29-0.59-0.22l-2.39,0.96c-0.5-0.38-1.03-0.7-1.62-0.94L14.4,2.81c-0.04-0.24-0.24-0.41-0.48-0.41 h-3.84c-0.24,0-0.43,0.17-0.47,0.41L9.25,5.35C8.66,5.59,8.12,5.92,7.63,6.29L5.24,5.33c-0.22-0.08-0.47,0-0.59,0.22L2.74,8.87 C2.62,9.08,2.66,9.34,2.86,9.48l2.03,1.58C4.84,11.36,4.8,11.69,4.8,12s0.02,0.64,0.07,0.94l-2.03,1.58 c-0.18,0.14-0.23,0.41-0.12,0.61l1.92,3.32c0.12,0.22,0.37,0.29,0.59,0.22l2.39-0.96c0.5,0.38,1.03,0.7,1.62,0.94l0.36,2.54 c0.05,0.24,0.24,0.41,0.48,0.41h3.84c0.24,0,0.44-0.17,0.47-0.41l0.36-2.54c0.59-0.24,1.13-0.56,1.62-0.94l2.39,0.96 c0.22,0.08,0.47,0,0.59-0.22l1.92-3.32c0.12-0.22,0.07-0.47-0.12-0.61L19.14,12.94z M12,15.6c-1.98,0-3.6-1.62-3.6-3.6 s1.62-3.6,3.6-3.6s3.6,1.62,3.6,3.6S13.98,15.6,12,15.6z"/>
      </svg>
      Settings
    </summary>
    
    <div class="settings-content">
      <label>
        <span>Quality</span>
        <select bind:value={quality} disabled={isRecording}>
          <option value="low">Low (64 kbps)</option>
          <option value="medium">Medium (128 kbps)</option>
          <option value="high">High (256 kbps)</option>
        </select>
      </label>
      
      <label>
        <input 
          type="checkbox" 
          bind:checked={enableStreaming}
          disabled={isRecording}
        />
        <span>Enable Live Streaming</span>
      </label>
      
      <label>
        <input 
          type="checkbox" 
          bind:checked={enableNoiseSuppression}
          disabled={isRecording}
        />
        <span>Noise Suppression</span>
      </label>
      
      <label>
        <input 
          type="checkbox" 
          bind:checked={enableEchoCancellation}
          disabled={isRecording}
        />
        <span>Echo Cancellation</span>
      </label>
      
      <label>
        <input 
          type="checkbox" 
          bind:checked={enableAutoGainControl}
          disabled={isRecording}
        />
        <span>Auto Gain Control</span>
      </label>
    </div>
  </details>
</div>

<style>
  .audio-recorder {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    padding: 1.5rem;
    background: rgba(0, 0, 0, 0.8);
    border-radius: 12px;
    border: 1px solid rgba(102, 126, 234, 0.3);
    min-width: 400px;
    transition: all 0.3s ease;
  }
  
  .audio-recorder.recording {
    border-color: rgba(240, 147, 251, 0.6);
    box-shadow: 0 0 30px rgba(240, 147, 251, 0.3);
  }
  
  /* Visualization */
  .visualization-container {
    position: relative;
    height: 200px;
    background: rgba(0, 0, 0, 0.5);
    border-radius: 8px;
    overflow: hidden;
  }
  
  .visualization-canvas {
    width: 100%;
    height: 100%;
    display: block;
  }
  
  .visualization-canvas.clipping {
    filter: brightness(1.2) saturate(1.5);
  }
  
  .visualization-controls {
    position: absolute;
    top: 0.5rem;
    right: 0.5rem;
    display: flex;
    gap: 0.25rem;
    background: rgba(0, 0, 0, 0.7);
    padding: 0.25rem;
    border-radius: 6px;
  }
  
  .viz-button {
    padding: 0.5rem;
    background: transparent;
    border: 1px solid transparent;
    color: #999;
    border-radius: 4px;
    cursor: pointer;
    transition: all 0.2s;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  
  .viz-button:hover {
    color: #ccc;
    background: rgba(255, 255, 255, 0.1);
  }
  
  .viz-button.active {
    color: #667eea;
    border-color: #667eea;
    background: rgba(102, 126, 234, 0.2);
  }
  
  .psi-symbol {
    font-size: 1.2rem;
    font-weight: bold;
  }
  
  .psi-indicators {
    position: absolute;
    bottom: 0.5rem;
    left: 0.5rem;
    display: flex;
    gap: 1rem;
    background: rgba(0, 0, 0, 0.7);
    padding: 0.5rem 1rem;
    border-radius: 6px;
    font-size: 0.85rem;
  }
  
  .indicator {
    display: flex;
    flex-direction: column;
    align-items: center;
  }
  
  .indicator .label {
    color: #999;
    font-size: 0.75rem;
  }
  
  .indicator .value {
    color: #667eea;
    font-weight: bold;
  }
  
  /* Level Meters */
  .level-meters {
    display: flex;
    align-items: center;
    gap: 1rem;
  }
  
  .meter {
    flex: 1;
  }
  
  .meter-label {
    font-size: 0.75rem;
    color: #999;
    margin-bottom: 0.25rem;
  }
  
  .meter-bar {
    position: relative;
    height: 6px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 3px;
    overflow: hidden;
  }
  
  .meter-fill {
    position: absolute;
    top: 0;
    left: 0;
    height: 100%;
    background: linear-gradient(90deg, #4ecdc4, #44a08d);
    transition: width 0.1s ease;
  }
  
  .meter-peak {
    position: absolute;
    top: -2px;
    width: 2px;
    height: 10px;
    background: #ff6b6b;
    transition: left 2s ease;
  }
  
  .clipping-warning {
    padding: 0.25rem 0.5rem;
    background: #ff6b6b;
    color: white;
    font-size: 0.75rem;
    font-weight: bold;
    border-radius: 4px;
    animation: blink 0.5s ease infinite;
  }
  
  @keyframes blink {
    0%, 50% { opacity: 1; }
    25%, 75% { opacity: 0.5; }
  }
  
  /* Controls */
  .controls {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    align-items: center;
  }
  
  .timer {
    font-family: 'Courier New', monospace;
    font-size: 1.2rem;
    color: #ccc;
  }
  
  .main-controls {
    display: flex;
    gap: 1rem;
  }
  
  button {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.75rem 1.5rem;
    border: none;
    border-radius: 8px;
    font-size: 1rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
    color: white;
  }
  
  button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  
  .record-button {
    background: linear-gradient(45deg, #ff6b6b, #ff8e53);
  }
  
  .record-button:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 5px 20px rgba(255, 107, 107, 0.4);
  }
  
  .pause-button {
    background: linear-gradient(45deg, #feca57, #ff9ff3);
  }
  
  .stop-button {
    background: linear-gradient(45deg, #54a0ff, #667eea);
  }
  
  .reset-button {
    background: linear-gradient(45deg, #48dbfb, #0abde3);
  }
  
  .download-button {
    background: linear-gradient(45deg, #1dd1a1, #10ac84);
  }
  
  /* Processing */
  .processing-indicator {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: #667eea;
  }
  
  .spinner {
    width: 20px;
    height: 20px;
    border: 2px solid rgba(102, 126, 234, 0.3);
    border-top-color: #667eea;
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }
  
  @keyframes spin {
    to { transform: rotate(360deg); }
  }
  
  /* Streaming Feedback */
  .streaming-feedback {
    margin-top: 1rem;
    padding: 1rem;
    background: rgba(0, 0, 0, 0.6);
    border-radius: 8px;
    border: 1px solid rgba(102, 126, 234, 0.3);
  }
  
  .stream-status {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 1rem;
    color: #999;
    font-size: 0.85rem;
  }
  
  .status-indicator {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #666;
    transition: all 0.3s;
  }
  
  .status-indicator.active {
    background: #4ecdc4;
    box-shadow: 0 0 10px #4ecdc4;
    animation: pulse 1s infinite;
  }
  
  .stream-error {
    padding: 0.5rem;
    background: rgba(255, 107, 107, 0.1);
    border: 1px solid rgba(255, 107, 107, 0.3);
    border-radius: 4px;
    color: #ff6b6b;
    font-size: 0.85rem;
    margin-bottom: 1rem;
  }
  
  .live-transcript {
    margin-bottom: 1rem;
  }
  
  .live-transcript h4 {
    margin: 0 0 0.5rem 0;
    color: #667eea;
    font-size: 0.9rem;
  }
  
  .live-transcript p {
    margin: 0;
    color: #ccc;
    font-size: 0.95rem;
    line-height: 1.4;
    max-height: 3em;
    overflow-y: auto;
  }
  
  .live-analysis {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
  }
  
  .metric {
    flex: 1;
    min-width: 100px;
  }
  
  .metric .label {
    display: block;
    font-size: 0.75rem;
    color: #999;
    margin-bottom: 0.25rem;
  }
  
  .metric .value {
    display: block;
    font-size: 0.9rem;
    color: #ccc;
    font-weight: 500;
  }
  
  .emotion-excited { color: #ff6b6b !important; }
  .emotion-calm { color: #4ecdc4 !important; }
  .emotion-energetic { color: #feca57 !important; }
  .emotion-focused { color: #667eea !important; }
  .emotion-neutral { color: #999 !important; }
  
  .hologram-preview {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    transition: all 0.3s;
    box-shadow: 0 0 20px currentColor;
  }
  
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }
  
  /* Error */
  .error-message {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.75rem;
    background: rgba(255, 107, 107, 0.1);
    border: 1px solid rgba(255, 107, 107, 0.3);
    border-radius: 6px;
    color: #ff6b6b;
    font-size: 0.9rem;
  }
  
  /* Settings */
  .settings-panel {
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
  }
  
  .settings-panel summary {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    cursor: pointer;
    color: #999;
    font-size: 0.9rem;
    user-select: none;
  }
  
  .settings-panel summary:hover {
    color: #ccc;
  }
  
  .settings-content {
    margin-top: 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }
  
  .settings-content label {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: #ccc;
    font-size: 0.9rem;
  }
  
  .settings-content select {
    flex: 1;
    padding: 0.25rem 0.5rem;
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 4px;
    color: #ccc;
  }
  
  .settings-content input[type="checkbox"] {
    width: 16px;
    height: 16px;
  }
  
  /* Responsive */
  @media (max-width: 480px) {
    .audio-recorder {
      min-width: auto;
      padding: 1rem;
    }
    
    .visualization-container {
      height: 150px;
    }
    
    .main-controls {
      flex-wrap: wrap;
      justify-content: center;
    }
    
    button {
      padding: 0.5rem 1rem;
      font-size: 0.9rem;
    }
    
    button span {
      display: none;
    }
  }
</style>