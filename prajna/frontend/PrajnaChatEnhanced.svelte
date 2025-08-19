<!--
Enhanced Prajna Chat Interface with Audio/Visual Avatar
======================================================

This component integrates Prajna's chat interface with an animated avatar
and audio capabilities for a full multimodal experience.
-->

<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { writable } from 'svelte/store';
  import PrajnaAvatar from './PrajnaAvatar.svelte';
  
  // Types (same as before)
  interface PrajnaRequest {
    user_query: string;
    focus_concept?: string;
    conversation_id?: string;
    streaming?: boolean;
    audio_input?: boolean;
  }

  interface PrajnaResponse {
    answer: string;
    sources: string[];
    audit: AuditReport;
    ghost_overlays: GhostFeedback;
    context_used: string;
    processing_time: number;
    trust_score: number;
    audio_url?: string; // For TTS response
  }

  interface AuditReport {
    trust_score: number;
    alien_detections: AlienDetection[];
    phase_analysis: PhaseAnalysis;
    supported_ratio: number;
    confidence_score: number;
    recommendations: string[];
    audit_time: number;
  }

  interface AlienDetection {
    sentence: string;
    confidence: number;
    reason: string;
    suggested_fix?: string;
  }

  interface PhaseAnalysis {
    phase_drift: number;
    coherence_score: number;
    reasoning_scars: string[];
    stability_index: number;
  }

  interface GhostFeedback {
    ghost_questions: GhostQuestion[];
    reasoning_gaps: string[];
    implicit_assumptions: string[];
    completeness_score: number;
    leaps_detected: boolean;
  }

  interface GhostQuestion {
    question: string;
    confidence: number;
    context_gap: string;
  }

  interface ChatMessage {
    id: string;
    type: 'user' | 'prajna';
    content: string;
    timestamp: Date;
    response?: PrajnaResponse;
    streaming?: boolean;
    audio?: boolean;
  }

  // State
  let userInput = '';
  let focusConcept = '';
  let conversationId = generateConversationId();
  let messages: ChatMessage[] = [];
  let isLoading = false;
  let isStreaming = false;
  let currentStreamingMessage: ChatMessage | null = null;
  let websocket: WebSocket | null = null;
  
  // Audio state
  let isRecording = false;
  let mediaRecorder: MediaRecorder | null = null;
  let audioChunks: Blob[] = [];
  let audioLevel = 0;
  let audioContext: AudioContext | null = null;
  let analyser: AnalyserNode | null = null;
  let audioPlayer: HTMLAudioElement | null = null;
  
  // Avatar state
  let avatarState: 'idle' | 'listening' | 'thinking' | 'speaking' | 'processing' = 'idle';
  let avatarMood: 'neutral' | 'happy' | 'confused' | 'focused' = 'neutral';
  
  // Configuration
  const PRAJNA_API_BASE = 'http://localhost:8001';
  const WS_ENDPOINT = 'ws://localhost:8001/api/stream';
  
  // Reactive states
  const chatHistory = writable<ChatMessage[]>([]);
  const connectionStatus = writable<'connected' | 'disconnected' | 'connecting'>('disconnected');
  
  // Lifecycle
  onMount(() => {
    initializeWebSocket();
    loadChatHistory();
    initializeAudioContext();
  });

  onDestroy(() => {
    if (websocket) {
      websocket.close();
    }
    if (audioContext) {
      audioContext.close();
    }
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
      mediaRecorder.stop();
    }
  });

  // Functions
  function generateConversationId(): string {
    return `prajna_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  function initializeWebSocket() {
    connectionStatus.set('connecting');
    
    try {
      websocket = new WebSocket(WS_ENDPOINT);
      
      websocket.onopen = () => {
        console.log('🔌 Connected to Prajna WebSocket');
        connectionStatus.set('connected');
      };
      
      websocket.onmessage = (event) => {
        handleWebSocketMessage(JSON.parse(event.data));
      };
      
      websocket.onclose = () => {
        console.log('🔌 Disconnected from Prajna WebSocket');
        connectionStatus.set('disconnected');
        // Attempt reconnection after 3 seconds
        setTimeout(initializeWebSocket, 3000);
      };
      
      websocket.onerror = (error) => {
        console.error('❌ Prajna WebSocket error:', error);
        connectionStatus.set('disconnected');
      };
      
    } catch (error) {
      console.error('❌ Failed to initialize WebSocket:', error);
      connectionStatus.set('disconnected');
    }
  }

  async function initializeAudioContext() {
    try {
      audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      console.log('🎵 Audio context initialized');
    } catch (error) {
      console.error('❌ Failed to initialize audio context:', error);
    }
  }

  function handleWebSocketMessage(data: any) {
    if (data.type === 'chunk' && currentStreamingMessage) {
      // Update streaming message content
      currentStreamingMessage.content += data.content;
      messages = [...messages]; // Trigger reactivity
      
      // Update avatar state
      if (avatarState !== 'speaking') {
        avatarState = 'speaking';
      }
    } else if (data.type === 'complete' && currentStreamingMessage) {
      // Finalize streaming message
      currentStreamingMessage.streaming = false;
      currentStreamingMessage.response = {
        answer: currentStreamingMessage.content,
        sources: data.sources || [],
        audit: data.audit || {},
        ghost_overlays: data.ghost_overlays || {},
        context_used: '',
        processing_time: 0,
        trust_score: data.audit?.trust_score || 0.5,
        audio_url: data.audio_url
      };
      
      // Play audio if available
      if (data.audio_url) {
        playAudioResponse(data.audio_url);
      } else {
        avatarState = 'idle';
      }
      
      isStreaming = false;
      currentStreamingMessage = null;
      saveChatHistory();
    } else if (data.type === 'error') {
      console.error('❌ Prajna streaming error:', data.error);
      if (currentStreamingMessage) {
        currentStreamingMessage.content = `Error: ${data.error}`;
        currentStreamingMessage.streaming = false;
      }
      isStreaming = false;
      currentStreamingMessage = null;
      avatarState = 'idle';
      avatarMood = 'confused';
    }
  }

  async function startRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder = new MediaRecorder(stream);
      audioChunks = [];
      
      // Set up audio analysis
      if (audioContext) {
        const source = audioContext.createMediaStreamSource(stream);
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        source.connect(analyser);
        
        // Start monitoring audio levels
        monitorAudioLevels();
      }
      
      mediaRecorder.ondataavailable = (event) => {
        audioChunks.push(event.data);
      };
      
      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
        await sendAudioQuery(audioBlob);
        
        // Clean up
        stream.getTracks().forEach(track => track.stop());
        isRecording = false;
        avatarState = 'thinking';
      };
      
      mediaRecorder.start();
      isRecording = true;
      avatarState = 'listening';
      
    } catch (error) {
      console.error('❌ Failed to start recording:', error);
      avatarMood = 'confused';
    }
  }

  function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
      mediaRecorder.stop();
    }
  }

  function monitorAudioLevels() {
    if (!analyser) return;
    
    const dataArray = new Uint8Array(analyser.frequencyBinCount);
    
    const checkLevel = () => {
      if (!isRecording) return;
      
      analyser.getByteFrequencyData(dataArray);
      const average = dataArray.reduce((a, b) => a + b) / dataArray.length;
      audioLevel = average / 255;
      
      requestAnimationFrame(checkLevel);
    };
    
    checkLevel();
  }

  async function sendAudioQuery(audioBlob: Blob) {
    const formData = new FormData();
    formData.append('audio', audioBlob);
    formData.append('conversation_id', conversationId);
    if (focusConcept) {
      formData.append('focus_concept', focusConcept);
    }
    
    try {
      avatarState = 'processing';
      
      const response = await fetch(`${PRAJNA_API_BASE}/api/answer/audio`, {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) {
        throw new Error(`Audio query failed: ${response.status}`);
      }
      
      const prajnaResponse: PrajnaResponse = await response.json();
      
      const userMessage: ChatMessage = {
        id: `user_${Date.now()}`,
        type: 'user',
        content: prajnaResponse.context_used || '[Audio Input]',
        timestamp: new Date(),
        audio: true
      };
      
      const prajnaMessage: ChatMessage = {
        id: `prajna_${Date.now()}`,
        type: 'prajna',
        content: prajnaResponse.answer,
        timestamp: new Date(),
        response: prajnaResponse
      };
      
      messages = [...messages, userMessage, prajnaMessage];
      
      // Play audio response if available
      if (prajnaResponse.audio_url) {
        playAudioResponse(prajnaResponse.audio_url);
      } else {
        avatarState = 'idle';
      }
      
    } catch (error) {
      console.error('❌ Failed to send audio query:', error);
      avatarState = 'idle';
      avatarMood = 'confused';
    }
  }

  async function playAudioResponse(audioUrl: string) {
    try {
      avatarState = 'speaking';
      
      if (audioPlayer) {
        audioPlayer.pause();
      }
      
      audioPlayer = new Audio(audioUrl);
      
      // Monitor audio playback for visualization
      if (audioContext) {
        const source = audioContext.createMediaElementSource(audioPlayer);
        const analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        source.connect(analyser);
        source.connect(audioContext.destination);
        
        // Update audio levels during playback
        const dataArray = new Uint8Array(analyser.frequencyBinCount);
        const updateLevel = () => {
          if (audioPlayer && !audioPlayer.paused) {
            analyser.getByteFrequencyData(dataArray);
            const average = dataArray.reduce((a, b) => a + b) / dataArray.length;
            audioLevel = average / 255;
            requestAnimationFrame(updateLevel);
          } else {
            audioLevel = 0;
          }
        };
        updateLevel();
      }
      
      audioPlayer.onended = () => {
        avatarState = 'idle';
        audioLevel = 0;
      };
      
      await audioPlayer.play();
      
    } catch (error) {
      console.error('❌ Failed to play audio response:', error);
      avatarState = 'idle';
    }
  }

  async function sendQuery(useStreaming: boolean = false) {
    if (!userInput.trim()) return;

    const userMessage: ChatMessage = {
      id: `user_${Date.now()}`,
      type: 'user',
      content: userInput.trim(),
      timestamp: new Date()
    };

    messages = [...messages, userMessage];
    
    const query = userInput.trim();
    userInput = '';
    isLoading = true;
    avatarState = 'thinking';

    try {
      if (useStreaming && websocket && websocket.readyState === WebSocket.OPEN) {
        await sendStreamingQuery(query);
      } else {
        await sendRegularQuery(query);
      }
    } catch (error) {
      console.error('❌ Failed to send query:', error);
      
      const errorMessage: ChatMessage = {
        id: `error_${Date.now()}`,
        type: 'prajna',
        content: `Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
        timestamp: new Date()
      };
      
      messages = [...messages, errorMessage];
      avatarState = 'idle';
      avatarMood = 'confused';
    } finally {
      isLoading = false;
      saveChatHistory();
    }
  }

  async function sendStreamingQuery(query: string) {
    if (!websocket || websocket.readyState !== WebSocket.OPEN) {
      throw new Error('WebSocket not connected');
    }

    isStreaming = true;
    avatarState = 'processing';
    
    // Create placeholder message for streaming content
    const streamingMessage: ChatMessage = {
      id: `prajna_${Date.now()}`,
      type: 'prajna',
      content: '',
      timestamp: new Date(),
      streaming: true
    };
    
    messages = [...messages, streamingMessage];
    currentStreamingMessage = streamingMessage;

    const request: PrajnaRequest = {
      user_query: query,
      focus_concept: focusConcept || undefined,
      conversation_id: conversationId,
      streaming: true
    };

    websocket.send(JSON.stringify(request));
  }

  async function sendRegularQuery(query: string) {
    avatarState = 'processing';
    
    const request: PrajnaRequest = {
      user_query: query,
      focus_concept: focusConcept || undefined,
      conversation_id: conversationId,
      streaming: false
    };

    const response = await fetch(`${PRAJNA_API_BASE}/api/answer`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request)
    });

    if (!response.ok) {
      throw new Error(`Prajna API error: ${response.status} ${response.statusText}`);
    }

    const prajnaResponse: PrajnaResponse = await response.json();

    const prajnaMessage: ChatMessage = {
      id: `prajna_${Date.now()}`,
      type: 'prajna',
      content: prajnaResponse.answer,
      timestamp: new Date(),
      response: prajnaResponse
    };

    messages = [...messages, prajnaMessage];
    
    // Play audio if available
    if (prajnaResponse.audio_url) {
      playAudioResponse(prajnaResponse.audio_url);
    } else {
      avatarState = 'idle';
    }
    
    // Update mood based on response
    if (prajnaResponse.trust_score > 0.8) {
      avatarMood = 'happy';
    } else if (prajnaResponse.trust_score < 0.5) {
      avatarMood = 'confused';
    } else {
      avatarMood = 'neutral';
    }
  }

  function clearChat() {
    messages = [];
    conversationId = generateConversationId();
    avatarState = 'idle';
    avatarMood = 'neutral';
    saveChatHistory();
  }

  function saveChatHistory() {
    try {
      localStorage.setItem('prajna_chat_history', JSON.stringify(messages));
      localStorage.setItem('prajna_conversation_id', conversationId);
    } catch (error) {
      console.warn('Failed to save chat history:', error);
    }
  }

  function loadChatHistory() {
    try {
      const savedHistory = localStorage.getItem('prajna_chat_history');
      const savedConversationId = localStorage.getItem('prajna_conversation_id');
      
      if (savedHistory) {
        messages = JSON.parse(savedHistory).map((msg: any) => ({
          ...msg,
          timestamp: new Date(msg.timestamp)
        }));
      }
      
      if (savedConversationId) {
        conversationId = savedConversationId;
      }
    } catch (error) {
      console.warn('Failed to load chat history:', error);
    }
  }

  function getTrustScoreColor(score: number): string {
    if (score >= 0.8) return 'text-green-600';
    if (score >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  }

  function getTrustScoreLabel(score: number): string {
    if (score >= 0.8) return 'High Trust';
    if (score >= 0.6) return 'Medium Trust';
    return 'Low Trust';
  }

  function onKeydown(event: KeyboardEvent) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      sendQuery(false);
    } else if (event.key === 'Enter' && event.shiftKey) {
      event.preventDefault();
      sendQuery(true); // Streaming mode
    }
  }
</script>

<svelte:head>
  <title>Prajna - TORI's Voice</title>
</svelte:head>

<div class="prajna-chat-container">
  <!-- Header with Avatar -->
  <header class="prajna-header">
    <div class="header-content">
      <div class="avatar-section">
        <PrajnaAvatar 
          state={avatarState}
          audioLevel={audioLevel}
          mood={avatarMood}
          size="small"
          showAudioWaves={true}
          showStatusText={false}
        />
      </div>
      
      <div class="header-info">
        <h1>🧠 Prajna</h1>
        <span class="subtitle">TORI's Voice & Language Model</span>
      </div>
    </div>
    
    <div class="header-status">
      <div class="connection-status {$connectionStatus}">
        {#if $connectionStatus === 'connected'}
          🟢 Connected
        {:else if $connectionStatus === 'connecting'}
          🟡 Connecting...
        {:else}
          🔴 Disconnected
        {/if}
      </div>
      
      <button class="clear-btn" on:click={clearChat}>
        🗑️ Clear Chat
      </button>
    </div>
  </header>

  <!-- Chat Messages -->
  <main class="chat-messages">
    {#each messages as message (message.id)}
      <div class="message {message.type}">
        <div class="message-header">
          <span class="sender">
            {message.type === 'user' ? '👤 You' : '🧠 Prajna'}
            {#if message.audio}
              <span class="audio-indicator">🎤</span>
            {/if}
          </span>
          <span class="timestamp">
            {message.timestamp.toLocaleTimeString()}
          </span>
        </div>
        
        <div class="message-content">
          {#if message.streaming}
            <div class="streaming-content">
              {message.content}<span class="cursor">|</span>
            </div>
          {:else}
            <div class="static-content">
              {message.content}
            </div>
          {/if}
        </div>

        {#if message.response && message.type === 'prajna'}
          <div class="response-metadata">
            <!-- Trust Score Display -->
            <div class="trust-score">
              <span class="label">Trust Score:</span>
              <span class="score {getTrustScoreColor(message.response.trust_score)}">
                {(message.response.trust_score * 100).toFixed(0)}%
                ({getTrustScoreLabel(message.response.trust_score)})
              </span>
            </div>

            <!-- Sources -->
            {#if message.response.sources.length > 0}
              <div class="sources">
                <span class="label">Sources:</span>
                <div class="source-list">
                  {#each message.response.sources as source}
                    <span class="source-tag">{source}</span>
                  {/each}
                </div>
              </div>
            {/if}

            <!-- Audit Information -->
            {#if message.response.audit.alien_detections?.length > 0}
              <details class="audit-details">
                <summary class="audit-summary">
                  ⚠️ {message.response.audit.alien_detections.length} Potential Issues Detected
                </summary>
                
                <div class="alien-detections">
                  {#each message.response.audit.alien_detections as detection}
                    <div class="alien-detection">
                      <div class="detection-sentence">"{detection.sentence}"</div>
                      <div class="detection-reason">
                        <strong>Issue:</strong> {detection.reason}
                        <span class="confidence">(Confidence: {(detection.confidence * 100).toFixed(0)}%)</span>
                      </div>
                      {#if detection.suggested_fix}
                        <div class="suggested-fix">
                          <strong>Suggestion:</strong> {detection.suggested_fix}
                        </div>
                      {/if}
                    </div>
                  {/each}
                </div>
              </details>
            {/if}

            <!-- Ghost Feedback -->
            {#if message.response.ghost_overlays.leaps_detected}
              <details class="ghost-details">
                <summary class="ghost-summary">
                  👻 Reasoning Analysis
                </summary>
                
                <div class="ghost-feedback">
                  <div class="completeness">
                    <strong>Completeness:</strong> 
                    {(message.response.ghost_overlays.completeness_score * 100).toFixed(0)}%
                  </div>
                  
                  {#if message.response.ghost_overlays.ghost_questions?.length > 0}
                    <div class="ghost-questions">
                      <strong>Implicit Questions:</strong>
                      <ul>
                        {#each message.response.ghost_overlays.ghost_questions as ghostQ}
                          <li>{ghostQ.question}</li>
                        {/each}
                      </ul>
                    </div>
                  {/if}
                  
                  {#if message.response.ghost_overlays.reasoning_gaps?.length > 0}
                    <div class="reasoning-gaps">
                      <strong>Reasoning Gaps:</strong>
                      <ul>
                        {#each message.response.ghost_overlays.reasoning_gaps as gap}
                          <li>{gap}</li>
                        {/each}
                      </ul>
                    </div>
                  {/if}
                </div>
              </details>
            {/if}

            <!-- Performance Metrics -->
            <div class="performance-metrics">
              <span class="metric">
                ⏱️ {message.response.processing_time.toFixed(2)}s
              </span>
              <span class="metric">
                📊 Support: {(message.response.audit.supported_ratio * 100).toFixed(0)}%
              </span>
              {#if message.response.audit.phase_analysis}
                <span class="metric">
                  🌊 Stability: {(message.response.audit.phase_analysis.stability_index * 100).toFixed(0)}%
                </span>
              {/if}
            </div>
          </div>
        {/if}
      </div>
    {/each}

    {#if isLoading && !isStreaming}
      <div class="loading-indicator">
        <PrajnaAvatar 
          state="thinking"
          audioLevel={0}
          mood="focused"
          size="small"
          showAudioWaves={false}
          showStatusText={true}
        />
      </div>
    {/if}
  </main>

  <!-- Input Area -->
  <footer class="input-area">
    <div class="input-controls">
      <input 
        type="text"
        bind:value={focusConcept}
        placeholder="Focus concept (optional)"
        class="focus-input"
      />
    </div>
    
    <div class="main-input">
      <textarea
        bind:value={userInput}
        placeholder="Ask Prajna anything about your TORI knowledge..."
        class="query-input"
        on:keydown={onKeydown}
        disabled={isLoading || isRecording}
        rows="3"
      ></textarea>
      
      <div class="input-actions">
        <button 
          class="send-btn regular"
          on:click={() => sendQuery(false)}
          disabled={isLoading || !userInput.trim() || isRecording}
        >
          📤 Send
        </button>
        
        <button 
          class="send-btn streaming"
          on:click={() => sendQuery(true)}
          disabled={isLoading || !userInput.trim() || $connectionStatus !== 'connected' || isRecording}
          title="Stream response (Shift+Enter)"
        >
          ⚡ Stream
        </button>
        
        <button
          class="send-btn audio"
          on:click={() => isRecording ? stopRecording() : startRecording()}
          disabled={isLoading}
          class:recording={isRecording}
          title="Voice input"
        >
          {isRecording ? '⏹️' : '🎤'} {isRecording ? 'Stop' : 'Voice'}
        </button>
      </div>
    </div>
    
    <div class="input-help">
      <span>💡 Enter to send • Shift+Enter to stream • Click mic for voice • Prajna speaks only from your TORI knowledge</span>
    </div>
  </footer>
</div>

<style>
  .prajna-chat-container {
    display: flex;
    flex-direction: column;
    height: 100vh;
    max-width: 1200px;
    margin: 0 auto;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    overflow: hidden;
    background: #fafafa;
  }

  .prajna-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  }

  .header-content {
    display: flex;
    align-items: center;
    gap: 1rem;
  }

  .avatar-section {
    display: flex;
    align-items: center;
  }

  .header-info h1 {
    margin: 0;
    font-size: 1.5rem;
    font-weight: 600;
  }

  .subtitle {
    font-size: 0.9rem;
    opacity: 0.9;
  }

  .header-status {
    display: flex;
    gap: 1rem;
    align-items: center;
  }

  .connection-status {
    font-size: 0.9rem;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    background: rgba(255,255,255,0.2);
  }

  .clear-btn {
    background: rgba(255,255,255,0.2);
    color: white;
    border: none;
    padding: 0.5rem 1rem;
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.9rem;
  }

  .clear-btn:hover {
    background: rgba(255,255,255,0.3);
  }

  .chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 1rem;
    background: white;
  }

  .message {
    margin-bottom: 1.5rem;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid;
  }

  .message.user {
    border-left-color: #667eea;
    background: #f8f9ff;
    margin-left: 2rem;
  }

  .message.prajna {
    border-left-color: #764ba2;
    background: #faf8ff;
    margin-right: 2rem;
  }

  .message-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.5rem;
    font-size: 0.9rem;
    color: #666;
  }

  .sender {
    font-weight: 600;
  }

  .audio-indicator {
    margin-left: 0.25rem;
    font-size: 0.8rem;
  }

  .message-content {
    margin-bottom: 1rem;
    line-height: 1.6;
  }

  .streaming-content {
    font-family: monospace;
  }

  .cursor {
    animation: blink 1s infinite;
  }

  @keyframes blink {
    0%, 50% { opacity: 1; }
    51%, 100% { opacity: 0; }
  }

  .response-metadata {
    border-top: 1px solid #e0e0e0;
    padding-top: 1rem;
    margin-top: 1rem;
  }

  .trust-score {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.75rem;
  }

  .trust-score .label {
    font-weight: 600;
    color: #333;
  }

  .trust-score .score {
    font-weight: 600;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    background: rgba(0,0,0,0.05);
  }

  .text-green-600 { color: #059669; }
  .text-yellow-600 { color: #d97706; }
  .text-red-600 { color: #dc2626; }

  .sources {
    margin-bottom: 0.75rem;
  }

  .sources .label {
    font-weight: 600;
    color: #333;
    display: block;
    margin-bottom: 0.25rem;
  }

  .source-list {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
  }

  .source-tag {
    background: #e0e7ff;
    color: #3730a3;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.8rem;
  }

  .audit-details, .ghost-details {
    margin: 0.75rem 0;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
  }

  .audit-summary, .ghost-summary {
    padding: 0.5rem;
    background: #f9fafb;
    cursor: pointer;
    font-weight: 600;
  }

  .alien-detections, .ghost-feedback {
    padding: 0.75rem;
  }

  .alien-detection {
    margin-bottom: 1rem;
    padding: 0.75rem;
    background: #fef2f2;
    border: 1px solid #fecaca;
    border-radius: 4px;
  }

  .detection-sentence {
    font-style: italic;
    margin-bottom: 0.5rem;
    color: #374151;
  }

  .detection-reason, .suggested-fix {
    font-size: 0.9rem;
    margin-bottom: 0.25rem;
  }

  .confidence {
    color: #6b7280;
    font-size: 0.8rem;
  }

  .performance-metrics {
    display: flex;
    gap: 1rem;
    margin-top: 0.75rem;
    font-size: 0.8rem;
    color: #6b7280;
  }

  .loading-indicator {
    display: flex;
    justify-content: center;
    padding: 2rem;
  }

  .input-area {
    background: #f9fafb;
    border-top: 1px solid #e0e0e0;
    padding: 1rem;
  }

  .input-controls {
    margin-bottom: 0.5rem;
  }

  .focus-input {
    width: 100%;
    padding: 0.5rem;
    border: 1px solid #d1d5db;
    border-radius: 4px;
    font-size: 0.9rem;
  }

  .main-input {
    display: flex;
    gap: 0.75rem;
    align-items: flex-end;
  }

  .query-input {
    flex: 1;
    padding: 0.75rem;
    border: 1px solid #d1d5db;
    border-radius: 6px;
    font-size: 1rem;
    font-family: inherit;
    resize: vertical;
    min-height: 3rem;
  }

  .query-input:focus {
    outline: none;
    border-color: #667eea;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
  }

  .input-actions {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .send-btn {
    padding: 0.75rem 1.5rem;
    border: none;
    border-radius: 6px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
    min-width: 100px;
  }

  .send-btn.regular {
    background: #667eea;
    color: white;
  }

  .send-btn.streaming {
    background: #10b981;
    color: white;
  }

  .send-btn.audio {
    background: #ec4899;
    color: white;
  }

  .send-btn.audio.recording {
    background: #dc2626;
    animation: pulse 1s infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.8; }
  }

  .send-btn:hover:not(:disabled) {
    transform: translateY(-1px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
  }

  .send-btn:disabled {
    background: #d1d5db;
    color: #9ca3af;
    cursor: not-allowed;
    transform: none;
    box-shadow: none;
  }

  .input-help {
    margin-top: 0.5rem;
    font-size: 0.8rem;
    color: #6b7280;
    text-align: center;
  }

  /* Responsive Design */
  @media (max-width: 768px) {
    .prajna-header {
      flex-direction: column;
      gap: 0.5rem;
      align-items: stretch;
    }

    .header-content {
      justify-content: center;
    }

    .header-status {
      justify-content: space-between;
    }

    .message.user {
      margin-left: 0.5rem;
    }

    .message.prajna {
      margin-right: 0.5rem;
    }

    .main-input {
      flex-direction: column;
      align-items: stretch;
    }

    .input-actions {
      flex-direction: row;
      justify-content: stretch;
    }

    .send-btn {
      flex: 1;
      padding: 0.5rem;
      min-width: auto;
    }
  }
</style>
