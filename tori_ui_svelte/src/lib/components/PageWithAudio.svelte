<script lang="ts">
  import { onMount, afterUpdate, tick } from 'svelte';
  import { conceptMesh, addConceptDiff } from '$lib/stores/conceptMesh';
  import ConceptDebugPanel from '$lib/components/ConceptDebugPanel.svelte';
  import { dynamicsIntegration } from '$lib/integrations/dynamicsIntegration';
  import { darkMode } from '$lib/stores/darkMode';
  import type { KoopmanUpdate, LyapunovSpike } from '$lib/cognitive/dynamics';
  
  // ? Import Soliton Memory System - FIXED: Use proper service imports
  import { 
    initializeUserMemory, 
    storeMemory, 
    fetchMemoryStats, 
    findRelatedMemories,
    vaultMemory,
    analyzeValence,
    extractPhaseTags,
    measureAmplitude,
    estimateFrequency,
    phaseChangeEvent
  } from '$lib/services/solitonMemory';
  
  // ?? NEW IMPORTS FOR INTEGRATED SERVICES
  import { toriStorage, type Message as StorageMessage } from '$lib/services/toriStorage';
  import { prosodyAnalyzer, type ProsodyMetrics, type ProsodyInsight } from '$lib/services/typingProsody';
  import { intentTracker, type Intent, type IntentContext, ReasoningIntent } from '$lib/services/intentTracking';
  
  // ?? AUDIO/VIDEO IMPORTS
  import { browser } from '$app/environment';
  
  // Destructure the stores from intentTracker
  const { currentIntent, conversationInsights, intentHistory } = intentTracker;
  
  // STEP 1-4: Import ALL systems
  let braidMemory: any = null;
  let cognitiveEngine: any = null;
  let holographicMemory: any = null;
  let ghostCollective: any = null;

  let files: FileList | null = null;
  let fileInput: HTMLInputElement;
  let uploadingFiles = false;
  let uploadProgress = 0;

  // Get user data from server via layout
  export let data: { user: { name: string; role: 'admin' | 'user' } | null };
  
  let mounted = false;
  let messageInput = '';
  let isTyping = false;
  let showDebugPanel = false;
  let conversationHistory: StorageMessage[] = [];
  
  // ?? AUDIO/VIDEO STATE
  let isRecording = false;
  let mediaRecorder: MediaRecorder | null = null;
  let audioChunks: Blob[] = [];
  let audioPlayback: HTMLAudioElement | null = null;
  let avatarWebSocket: WebSocket | null = null;
  let avatarState = {
    state: 'idle',
    mood: 'neutral',
    audio_level: 0
  };
  let showVideoCapture = false;
  let videoStream: MediaStream | null = null;
  let videoElement: HTMLVideoElement;
  let canvasElement: HTMLCanvasElement;
  let currentPlayingMessageId: string | null = null;
  let ttsEnabled = true; // Global TTS toggle
  let autoPlayTTS = true; // Auto-play responses
  
  // ?? NEW: Intent and prosody tracking
  let currentProsodyMetrics: ProsodyMetrics | null = null;
  let currentProsodyInsights: ProsodyInsight[] = [];
  let currentIntentContext: IntentContext | null = null;
  let showIntentInfo = false;
  
  // System stats - FIXED: Proper Soliton integration
  let solitonStats: any = null;
  let braidStats: any = null;
  let holographicStats: any = null;
  let ghostStats: any = null;
  let memoryInitialized = false;
  let currentUserId = '';
  
  // FIXED: Phase monitoring for Ghost AI
  let currentPhase = 'idle';
  let phaseAmplitude = 0;
  let phaseFrequency = 0;
  let systemCoherence = 1.0;
  
  // Enhanced holographic memory tracking
  let userConceptNode: any = null;
  let recentCrossings: any[] = [];
  let currentCoherence = 0.5;
  let currentContradiction = 0.0;
  
  // Dynamics analysis tracking
  let latestKoopmanUpdate: KoopmanUpdate | null = null;
  let recentLyapunovSpikes: LyapunovSpike[] = [];
  let showDynamicsInfo = false;
  
  // ? AUTO-SCROLL FUNCTIONALITY
  let scrollContainer: HTMLElement;
  let isUserScrolledUp = false;
  let showScrollToBottom = false;
  let lastMessageCount = 0;
  let shouldAutoScroll = true;
  
  // </script>