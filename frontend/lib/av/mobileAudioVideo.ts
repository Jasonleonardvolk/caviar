/* frontend/lib/av/mobileAudioVideo.ts
 * iOS Safari audio/video handling with gesture requirements
 * Handles autoplay policies, WebRTC quirks, and mobile-specific issues
 */

export interface AVConfig {
  enableAudio?: boolean;
  enableVideo?: boolean;
  enableWebRTC?: boolean;
  audioLatencyHint?: 'interactive' | 'balanced' | 'playback';
  videoConstraints?: MediaTrackConstraints;
}

export interface AVState {
  audioUnlocked: boolean;
  videoUnlocked: boolean;
  audioContext: AudioContext | null;
  mediaStream: MediaStream | null;
  isIOS: boolean;
  isSafari: boolean;
  hasUserGesture: boolean;
}

class MobileAVManager {
  private state: AVState;
  private config: AVConfig;
  private audioContext: AudioContext | null = null;
  private gainNode: GainNode | null = null;
  private mediaStream: MediaStream | null = null;
  private silentAudioElement: HTMLAudioElement | null = null;
  private unlockPromise: Promise<void> | null = null;
  private gestureListeners: Set<() => void> = new Set();
  
  constructor(config: AVConfig = {}) {
    this.config = {
      enableAudio: true,
      enableVideo: false,
      enableWebRTC: false,
      audioLatencyHint: 'interactive',
      videoConstraints: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: 'user'
      },
      ...config
    };
    
    // Detect platform
    const ua = navigator.userAgent;
    const isIOS = /iPad|iPhone|iPod/.test(ua) || 
      (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
    const isSafari = /^((?!chrome|android).)*safari/i.test(ua);
    
    this.state = {
      audioUnlocked: false,
      videoUnlocked: false,
      audioContext: null,
      mediaStream: null,
      isIOS,
      isSafari,
      hasUserGesture: false
    };
    
    // Set up gesture detection
    this.setupGestureDetection();
    
    // Log platform info
    console.log('[MobileAV] Initialized:', {
      isIOS,
      isSafari,
      config: this.config
    });
  }
  
  /**
   * Set up gesture detection for iOS Safari
   */
  private setupGestureDetection() {
    const gestureEvents = ['click', 'touchstart', 'touchend', 'mousedown'];
    
    const handleGesture = () => {
      if (!this.state.hasUserGesture) {
        this.state.hasUserGesture = true;
        console.log('[MobileAV] User gesture detected');
        
        // Trigger any pending gesture callbacks
        this.gestureListeners.forEach(callback => callback());
        this.gestureListeners.clear();
      }
    };
    
    // Add listeners with passive flag for better mobile performance
    gestureEvents.forEach(event => {
      document.addEventListener(event, handleGesture, { 
        once: true, 
        passive: true 
      });
    });
  }
  
  /**
   * Wait for user gesture if needed
   */
  private waitForGesture(): Promise<void> {
    if (this.state.hasUserGesture) {
      return Promise.resolve();
    }
    
    return new Promise(resolve => {
      this.gestureListeners.add(resolve);
    });
  }
  
  /**
   * Initialize and unlock audio context
   */
  async initAudio(): Promise<AudioContext> {
    if (this.audioContext && this.state.audioUnlocked) {
      return this.audioContext;
    }
    
    // Create audio context with appropriate latency hint
    const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
    if (!AudioContextClass) {
      throw new Error('Web Audio API not supported');
    }
    
    this.audioContext = new AudioContextClass({
      latencyHint: this.config.audioLatencyHint,
      sampleRate: 48000 // Standard for most devices
    });
    
    // Create gain node for volume control
    this.gainNode = this.audioContext.createGain();
    this.gainNode.connect(this.audioContext.destination);
    
    // iOS requires user gesture to start audio
    if (this.state.isIOS || this.state.isSafari) {
      await this.unlockAudioContext();
    }
    
    this.state.audioContext = this.audioContext;
    this.state.audioUnlocked = true;
    
    console.log('[MobileAV] Audio initialized:', {
      state: this.audioContext.state,
      sampleRate: this.audioContext.sampleRate,
      outputLatency: (this.audioContext as any).outputLatency || 'unknown'
    });
    
    return this.audioContext;
  }
  
  /**
   * Unlock audio context on iOS/Safari
   */
  private async unlockAudioContext(): Promise<void> {
    if (!this.audioContext) return;
    
    // If already unlocking, wait for that
    if (this.unlockPromise) {
      return this.unlockPromise;
    }
    
    this.unlockPromise = (async () => {
      // Wait for user gesture if needed
      if (!this.state.hasUserGesture) {
        console.log('[MobileAV] Waiting for user gesture to unlock audio...');
        await this.waitForGesture();
      }
      
      // Resume context if suspended
      if (this.audioContext!.state === 'suspended') {
        await this.audioContext!.resume();
      }
      
      // iOS Safari hack: Play silent audio to unlock
      if (this.state.isIOS) {
        await this.playSilentAudio();
      }
      
      // Create and play a silent buffer to fully unlock
      const buffer = this.audioContext!.createBuffer(1, 1, 22050);
      const source = this.audioContext!.createBufferSource();
      source.buffer = buffer;
      source.connect(this.audioContext!.destination);
      source.start(0);
      
      // Wait a bit for the unlock to take effect
      await new Promise(resolve => setTimeout(resolve, 100));
      
      console.log('[MobileAV] Audio context unlocked');
    })();
    
    return this.unlockPromise;
  }
  
  /**
   * Play silent audio element to unlock iOS audio
   */
  private async playSilentAudio(): Promise<void> {
    if (!this.silentAudioElement) {
      // Create silent audio element
      this.silentAudioElement = document.createElement('audio');
      this.silentAudioElement.setAttribute('playsinline', 'true');
      this.silentAudioElement.setAttribute('webkit-playsinline', 'true');
      this.silentAudioElement.muted = true;
      
      // Use data URL for silent audio (1 second of silence)
      this.silentAudioElement.src = 'data:audio/mp3;base64,SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU4LjI5LjEwMAAAAAAAAAAAAAAA//tQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAASW5mbwAAAA8AAAACAAABhgC7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7v////////////////////////////////////////////////////////////AAAAAExhdmY1OC4yOS4xMDABCQAAAAAAAAAAACQAAAAAAAAAAYbvhqbHAAAAAAAAAAAAAAAAAAAA//tAxAAAAAAGkAAAAAAAAA0gAAAAATEFNRTMuMTAwBCQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA//tAxEuAAAGkAAAAAAAAANIAAAAAEzBTUUzLjEwMAQoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA';
      
      document.body.appendChild(this.silentAudioElement);
    }
    
    try {
      // Unmute and play
      this.silentAudioElement.muted = false;
      this.silentAudioElement.volume = 0.01; // Very quiet
      await this.silentAudioElement.play();
      
      // Mute again after playing
      this.silentAudioElement.muted = true;
    } catch (err) {
      console.warn('[MobileAV] Silent audio play failed:', err);
    }
  }
  
  /**
   * Initialize camera/video stream
   */
  async initVideo(constraints?: MediaTrackConstraints): Promise<MediaStream> {
    if (this.mediaStream && this.state.videoUnlocked) {
      return this.mediaStream;
    }
    
    // iOS requires user gesture for camera access
    if ((this.state.isIOS || this.state.isSafari) && !this.state.hasUserGesture) {
      console.log('[MobileAV] Waiting for user gesture to access camera...');
      await this.waitForGesture();
    }
    
    // Request camera access
    try {
      const videoConstraints = constraints || this.config.videoConstraints || true;
      
      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        video: videoConstraints,
        audio: false // We handle audio separately
      });
      
      this.state.mediaStream = this.mediaStream;
      this.state.videoUnlocked = true;
      
      // Get video track settings
      const videoTrack = this.mediaStream.getVideoTracks()[0];
      const settings = videoTrack.getSettings();
      
      console.log('[MobileAV] Video initialized:', {
        width: settings.width,
        height: settings.height,
        frameRate: settings.frameRate,
        facingMode: settings.facingMode
      });
      
      return this.mediaStream;
    } catch (error) {
      console.error('[MobileAV] Failed to access camera:', error);
      throw error;
    }
  }
  
  /**
   * Initialize WebRTC with iOS-specific fixes
   */
  async initWebRTC(config?: RTCConfiguration): Promise<RTCPeerConnection> {
    // Ensure we have user gesture for iOS
    if ((this.state.isIOS || this.state.isSafari) && !this.state.hasUserGesture) {
      console.log('[MobileAV] Waiting for user gesture for WebRTC...');
      await this.waitForGesture();
    }
    
    // iOS Safari WebRTC configuration
    const iosConfig: RTCConfiguration = {
      iceServers: [
        { urls: 'stun:stun.l.google.com:19302' },
        { urls: 'stun:stun1.l.google.com:19302' }
      ],
      // sdpSemantics: 'unified-plan', // Note: This is the default in modern browsers
      ...config
    };
    
    // Create peer connection
    const pc = new RTCPeerConnection(iosConfig);
    
    // iOS-specific: Add transceiver for audio/video
    if (this.state.isIOS) {
      // Add transceivers to avoid issues with addTrack
      pc.addTransceiver('audio', { direction: 'sendrecv' });
      pc.addTransceiver('video', { direction: 'sendrecv' });
    }
    
    console.log('[MobileAV] WebRTC initialized with config:', iosConfig);
    
    return pc;
  }
  
  /**
   * Set volume (0.0 to 1.0)
   */
  setVolume(volume: number) {
    if (this.gainNode) {
      this.gainNode.gain.setValueAtTime(
        Math.max(0, Math.min(1, volume)),
        this.audioContext!.currentTime
      );
    }
  }
  
  /**
   * Play audio buffer
   */
  async playBuffer(
    buffer: AudioBuffer,
    options?: {
      volume?: number;
      loop?: boolean;
      onEnded?: () => void;
    }
  ): Promise<AudioBufferSourceNode> {
    if (!this.audioContext || !this.gainNode) {
      await this.initAudio();
    }
    
    const source = this.audioContext!.createBufferSource();
    source.buffer = buffer;
    source.loop = options?.loop || false;
    
    // Create gain for this specific sound
    const soundGain = this.audioContext!.createGain();
    soundGain.gain.value = options?.volume ?? 1.0;
    
    // Connect: source -> soundGain -> mainGain -> destination
    source.connect(soundGain);
    soundGain.connect(this.gainNode!);
    
    if (options?.onEnded) {
      source.onended = options.onEnded;
    }
    
    source.start(0);
    
    return source;
  }
  
  /**
   * Create oscillator for testing/tones
   */
  createOscillator(frequency: number = 440): OscillatorNode {
    if (!this.audioContext || !this.gainNode) {
      throw new Error('Audio context not initialized');
    }
    
    const oscillator = this.audioContext.createOscillator();
    oscillator.frequency.value = frequency;
    oscillator.connect(this.gainNode);
    
    return oscillator;
  }
  
  /**
   * Get current state
   */
  getState(): AVState {
    return { ...this.state };
  }
  
  /**
   * Clean up resources
   */
  async destroy() {
    // Stop all media tracks
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(track => track.stop());
      this.mediaStream = null;
    }
    
    // Close audio context
    if (this.audioContext && this.audioContext.state !== 'closed') {
      await this.audioContext.close();
      this.audioContext = null;
    }
    
    // Remove silent audio element
    if (this.silentAudioElement) {
      this.silentAudioElement.remove();
      this.silentAudioElement = null;
    }
    
    // Clear listeners
    this.gestureListeners.clear();
    
    console.log('[MobileAV] Destroyed');
  }
}

// Singleton instance
let instance: MobileAVManager | null = null;

/**
 * Get or create the mobile AV manager instance
 */
export function getMobileAVManager(config?: AVConfig): MobileAVManager {
  if (!instance) {
    instance = new MobileAVManager(config);
  }
  return instance;
}

/**
 * Helper to check if we need mobile AV workarounds
 */
export function needsMobileAVWorkarounds(): boolean {
  const ua = navigator.userAgent;
  const isIOS = /iPad|iPhone|iPod/.test(ua) || 
    (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
  const isSafari = /^((?!chrome|android).)*safari/i.test(ua);
  const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(ua);
  
  return isIOS || isSafari || isMobile;
}

export default MobileAVManager;
