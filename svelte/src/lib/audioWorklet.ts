// AudioWorklet with fallback support for older browsers

export interface AudioProcessorOptions {
  sampleRate: number;
  bufferSize: number;
  channelCount: number;
}

// Check if AudioWorklet is supported
export const isAudioWorkletSupported = (): boolean => {
  return typeof AudioWorkletNode !== 'undefined' && 
         typeof AudioContext !== 'undefined' &&
         'audioWorklet' in AudioContext.prototype;
};

// AudioWorklet processor code (as a string to be loaded)
export const audioWorkletProcessorCode = `
class AudioStreamProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.bufferSize = options.processorOptions?.bufferSize || 4096;
    this.buffer = new Float32Array(this.bufferSize);
    this.bufferIndex = 0;
    this.isRecording = true;
    
    // Handle messages from main thread
    this.port.onmessage = (event) => {
      if (event.data.type === 'stop') {
        this.isRecording = false;
      } else if (event.data.type === 'start') {
        this.isRecording = true;
      }
    };
  }
  
  process(inputs, outputs, parameters) {
    if (!this.isRecording) return true;
    
    const input = inputs[0];
    if (!input || !input[0]) return true;
    
    const inputChannel = input[0];
    
    // Copy samples to buffer
    for (let i = 0; i < inputChannel.length; i++) {
      this.buffer[this.bufferIndex++] = inputChannel[i];
      
      // When buffer is full, send it to main thread
      if (this.bufferIndex >= this.bufferSize) {
        // Convert to 16-bit PCM
        const pcmData = new Int16Array(this.bufferSize);
        for (let j = 0; j < this.bufferSize; j++) {
          const sample = Math.max(-1, Math.min(1, this.buffer[j]));
          pcmData[j] = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
        }
        
        // Send PCM data to main thread
        this.port.postMessage({
          type: 'audio',
          data: pcmData.buffer
        }, [pcmData.buffer]);
        
        // Reset buffer
        this.bufferIndex = 0;
      }
    }
    
    return true;
  }
}

registerProcessor('audio-stream-processor', AudioStreamProcessor);
`;

// Modern AudioWorklet implementation
export class AudioWorkletProcessor {
  private audioContext: AudioContext;
  private workletNode: AudioWorkletNode | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private onDataCallback: ((data: ArrayBuffer) => void) | null = null;
  
  constructor(audioContext: AudioContext, options: AudioProcessorOptions) {
    this.audioContext = audioContext;
  }
  
  async initialize(): Promise<void> {
    try {
      // Create a blob URL for the worklet code
      const blob = new Blob([audioWorkletProcessorCode], { type: 'application/javascript' });
      const workletUrl = URL.createObjectURL(blob);
      
      // Load the worklet module
      await this.audioContext.audioWorklet.addModule(workletUrl);
      
      // Clean up the blob URL
      URL.revokeObjectURL(workletUrl);
      
      // Create the worklet node
      this.workletNode = new AudioWorkletNode(this.audioContext, 'audio-stream-processor', {
        processorOptions: {
          bufferSize: 4096
        }
      });
      
      // Handle messages from the worklet
      this.workletNode.port.onmessage = (event) => {
        if (event.data.type === 'audio' && this.onDataCallback) {
          this.onDataCallback(event.data.data);
        }
      };
      
    } catch (error) {
      console.error('Failed to initialize AudioWorklet:', error);
      throw error;
    }
  }
  
  connect(source: MediaStreamAudioSourceNode): void {
    if (!this.workletNode) {
      throw new Error('AudioWorklet not initialized');
    }
    
    this.source = source;
    this.source.connect(this.workletNode);
    this.workletNode.connect(this.audioContext.destination);
  }
  
  disconnect(): void {
    if (this.source && this.workletNode) {
      this.source.disconnect(this.workletNode);
      this.workletNode.disconnect();
    }
  }
  
  start(): void {
    this.workletNode?.port.postMessage({ type: 'start' });
  }
  
  stop(): void {
    this.workletNode?.port.postMessage({ type: 'stop' });
  }
  
  onData(callback: (data: ArrayBuffer) => void): void {
    this.onDataCallback = callback;
  }
}

// Legacy ScriptProcessorNode fallback
export class ScriptProcessorFallback {
  private audioContext: AudioContext;
  private processorNode: ScriptProcessorNode | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private onDataCallback: ((data: ArrayBuffer) => void) | null = null;
  private bufferSize: number;
  private isRecording: boolean = false;
  
  constructor(audioContext: AudioContext, options: AudioProcessorOptions) {
    this.audioContext = audioContext;
    this.bufferSize = options.bufferSize || 4096;
  }
  
  async initialize(): Promise<void> {
    // Create ScriptProcessorNode
    this.processorNode = this.audioContext.createScriptProcessor(this.bufferSize, 1, 1);
    
    this.processorNode.onaudioprocess = (event) => {
      if (!this.isRecording || !this.onDataCallback) {
        // Pass through audio
        const inputData = event.inputBuffer.getChannelData(0);
        event.outputBuffer.copyToChannel(inputData, 0);
        return;
      }
      
      const inputData = event.inputBuffer.getChannelData(0);
      
      // Convert to 16-bit PCM
      const pcmData = new Int16Array(inputData.length);
      for (let i = 0; i < inputData.length; i++) {
        const sample = Math.max(-1, Math.min(1, inputData[i]));
        pcmData[i] = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
      }
      
      // Send PCM data
      this.onDataCallback(pcmData.buffer);
      
      // Pass through audio
      event.outputBuffer.copyToChannel(inputData, 0);
    };
  }
  
  connect(source: MediaStreamAudioSourceNode): void {
    if (!this.processorNode) {
      throw new Error('ScriptProcessor not initialized');
    }
    
    this.source = source;
    this.source.connect(this.processorNode);
    this.processorNode.connect(this.audioContext.destination);
  }
  
  disconnect(): void {
    if (this.source && this.processorNode) {
      this.source.disconnect(this.processorNode);
      this.processorNode.disconnect();
    }
  }
  
  start(): void {
    this.isRecording = true;
  }
  
  stop(): void {
    this.isRecording = false;
  }
  
  onData(callback: (data: ArrayBuffer) => void): void {
    this.onDataCallback = callback;
  }
}

// Factory function to create appropriate processor
export async function createAudioProcessor(
  audioContext: AudioContext,
  options: AudioProcessorOptions = {
    sampleRate: 16000,
    bufferSize: 4096,
    channelCount: 1
  }
): Promise<AudioWorkletProcessor | ScriptProcessorFallback> {
  
  if (isAudioWorkletSupported()) {
    console.log('Using AudioWorklet for audio processing');
    const processor = new AudioWorkletProcessor(audioContext, options);
    await processor.initialize();
    return processor;
  } else {
    console.log('Falling back to ScriptProcessorNode');
    const processor = new ScriptProcessorFallback(audioContext, options);
    await processor.initialize();
    return processor;
  }
}

// Utility function to create a resampler if needed
export function createResampler(fromSampleRate: number, toSampleRate: number) {
  const ratio = fromSampleRate / toSampleRate;
  
  return {
    resample(inputBuffer: Float32Array): Float32Array {
      if (ratio === 1) {
        return inputBuffer;
      }
      
      const outputLength = Math.floor(inputBuffer.length / ratio);
      const outputBuffer = new Float32Array(outputLength);
      
      for (let i = 0; i < outputLength; i++) {
        const inputIndex = i * ratio;
        const inputIndexInt = Math.floor(inputIndex);
        const fraction = inputIndex - inputIndexInt;
        
        if (inputIndexInt + 1 < inputBuffer.length) {
          // Linear interpolation
          outputBuffer[i] = inputBuffer[inputIndexInt] * (1 - fraction) + 
                           inputBuffer[inputIndexInt + 1] * fraction;
        } else {
          outputBuffer[i] = inputBuffer[inputIndexInt];
        }
      }
      
      return outputBuffer;
    }
  };
}

// Audio buffer utilities
export class AudioBufferQueue {
  private queue: Float32Array[] = [];
  private totalSamples: number = 0;
  private maxQueueSize: number;
  
  constructor(maxQueueSize: number = 10) {
    this.maxQueueSize = maxQueueSize;
  }
  
  push(buffer: Float32Array): void {
    this.queue.push(buffer);
    this.totalSamples += buffer.length;
    
    // Limit queue size
    while (this.queue.length > this.maxQueueSize) {
      const removed = this.queue.shift();
      if (removed) {
        this.totalSamples -= removed.length;
      }
    }
  }
  
  getBuffer(samples: number): Float32Array | null {
    if (this.totalSamples < samples) {
      return null;
    }
    
    const output = new Float32Array(samples);
    let outputIndex = 0;
    
    while (outputIndex < samples && this.queue.length > 0) {
      const buffer = this.queue[0];
      const remaining = samples - outputIndex;
      const toCopy = Math.min(buffer.length, remaining);
      
      output.set(buffer.subarray(0, toCopy), outputIndex);
      outputIndex += toCopy;
      
      if (toCopy === buffer.length) {
        this.queue.shift();
      } else {
        this.queue[0] = buffer.subarray(toCopy);
      }
    }
    
    this.totalSamples -= outputIndex;
    return outputIndex === samples ? output : null;
  }
  
  clear(): void {
    this.queue = [];
    this.totalSamples = 0;
  }
  
  get size(): number {
    return this.totalSamples;
  }
}

// WebAudio utilities
export async function getUserMedia(constraints: MediaStreamConstraints = {
  audio: {
    echoCancellation: true,
    noiseSuppression: true,
    autoGainControl: true,
    sampleRate: 16000
  }
}): Promise<MediaStream> {
  try {
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    return stream;
  } catch (error) {
    console.error('Failed to get user media:', error);
    throw error;
  }
}

// Export a complete audio processing setup
export class AudioStreamingProcessor {
  private audioContext: AudioContext | null = null;
  private processor: AudioWorkletProcessor | ScriptProcessorFallback | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private stream: MediaStream | null = null;
  private bufferQueue: AudioBufferQueue;
  private isProcessing: boolean = false;
  
  constructor() {
    this.bufferQueue = new AudioBufferQueue();
  }
  
  async initialize(options: AudioProcessorOptions = {
    sampleRate: 16000,
    bufferSize: 4096,
    channelCount: 1
  }): Promise<void> {
    // Create audio context
    this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)({
      sampleRate: options.sampleRate
    });
    
    // Create processor (with fallback)
    this.processor = await createAudioProcessor(this.audioContext, options);
  }
  
  async start(onData: (data: ArrayBuffer) => void): Promise<void> {
    if (!this.audioContext || !this.processor) {
      throw new Error('Not initialized');
    }
    
    // Get user media
    this.stream = await getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        sampleRate: this.audioContext.sampleRate
      }
    });
    
    // Create source
    this.source = this.audioContext.createMediaStreamSource(this.stream);
    
    // Set data callback
    this.processor.onData(onData);
    
    // Connect and start
    this.processor.connect(this.source);
    this.processor.start();
    this.isProcessing = true;
  }
  
  stop(): void {
    if (this.processor) {
      this.processor.stop();
      this.processor.disconnect();
    }
    
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
    }
    
    this.isProcessing = false;
  }
  
  async close(): Promise<void> {
    this.stop();
    
    if (this.audioContext) {
      await this.audioContext.close();
      this.audioContext = null;
    }
    
    this.processor = null;
    this.source = null;
    this.stream = null;
  }
  
  get isActive(): boolean {
    return this.isProcessing;
  }
  
  get context(): AudioContext | null {
    return this.audioContext;
  }
}