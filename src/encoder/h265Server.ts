/**
 * H.265/HEVC Encoder Server for Desktop → Mobile Streaming
 * Leverages NVENC on RTX 4070 for hardware acceleration
 */

import { EventEmitter } from 'events';
import { spawn, ChildProcess } from 'child_process';
import { WebSocket, WebSocketServer } from 'ws';
import SimplePeer from 'simple-peer';
import { createHash } from 'crypto';
import { promisify } from 'util';
import * as msgpack from '@msgpack/msgpack';

interface EncoderConfig {
  codec: 'h265' | 'vp9' | 'av1';
  bitrate: number;
  width: number;
  height: number;
  fps: number;
  preset: 'p1' | 'p2' | 'p3' | 'p4' | 'p5' | 'p6' | 'p7'; // NVENC presets
  profile?: 'main' | 'main10' | 'rext';
}

interface StreamSession {
  id: string;
  peer?: SimplePeer.Instance;
  encoder?: ChildProcess;
  config: EncoderConfig;
  jwt: string;
  capabilities: string[];
  lastActivity: number;
}

export class H265EncoderServer extends EventEmitter {
  private sessions: Map<string, StreamSession> = new Map();
  private wss?: WebSocketServer;
  private nvencAvailable: boolean = false;
  
  constructor(
    private port: number = 7691,
    private maxClients: number = 5,
    private bandwidthLimitMbps: number = 50
  ) {
    super();
    this.checkNvencSupport();
  }

  /**
   * Check if NVENC is available on the system
   */
  private async checkNvencSupport(): Promise<void> {
    try {
      const { exec } = await import('child_process');
      const execAsync = promisify(exec);
      
      // Check for NVENC support
      const result = await execAsync('ffmpeg -hide_banner -encoders | grep nvenc');
      this.nvencAvailable = result.stdout.includes('h265_nvenc');
      
      console.log(`NVENC H.265 support: ${this.nvencAvailable ? 'Available' : 'Not available'}`);
    } catch (error) {
      console.warn('NVENC not available, falling back to software encoding');
      this.nvencAvailable = false;
    }
  }

  /**
   * Start the encoder server
   */
  async start(): Promise<void> {
    // WebSocket server for mobile connections
    this.wss = new WebSocketServer({ 
      port: this.port,
      path: '/mobile/stream'
    });

    this.wss.on('connection', (ws, req) => {
      this.handleMobileConnection(ws, req);
    });

    console.log(`H.265 Encoder Server listening on port ${this.port}`);
  }

  /**
   * Handle incoming mobile connection
   */
  private handleMobileConnection(ws: WebSocket, req: any): void {
    const sessionId = createHash('sha256')
      .update(req.headers['sec-websocket-key'])
      .digest('hex')
      .substring(0, 16);

    // Check max clients
    if (this.sessions.size >= this.maxClients) {
      ws.send(JSON.stringify({ 
        type: 'error', 
        message: 'Server at capacity' 
      }));
      ws.close();
      return;
    }

    console.log(`New mobile connection: ${sessionId}`);

    ws.on('message', async (data) => {
      try {
        const message = msgpack.decode(data as Buffer);
        await this.handleMessage(sessionId, message, ws);
      } catch (error) {
        console.error('Message decode error:', error);
      }
    });

    ws.on('close', () => {
      this.cleanupSession(sessionId);
    });
  }

  /**
   * Handle messages from mobile client
   */
  private async handleMessage(
    sessionId: string, 
    message: any, 
    ws: WebSocket
  ): Promise<void> {
    switch (message.type) {
      case 'auth':
        await this.handleAuth(sessionId, message, ws);
        break;
        
      case 'config':
        await this.handleConfig(sessionId, message);
        break;
        
      case 'offer':
        await this.handleWebRTCOffer(sessionId, message, ws);
        break;
        
      case 'ice':
        await this.handleICECandidate(sessionId, message);
        break;
        
      case 'quality':
        await this.adjustQuality(sessionId, message.preset);
        break;
    }
  }

  /**
   * Authenticate mobile client with JWT
   */
  private async handleAuth(
    sessionId: string, 
    message: any, 
    ws: WebSocket
  ): Promise<void> {
    const { jwt } = message;
    
    // TODO: Verify JWT with TORI auth service
    // For now, extract claims directly (INSECURE - fix in production)
    try {
      const payload = JSON.parse(
        Buffer.from(jwt.split('.')[1], 'base64').toString()
      );
      
      if (!payload.capabilities?.includes('stream')) {
        throw new Error('Stream capability not granted');
      }

      const session: StreamSession = {
        id: sessionId,
        jwt,
        capabilities: payload.capabilities,
        config: this.getDefaultConfig(),
        lastActivity: Date.now()
      };

      this.sessions.set(sessionId, session);
      
      ws.send(msgpack.encode({
        type: 'auth_success',
        sessionId,
        capabilities: payload.capabilities
      }));

    } catch (error) {
      ws.send(msgpack.encode({
        type: 'auth_error',
        message: 'Invalid JWT'
      }));
      ws.close();
    }
  }

  /**
   * Configure encoder settings
   */
  private async handleConfig(sessionId: string, message: any): Promise<void> {
    const session = this.sessions.get(sessionId);
    if (!session) return;

    const { profile } = message;
    
    // Map profile to encoder config
    const configs: Record<string, Partial<EncoderConfig>> = {
      'low': {
        bitrate: 5_000_000,
        width: 640,
        height: 480,
        fps: 30,
        preset: 'p7' // Fastest
      },
      'medium': {
        bitrate: 15_000_000,
        width: 1280,
        height: 720,
        fps: 45,
        preset: 'p4'
      },
      'high': {
        bitrate: 50_000_000,
        width: 1920,
        height: 1080,
        fps: 60,
        preset: 'p1' // Best quality
      }
    };

    session.config = { ...session.config, ...configs[profile] };
    this.sessions.set(sessionId, session);
  }

  /**
   * Handle WebRTC offer from mobile
   */
  private async handleWebRTCOffer(
    sessionId: string, 
    message: any, 
    ws: WebSocket
  ): Promise<void> {
    const session = this.sessions.get(sessionId);
    if (!session) return;

    // Create WebRTC peer
    const peer = new SimplePeer({
      initiator: false,
      trickle: true,
      config: {
        iceServers: [
          { urls: 'stun:stun.l.google.com:19302' },
          // Add TURN servers from config
          {
            urls: 'turn:turn.tori.ai:3478',
            username: 'tori',
            credential: process.env.TURN_PASSWORD
          }
        ]
      }
    });

    session.peer = peer;

    // Handle peer events
    peer.on('signal', (data) => {
      ws.send(msgpack.encode({
        type: 'answer',
        sdp: data
      }));
    });

    peer.on('connect', () => {
      console.log(`WebRTC connected for session ${sessionId}`);
      this.startEncoding(session);
    });

    peer.on('data', (data) => {
      // Handle data channel messages
      const msg = msgpack.decode(data as Buffer);
      if (msg.type === 'quality') {
        this.adjustQuality(sessionId, msg.preset);
      }
    });

    peer.on('error', (err) => {
      console.error(`WebRTC error for ${sessionId}:`, err);
    });

    // Process the offer
    peer.signal(message.offer);
    this.sessions.set(sessionId, session);
  }

  /**
   * Handle ICE candidates
   */
  private async handleICECandidate(
    sessionId: string, 
    message: any
  ): Promise<void> {
    const session = this.sessions.get(sessionId);
    if (!session?.peer) return;

    session.peer.signal({ candidate: message.candidate });
  }

  /**
   * Start encoding and streaming
   */
  private startEncoding(session: StreamSession): void {
    const { config, peer } = session;
    if (!peer) return;

    // Build FFmpeg command
    const args = this.buildFFmpegArgs(config);
    
    console.log(`Starting encoder for ${session.id}:`, args.join(' '));
    
    // Spawn FFmpeg process
    const encoder = spawn('ffmpeg', args);
    session.encoder = encoder;

    // Pipe encoded data to WebRTC
    encoder.stdout.on('data', (chunk) => {
      if (peer.connected) {
        // Send as binary data with timestamp
        const packet = msgpack.encode({
          type: 'video',
          timestamp: Date.now(),
          data: chunk
        });
        peer.send(packet);
      }
    });

    encoder.stderr.on('data', (data) => {
      // Log encoder output for debugging
      if (process.env.DEBUG_ENCODER) {
        console.log(`Encoder [${session.id}]:`, data.toString());
      }
    });

    encoder.on('exit', (code) => {
      console.log(`Encoder exited for ${session.id} with code ${code}`);
      this.cleanupSession(session.id);
    });

    // Monitor bandwidth
    this.monitorBandwidth(session);
  }

  /**
   * Build FFmpeg arguments for encoding
   */
  private buildFFmpegArgs(config: EncoderConfig): string[] {
    const args = [
      '-f', 'rawvideo',
      '-pix_fmt', 'rgba',
      '-s', `${config.width}x${config.height}`,
      '-r', config.fps.toString(),
      '-i', 'pipe:0', // Input from stdin
    ];

    // Encoder selection
    if (this.nvencAvailable && config.codec === 'h265') {
      args.push(
        '-c:v', 'hevc_nvenc',
        '-preset', config.preset,
        '-profile:v', config.profile || 'main',
        '-tier', 'high',
        '-level', '5.1',
        '-rc', 'vbr',
        '-cq', '23', // Quality level
        '-b:v', config.bitrate.toString(),
        '-maxrate', (config.bitrate * 1.5).toString(),
        '-bufsize', (config.bitrate * 2).toString()
      );
    } else if (config.codec === 'vp9') {
      args.push(
        '-c:v', 'libvpx-vp9',
        '-b:v', config.bitrate.toString(),
        '-quality', 'realtime',
        '-speed', '6',
        '-tile-columns', '2',
        '-threads', '4'
      );
    } else {
      // Fallback to software H.265
      args.push(
        '-c:v', 'libx265',
        '-preset', 'ultrafast',
        '-b:v', config.bitrate.toString(),
        '-x265-params', 'keyint=60:min-keyint=60'
      );
    }

    // Common output settings
    args.push(
      '-pix_fmt', 'yuv420p',
      '-f', 'mpegts', // Transport stream format
      'pipe:1' // Output to stdout
    );

    return args;
  }

  /**
   * Monitor and enforce bandwidth limits
   */
  private monitorBandwidth(session: StreamSession): void {
    let bytesSent = 0;
    let lastCheck = Date.now();

    const interval = setInterval(() => {
      const now = Date.now();
      const elapsed = (now - lastCheck) / 1000;
      const mbps = (bytesSent * 8) / (elapsed * 1_000_000);

      if (mbps > this.bandwidthLimitMbps) {
        // Reduce quality if exceeding limit
        console.warn(`Session ${session.id} exceeding bandwidth limit: ${mbps.toFixed(2)} Mbps`);
        this.adjustQuality(session.id, 'low');
      }

      bytesSent = 0;
      lastCheck = now;

      // Check if session is still active
      if (!this.sessions.has(session.id)) {
        clearInterval(interval);
      }
    }, 5000); // Check every 5 seconds
  }

  /**
   * Adjust encoding quality on the fly
   */
  private async adjustQuality(sessionId: string, preset: string): Promise<void> {
    const session = this.sessions.get(sessionId);
    if (!session?.encoder) return;

    // Signal encoder to change bitrate
    // This is a simplified version - real implementation would use
    // FFmpeg's zmq filter or restart encoder with new settings
    console.log(`Adjusting quality for ${sessionId} to ${preset}`);
    
    // TODO: Implement dynamic bitrate adjustment
  }

  /**
   * Clean up session resources
   */
  private cleanupSession(sessionId: string): void {
    const session = this.sessions.get(sessionId);
    if (!session) return;

    // Stop encoder
    if (session.encoder) {
      session.encoder.kill('SIGTERM');
    }

    // Close WebRTC peer
    if (session.peer) {
      session.peer.destroy();
    }

    this.sessions.delete(sessionId);
    console.log(`Cleaned up session ${sessionId}`);
  }

  /**
   * Get default encoder configuration
   */
  private getDefaultConfig(): EncoderConfig {
    return {
      codec: 'h265',
      bitrate: 15_000_000, // 15 Mbps
      width: 1280,
      height: 720,
      fps: 45,
      preset: 'p4'
    };
  }

  /**
   * Stop the encoder server
   */
  async stop(): Promise<void> {
    // Clean up all sessions
    for (const sessionId of this.sessions.keys()) {
      this.cleanupSession(sessionId);
    }

    // Close WebSocket server
    if (this.wss) {
      this.wss.close();
    }

    this.emit('stopped');
  }

  /**
   * Get server statistics
   */
  getStats(): any {
    const stats = {
      activeSessions: this.sessions.size,
      nvencAvailable: this.nvencAvailable,
      sessions: Array.from(this.sessions.values()).map(s => ({
        id: s.id,
        config: s.config,
        capabilities: s.capabilities,
        uptime: Date.now() - s.lastActivity
      }))
    };

    return stats;
  }
}

// Export for use in main application
export default H265EncoderServer;
