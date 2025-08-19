/**
 * WebRTC Offer Handler for Hologram Streaming
 * Manages SDP negotiation and session setup
 */

import { Router, Request, Response } from 'express';
import { verifyJWT } from '../auth/jwt';
import H265EncoderServer from './h265Server';
import * as msgpack from '@msgpack/msgpack';

interface WebRTCOfferRequest {
  jwt: string;
  offer: RTCSessionDescriptionInit;
  profile: 'low' | 'medium' | 'high';
  capabilities: string[];
}

export class WebRTCOfferHandler {
  private router: Router;
  
  constructor(private encoderServer: H265EncoderServer) {
    this.router = Router();
    this.setupRoutes();
  }

  private setupRoutes(): void {
    // WebRTC offer endpoint
    this.router.post('/webrtc/offer', this.handleOffer.bind(this));
    
    // ICE candidate endpoint
    this.router.post('/webrtc/ice', this.handleICECandidate.bind(this));
    
    // Session status
    this.router.get('/webrtc/status/:sessionId', this.getSessionStatus.bind(this));
    
    // Close session
    this.router.delete('/webrtc/session/:sessionId', this.closeSession.bind(this));
  }

  /**
   * Handle WebRTC offer from mobile client
   */
  private async handleOffer(req: Request, res: Response): Promise<void> {
    try {
      const { jwt, offer, profile, capabilities } = req.body as WebRTCOfferRequest;
      
      // Verify JWT
      const claims = await verifyJWT(jwt);
      if (!claims.capabilities?.includes('stream')) {
        return res.status(403).json({
          error: 'Stream capability not granted'
        });
      }

      // Check if user has requested quality tier access
      const allowedProfile = this.getAllowedProfile(claims, profile);
      
      // Create session with encoder server
      const sessionId = this.generateSessionId();
      
      // Forward to encoder server via internal API
      const response = await this.encoderServer.createSession({
        sessionId,
        jwt,
        offer,
        profile: allowedProfile,
        capabilities: claims.capabilities
      });

      res.json({
        sessionId,
        answer: response.answer,
        iceServers: this.getICEServers(),
        profile: allowedProfile,
        encoderConfig: {
          codec: 'h265',
          hardwareAcceleration: response.nvencAvailable
        }
      });

    } catch (error) {
      console.error('WebRTC offer error:', error);
      res.status(500).json({
        error: 'Failed to process offer'
      });
    }
  }

  /**
   * Handle ICE candidates
   */
  private async handleICECandidate(req: Request, res: Response): Promise<void> {
    try {
      const { sessionId, candidate } = req.body;
      
      // Forward to encoder server
      await this.encoderServer.addICECandidate(sessionId, candidate);
      
      res.json({ success: true });
    } catch (error) {
      res.status(400).json({
        error: 'Failed to add ICE candidate'
      });
    }
  }

  /**
   * Get session status
   */
  private async getSessionStatus(req: Request, res: Response): Promise<void> {
    const { sessionId } = req.params;
    
    const stats = this.encoderServer.getSessionStats(sessionId);
    if (!stats) {
      return res.status(404).json({
        error: 'Session not found'
      });
    }

    res.json(stats);
  }

  /**
   * Close streaming session
   */
  private async closeSession(req: Request, res: Response): Promise<void> {
    const { sessionId } = req.params;
    
    await this.encoderServer.closeSession(sessionId);
    res.json({ success: true });
  }

  /**
   * Determine allowed quality profile based on JWT claims
   */
  private getAllowedProfile(
    claims: any, 
    requested: string
  ): 'low' | 'medium' | 'high' {
    // Premium users get their requested quality
    if (claims.tier === 'premium') {
      return requested as any;
    }
    
    // Free users limited to low quality
    if (claims.tier === 'free') {
      return 'low';
    }
    
    // Default to medium
    return 'medium';
  }

  /**
   * Get ICE server configuration
   */
  private getICEServers(): RTCIceServer[] {
    return [
      { urls: 'stun:stun.l.google.com:19302' },
      { urls: 'stun:stun.tori.ai:3478' },
      {
        urls: 'turn:turn.tori.ai:3478',
        username: 'tori',
        credential: process.env.TURN_PASSWORD || 'default-turn-password'
      },
      {
        urls: 'turn:turn2.tori.ai:3478',
        username: 'tori',
        credential: process.env.TURN_PASSWORD || 'default-turn-password'
      }
    ];
  }

  /**
   * Generate unique session ID
   */
  private generateSessionId(): string {
    return `webrtc-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Get Express router
   */
  getRouter(): Router {
    return this.router;
  }
}

// WebSocket fallback handler for poor network conditions
export class WebSocketStreamHandler {
  constructor(private encoderServer: H265EncoderServer) {}

  /**
   * Handle WebSocket upgrade for streaming
   */
  handleUpgrade(ws: WebSocket, request: any): void {
    // Extract JWT from query or header
    const jwt = this.extractJWT(request);
    if (!jwt) {
      ws.close(1008, 'Missing authentication');
      return;
    }

    // Verify JWT
    verifyJWT(jwt).then(claims => {
      if (!claims.capabilities?.includes('stream')) {
        ws.close(1008, 'Stream capability not granted');
        return;
      }

      this.handleStreamingSession(ws, claims);
    }).catch(err => {
      ws.close(1008, 'Invalid JWT');
    });
  }

  /**
   * Handle streaming session over WebSocket
   */
  private handleStreamingSession(ws: WebSocket, claims: any): void {
    const sessionId = this.generateSessionId();
    let frameBuffer: Buffer[] = [];
    let streaming = false;

    ws.on('message', async (data) => {
      try {
        const message = msgpack.decode(data as Buffer);
        
        switch (message.type) {
          case 'start':
            streaming = true;
            await this.startStreaming(sessionId, message.profile);
            break;
            
          case 'stop':
            streaming = false;
            await this.stopStreaming(sessionId);
            break;
            
          case 'quality':
            await this.adjustQuality(sessionId, message.profile);
            break;
        }
      } catch (error) {
        console.error('WebSocket message error:', error);
      }
    });

    // Stream encoded frames
    this.encoderServer.on(`frame:${sessionId}`, (frame: Buffer) => {
      if (streaming && ws.readyState === WebSocket.OPEN) {
        ws.send(msgpack.encode({
          type: 'frame',
          timestamp: Date.now(),
          data: frame
        }));
      }
    });

    ws.on('close', () => {
      this.stopStreaming(sessionId);
    });
  }

  private extractJWT(request: any): string | null {
    // Check query parameter
    if (request.url) {
      const url = new URL(request.url, `http://${request.headers.host}`);
      const jwt = url.searchParams.get('jwt');
      if (jwt) return jwt;
    }

    // Check Authorization header
    const auth = request.headers.authorization;
    if (auth?.startsWith('Bearer ')) {
      return auth.substring(7);
    }

    return null;
  }

  private generateSessionId(): string {
    return `ws-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  private async startStreaming(sessionId: string, profile: string): Promise<void> {
    // Implementation depends on encoder server API
  }

  private async stopStreaming(sessionId: string): Promise<void> {
    // Implementation depends on encoder server API
  }

  private async adjustQuality(sessionId: string, profile: string): Promise<void> {
    // Implementation depends on encoder server API
  }
}

export default WebRTCOfferHandler;
