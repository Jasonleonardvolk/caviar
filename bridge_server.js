import express from 'express';
import cors from 'cors';
import { WebSocketServer } from 'ws';
import { createServer } from 'http';
import { 
  CognitiveEngineBridge, 
  MemoryVaultBridge, 
  EigenvalueMonitorBridge,
  createPythonBridge 
} from './src/bridges/PythonBridge.js';

interface BridgeService {
  name: string;
  bridge: any;
  isConnected: boolean;
  lastError?: string;
}

class TORIBridgeServer {
  private app = express();
  private server = createServer(this.app);
  private wss = new WebSocketServer({ server: this.server });
  private port = process.env.PORT || 8080;
  
  private services: Map<string, BridgeService> = new Map();
  
  constructor() {
    this.setupExpress();
    this.setupWebSocket();
    this.setupRoutes();
  }
  
  private setupExpress() {
    this.app.use(cors());
    this.app.use(express.json({ limit: '10mb' }));
    this.app.use(express.static('public'));
    
    // Request logging
    this.app.use((req, res, next) => {
      console.log(`${req.method} ${req.path} - ${new Date().toISOString()}`);
      next();
    });
  }
  
  private setupWebSocket() {
    this.wss.on('connection', (ws) => {
      console.log('WebSocket client connected');
      
      ws.on('message', async (data) => {
        try {
          const message = JSON.parse(data.toString());
          const response = await this.handleWebSocketMessage(message);
          ws.send(JSON.stringify(response));
        } catch (error) {
          ws.send(JSON.stringify({ 
            error: `Invalid message: ${error.message}` 
          }));
        }
      });
      
      ws.on('close', () => {
        console.log('WebSocket client disconnected');
      });
    });
  }
  
  private async handleWebSocketMessage(message: any) {
    const { service, method, args = [], id } = message;
    
    try {
      const serviceObj = this.services.get(service);
      if (!serviceObj || !serviceObj.isConnected) {
        throw new Error(`Service ${service} not available`);
      }
      
      const result = await serviceObj.bridge.call(method, ...args);
      
      return {
        id,
        success: true,
        result
      };
    } catch (error) {
      return {
        id,
        success: false,
        error: error.message
      };
    }
  }
  
  private setupRoutes() {
    // Health check
    this.app.get('/health', (req, res) => {
      const serviceStatus = Array.from(this.services.entries()).reduce((acc, [name, service]) => {
        acc[name] = {
          connected: service.isConnected,
          lastError: service.lastError
        };
        return acc;
      }, {} as Record<string, any>);
      
      res.json({
        status: 'ok',
        timestamp: new Date().toISOString(),
        services: serviceStatus
      });
    });
    
    // Service status
    this.app.get('/services', (req, res) => {
      const services = Array.from(this.services.entries()).map(([name, service]) => ({
        name,
        connected: service.isConnected,
        lastError: service.lastError
      }));
      
      res.json({ services });
    });
    
    // Cognitive Engine API
    this.app.post('/api/cognitive/process', async (req, res) => {
      try {
        const cognitive = this.services.get('cognitive');
        if (!cognitive?.isConnected) {
          return res.status(503).json({ error: 'Cognitive engine not available' });
        }
        
        const { input, context } = req.body;
        const result = await cognitive.bridge.process(input, context);
        
        res.json({ success: true, result });
      } catch (error) {
        res.status(500).json({ error: error.message });
      }
    });
    
    this.app.get('/api/cognitive/stability', async (req, res) => {
      try {
        const cognitive = this.services.get('cognitive');
        if (!cognitive?.isConnected) {
          return res.status(503).json({ error: 'Cognitive engine not available' });
        }
        
        const stability = await cognitive.bridge.get_current_stability();
        res.json({ success: true, stability });
      } catch (error) {
        res.status(500).json({ error: error.message });
      }
    });
    
    // Memory Vault API
    this.app.post('/api/memory/store', async (req, res) => {
      try {
        const memory = this.services.get('memory');
        if (!memory?.isConnected) {
          return res.status(503).json({ error: 'Memory vault not available' });
        }
        
        const { content, memoryType, metadata } = req.body;
        const memoryId = await memory.bridge.store(content, memoryType, metadata);
        
        res.json({ success: true, memoryId });
      } catch (error) {
        res.status(500).json({ error: error.message });
      }
    });
    
    this.app.get('/api/memory/:id', async (req, res) => {
      try {
        const memory = this.services.get('memory');
        if (!memory?.isConnected) {
          return res.status(503).json({ error: 'Memory vault not available' });
        }
        
        const result = await memory.bridge.retrieve(req.params.id);
        
        if (result) {
          res.json({ success: true, memory: result });
        } else {
          res.status(404).json({ error: 'Memory not found' });
        }
      } catch (error) {
        res.status(500).json({ error: error.message });
      }
    });
    
    this.app.post('/api/memory/search', async (req, res) => {
      try {
        const memory = this.services.get('memory');
        if (!memory?.isConnected) {
          return res.status(503).json({ error: 'Memory vault not available' });
        }
        
        const results = await memory.bridge.search(req.body);
        res.json({ success: true, results });
      } catch (error) {
        res.status(500).json({ error: error.message });
      }
    });
    
    // Eigenvalue Monitor API  
    this.app.post('/api/stability/analyze', async (req, res) => {
      try {
        const eigenvalue = this.services.get('eigenvalue');
        if (!eigenvalue?.isConnected) {
          return res.status(503).json({ error: 'Eigenvalue monitor not available' });
        }
        
        const { matrix } = req.body;
        const analysis = await eigenvalue.bridge.analyze_matrix(matrix);
        
        res.json({ success: true, analysis });
      } catch (error) {
        res.status(500).json({ error: error.message });
      }
    });
    
    this.app.get('/api/stability/metrics', async (req, res) => {
      try {
        const eigenvalue = this.services.get('eigenvalue');
        if (!eigenvalue?.isConnected) {
          return res.status(503).json({ error: 'Eigenvalue monitor not available' });
        }
        
        const metrics = await eigenvalue.bridge.get_stability_metrics();
        res.json({ success: true, metrics });
      } catch (error) {
        res.status(500).json({ error: error.message });
      }
    });
    
    // Generic bridge API
    this.app.post('/api/bridge/:service/:method', async (req, res) => {
      try {
        const { service, method } = req.params;
        const args = req.body.args || [];
        
        const serviceObj = this.services.get(service);
        if (!serviceObj?.isConnected) {
          return res.status(503).json({ error: `Service ${service} not available` });
        }
        
        const result = await serviceObj.bridge.call(method, ...args);
        res.json({ success: true, result });
      } catch (error) {
        res.status(500).json({ error: error.message });
      }
    });
    
    // Error handler
    this.app.use((error: any, req: any, res: any, next: any) => {
      console.error('Express error:', error);
      res.status(500).json({ error: 'Internal server error' });
    });
  }
  
  private async initializeServices() {
    console.log('🔄 Initializing Python bridge services...');
    
    const serviceConfigs = [
      {
        name: 'cognitive',
        factory: () => new CognitiveEngineBridge(),
        required: true
      },
      {
        name: 'memory', 
        factory: () => new MemoryVaultBridge(),
        required: true
      },
      {
        name: 'eigenvalue',
        factory: () => new EigenvalueMonitorBridge(),
        required: true
      },
      {
        name: 'lyapunov',
        factory: () => createPythonBridge('python/stability/lyapunov_analyzer.py'),
        required: false
      },
      {
        name: 'koopman',
        factory: () => createPythonBridge('python/stability/koopman_operator.py'),
        required: false
      }
    ];
    
    for (const config of serviceConfigs) {
      try {
        console.log(`Initializing ${config.name} service...`);
        
        const bridge = config.factory();
        
        // Test connection
        await new Promise((resolve, reject) => {
          const timeout = setTimeout(() => {
            reject(new Error(`Timeout initializing ${config.name}`));
          }, 15000);
          
          bridge.on('ready', () => {
            clearTimeout(timeout);
            resolve(true);
          });
          
          bridge.on('error', (error: Error) => {
            clearTimeout(timeout);
            reject(error);
          });
        });
        
        this.services.set(config.name, {
          name: config.name,
          bridge,
          isConnected: true
        });
        
        console.log(`✅ ${config.name} service initialized`);
        
      } catch (error) {
        console.error(`❌ Failed to initialize ${config.name}: ${error.message}`);
        
        this.services.set(config.name, {
          name: config.name,
          bridge: null,
          isConnected: false,
          lastError: error.message
        });
        
        if (config.required) {
          console.error(`Required service ${config.name} failed to start`);
          // Don't exit - let other services try to start
        }
      }
    }
    
    const connectedServices = Array.from(this.services.values()).filter(s => s.isConnected);
    console.log(`🎯 ${connectedServices.length}/${serviceConfigs.length} services connected`);
  }
  
  async start() {
    try {
      // Initialize Python bridges
      await this.initializeServices();
      
      // Start HTTP server
      await new Promise<void>((resolve) => {
        this.server.listen(this.port, () => {
          console.log(`🚀 TORI Bridge Server running on port ${this.port}`);
          console.log(`📊 Health check: http://localhost:${this.port}/health`);
          console.log(`🔌 WebSocket: ws://localhost:${this.port}`);
          resolve();
        });
      });
      
    } catch (error) {
      console.error('❌ Failed to start TORI Bridge Server:', error);
      process.exit(1);
    }
  }
  
  async shutdown() {
    console.log('🛑 Shutting down TORI Bridge Server...');
    
    // Close Python bridges
    for (const [name, service] of this.services) {
      if (service.isConnected && service.bridge?.close) {
        try {
          await service.bridge.close();
          console.log(`✅ Closed ${name} service`);
        } catch (error) {
          console.error(`❌ Error closing ${name}: ${error.message}`);
        }
      }
    }
    
    // Close WebSocket server
    this.wss.close();
    
    // Close HTTP server
    this.server.close();
    
    console.log('✅ TORI Bridge Server shutdown complete');
  }
}

// Handle shutdown signals
const bridgeServer = new TORIBridgeServer();

process.on('SIGINT', async () => {
  console.log('\nReceived SIGINT, shutting down gracefully...');
  await bridgeServer.shutdown();
  process.exit(0);
});

process.on('SIGTERM', async () => {
  console.log('\nReceived SIGTERM, shutting down gracefully...');
  await bridgeServer.shutdown();
  process.exit(0);
});

// Start server
bridgeServer.start().catch((error) => {
  console.error('Failed to start bridge server:', error);
  process.exit(1);
});

export default TORIBridgeServer;
