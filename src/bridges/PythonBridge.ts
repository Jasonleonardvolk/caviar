import { spawn, ChildProcess } from 'child_process';
import { EventEmitter } from 'events';
import { v4 as uuidv4 } from 'uuid';
import * as path from 'path';

interface PendingCall {
  resolve: (value: any) => void;
  reject: (reason: any) => void;
  timeout: NodeJS.Timeout;
}

interface PythonMessage {
  id?: string;
  type: 'ready' | 'response' | 'event' | 'error';
  method?: string;
  result?: any;
  error?: string;
  event?: string;
  data?: any;
}

export class PythonBridge extends EventEmitter {
  private process: ChildProcess | null = null;
  private pendingCalls: Map<string, PendingCall> = new Map();
  private isReady: boolean = false;
  private readyPromise: Promise<void>;
  private messageBuffer: string = '';
  
  constructor(private modulePath: string, private config: any = {}) {
    super();
    this.readyPromise = this.initialize();
  }
  
  private async initialize(): Promise<void> {
    return new Promise((resolve, reject) => {
      const pythonPath = this.config.pythonPath || 'python';
      const bridgeServerPath = path.join(__dirname, 'python_bridge_server.py');
      
      // Spawn Python process
      this.process = spawn(pythonPath, [
        '-u',  // Unbuffered output
        bridgeServerPath,
        this.modulePath
      ], {
        stdio: ['pipe', 'pipe', 'pipe'],
        env: { ...process.env, PYTHONPATH: path.dirname(this.modulePath) }
      });
      
      // Handle stdout
      this.process.stdout?.on('data', (data) => {
        this.messageBuffer += data.toString();
        this.processMessages();
      });
      
      // Handle stderr
      this.process.stderr?.on('data', (data) => {
        console.error(`Python Bridge Error: ${data}`);
        this.emit('error', new Error(data.toString()));
      });
      
      // Handle process exit
      this.process.on('exit', (code) => {
        console.log(`Python process exited with code ${code}`);
        this.isReady = false;
        this.rejectAllPending(new Error(`Python process exited with code ${code}`));
        this.emit('exit', code);
      });
      
      // Handle process error
      this.process.on('error', (err) => {
        console.error('Python process error:', err);
        this.isReady = false;
        this.rejectAllPending(err);
        reject(err);
      });
      
      // Wait for ready signal
      this.once('ready', () => {
        this.isReady = true;
        resolve();
      });
      
      // Timeout
      setTimeout(() => {
        if (!this.isReady) {
          const error = new Error('Python bridge initialization timeout');
          this.rejectAllPending(error);
          reject(error);
        }
      }, this.config.initTimeout || 30000);
    });
  }
  
  private processMessages(): void {
    const lines = this.messageBuffer.split('\n');
    this.messageBuffer = lines.pop() || '';
    
    for (const line of lines) {
      if (!line.trim()) continue;
      
      try {
        const message: PythonMessage = JSON.parse(line);
        this.handleMessage(message);
      } catch (e) {
        console.error('Failed to parse Python message:', line, e);
      }
    }
  }
  
  private handleMessage(message: PythonMessage): void {
    switch (message.type) {
      case 'ready':
        this.emit('ready');
        break;
        
      case 'response':
        if (message.id) {
          const pending = this.pendingCalls.get(message.id);
          if (pending) {
            clearTimeout(pending.timeout);
            this.pendingCalls.delete(message.id);
            
            if (message.error) {
              pending.reject(new Error(message.error));
            } else {
              pending.resolve(message.result);
            }
          }
        }
        break;
        
      case 'event':
        if (message.event) {
          this.emit(message.event, message.data);
        }
        break;
        
      case 'error':
        console.error('Python error:', message.error);
        this.emit('error', new Error(message.error || 'Unknown Python error'));
        break;
    }
  }
  
  public async call(method: string, ...args: any[]): Promise<any> {
    // Wait for initialization
    await this.readyPromise;
    
    if (!this.isReady || !this.process || !this.process.stdin) {
      throw new Error('Python bridge not ready');
    }
    
    const callId = uuidv4();
    
    return new Promise((resolve, reject) => {
      // Set timeout
      const timeout = setTimeout(() => {
        this.pendingCalls.delete(callId);
        reject(new Error(`Call timeout: ${method}`));
      }, this.config.callTimeout || 30000);
      
      // Store pending call
      this.pendingCalls.set(callId, { resolve, reject, timeout });
      
      // Send call
      const message = JSON.stringify({
        id: callId,
        method,
        args
      }) + '\n';
      
      try {
        this.process!.stdin!.write(message);
      } catch (err) {
        clearTimeout(timeout);
        this.pendingCalls.delete(callId);
        reject(err);
      }
    });
  }
  
  private rejectAllPending(error: Error): void {
    for (const [id, pending] of this.pendingCalls) {
      clearTimeout(pending.timeout);
      pending.reject(error);
    }
    this.pendingCalls.clear();
  }
  
  public async close(): Promise<void> {
    if (this.process) {
      // Send shutdown signal
      try {
        await this.call('shutdown');
      } catch (e) {
        // Ignore errors during shutdown
      }
      
      // Give it a moment to shut down gracefully
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Kill process if still running
      if (!this.process.killed) {
        this.process.kill();
      }
      
      this.process = null;
    }
    
    this.rejectAllPending(new Error('Bridge closed'));
    this.isReady = false;
  }
  
  public isActive(): boolean {
    return this.isReady && this.process !== null && !this.process.killed;
  }
  
  // Convenience methods for common Python modules
  
  public async importModule(moduleName: string): Promise<void> {
    await this.call('import_module', moduleName);
  }
  
  public async getAttr(attrName: string): Promise<any> {
    return await this.call('get_attr', attrName);
  }
  
  public async setAttr(attrName: string, value: any): Promise<void> {
    await this.call('set_attr', attrName, value);
  }
  
  public async evaluate(expression: string): Promise<any> {
    return await this.call('eval', expression);
  }
  
  public async execute(code: string): Promise<void> {
    await this.call('exec', code);
  }
}

// Export factory function
export function createPythonBridge(modulePath: string, config?: any): PythonBridge {
  return new PythonBridge(modulePath, config);
}

// Type definitions for specific Python modules
export interface CognitiveEngineInterface {
  process(input: any, context?: any): Promise<any>;
  get_current_stability(): Promise<any>;
  reset(): Promise<void>;
}

export interface MemoryVaultInterface {
  store(content: any, memoryType: string, metadata?: any): Promise<string>;
  retrieve(memoryId: string): Promise<any>;
  search(query?: any): Promise<any[]>;
  delete(memoryId: string): Promise<boolean>;
  get_statistics(): Promise<any>;
}

export interface EigenvalueMonitorInterface {
  analyze_matrix(matrix: number[][]): Promise<any>;
  compute_lyapunov_stability(matrix: number[][]): Promise<any>;
  predict_epsilon_cloud(stepsAhead?: number): Promise<any>;
  get_stability_metrics(): Promise<any>;
}

// Specialized bridge classes
export class CognitiveEngineBridge extends PythonBridge implements CognitiveEngineInterface {
  constructor(config?: any) {
    const modulePath = path.join(process.cwd(), 'python/core/CognitiveEngine.py');
    super(modulePath, config);
  }
  
  async process(input: any, context?: any): Promise<any> {
    return await this.call('process', input, context);
  }
  
  async get_current_stability(): Promise<any> {
    return await this.call('get_current_stability');
  }
  
  async reset(): Promise<void> {
    await this.call('reset');
  }
}

export class MemoryVaultBridge extends PythonBridge implements MemoryVaultInterface {
  constructor(config?: any) {
    const modulePath = path.join(process.cwd(), 'python/core/memory_vault.py');
    super(modulePath, config);
  }
  
  async store(content: any, memoryType: string, metadata?: any): Promise<string> {
    return await this.call('store', content, memoryType, metadata);
  }
  
  async retrieve(memoryId: string): Promise<any> {
    return await this.call('retrieve', memoryId);
  }
  
  async search(query?: any): Promise<any[]> {
    return await this.call('search', query);
  }
  
  async delete(memoryId: string): Promise<boolean> {
    return await this.call('delete', memoryId);
  }
  
  async get_statistics(): Promise<any> {
    return await this.call('get_statistics');
  }
}

export class EigenvalueMonitorBridge extends PythonBridge implements EigenvalueMonitorInterface {
  constructor(config?: any) {
    const modulePath = path.join(process.cwd(), 'python/stability/eigenvalue_monitor.py');
    super(modulePath, config);
  }
  
  async analyze_matrix(matrix: number[][]): Promise<any> {
    return await this.call('analyze_matrix', matrix);
  }
  
  async compute_lyapunov_stability(matrix: number[][]): Promise<any> {
    return await this.call('compute_lyapunov_stability', matrix);
  }
  
  async predict_epsilon_cloud(stepsAhead?: number): Promise<any> {
    return await this.call('predict_epsilon_cloud', stepsAhead);
  }
  
  async get_stability_metrics(): Promise<any> {
    return await this.call('get_stability_metrics');
  }
}