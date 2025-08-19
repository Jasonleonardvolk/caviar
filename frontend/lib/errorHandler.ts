/**
 * Bulletproof Error Handler for TORI/Saigon v5
 * Ensures system never fails without recovery
 */

export interface ErrorContext {
  component: string;
  action: string;
  userId?: string;
  data?: any;
  timestamp: number;
}

export interface RecoveryStrategy {
  type: 'retry' | 'fallback' | 'reset' | 'reload';
  action: () => Promise<void>;
  maxAttempts?: number;
}

class BulletproofErrorHandler {
  private errorLog: Array<{error: Error, context: ErrorContext}> = [];
  private recoveryAttempts: Map<string, number> = new Map();
  private userNotified: boolean = false;
  private fallbackMode: boolean = false;
  
  constructor() {
    this.setupGlobalHandlers();
    this.setupPerformanceMonitoring();
  }
  
  private setupGlobalHandlers() {
    // Catch unhandled errors
    window.addEventListener('error', (event) => {
      this.handleError(
        new Error(event.message),
        {
          component: 'global',
          action: 'unhandled_error',
          timestamp: Date.now()
        }
      );
      event.preventDefault();
    });
    
    // Catch unhandled promise rejections
    window.addEventListener('unhandledrejection', (event) => {
      this.handleError(
        new Error(event.reason),
        {
          component: 'promise',
          action: 'unhandled_rejection',
          timestamp: Date.now()
        }
      );
      event.preventDefault();
    });
    
    // Monitor WebGPU device loss
    if ('gpu' in navigator) {
      navigator.gpu.requestAdapter().then(adapter => {
        adapter?.requestDevice().then(device => {
          device.lost.then((info) => {
            this.handleDeviceLoss(info);
          });
        });
      });
    }
  }
  
  private setupPerformanceMonitoring() {
    // Monitor performance degradation
    let lastFrameTime = performance.now();
    let lowFpsCount = 0;
    
    const checkPerformance = () => {
      const currentTime = performance.now();
      const deltaTime = currentTime - lastFrameTime;
      lastFrameTime = currentTime;
      
      // If frame time > 100ms (< 10 FPS), we have a problem
      if (deltaTime > 100) {
        lowFpsCount++;
        
        if (lowFpsCount > 10 && !this.fallbackMode) {
          this.handlePerformanceDegradation();
        }
      } else {
        lowFpsCount = Math.max(0, lowFpsCount - 1);
      }
      
      requestAnimationFrame(checkPerformance);
    };
    
    requestAnimationFrame(checkPerformance);
  }
  
  public handleError(error: Error, context: ErrorContext): void {
    // Log error
    this.errorLog.push({ error, context });
    console.error(`[ErrorHandler] ${context.component}/${context.action}:`, error);
    
    // Log to backend
    this.logToBackend(error, context);
    
    // Determine recovery strategy
    const strategy = this.determineRecoveryStrategy(error, context);
    
    // Execute recovery
    this.executeRecovery(strategy, error, context);
    
    // Notify user if necessary
    if (!this.userNotified && this.shouldNotifyUser(error, context)) {
      this.notifyUser(error, context, strategy);
      this.userNotified = true;
    }
  }
  
  private determineRecoveryStrategy(error: Error, context: ErrorContext): RecoveryStrategy {
    const errorMessage = error.message.toLowerCase();
    
    // Network errors - retry
    if (errorMessage.includes('network') || errorMessage.includes('fetch')) {
      return {
        type: 'retry',
        action: async () => {
          await this.wait(1000);
          // Retry the failed network request
          window.dispatchEvent(new CustomEvent('retry-request', { detail: context }));
        },
        maxAttempts: 3
      };
    }
    
    // GPU/Rendering errors - fallback
    if (errorMessage.includes('gpu') || errorMessage.includes('webgpu') || 
        errorMessage.includes('render') || context.component === 'renderer') {
      return {
        type: 'fallback',
        action: async () => {
          this.fallbackMode = true;
          window.dispatchEvent(new CustomEvent('switch-to-fallback-renderer'));
        }
      };
    }
    
    // Memory errors - reset
    if (errorMessage.includes('memory') || errorMessage.includes('oom')) {
      return {
        type: 'reset',
        action: async () => {
          // Clear caches and reset
          this.clearMemory();
          window.dispatchEvent(new CustomEvent('reset-system'));
        }
      };
    }
    
    // Critical errors - reload
    if (errorMessage.includes('critical') || this.errorLog.length > 50) {
      return {
        type: 'reload',
        action: async () => {
          await this.saveState();
          window.location.reload();
        }
      };
    }
    
    // Default - try to recover gracefully
    return {
      type: 'fallback',
      action: async () => {
        this.fallbackMode = true;
        this.reducedQualityMode();
      }
    };
  }
  
  private async executeRecovery(strategy: RecoveryStrategy, error: Error, context: ErrorContext): Promise<void> {
    const attemptKey = `${context.component}/${context.action}`;
    const attempts = this.recoveryAttempts.get(attemptKey) || 0;
    
    if (strategy.maxAttempts && attempts >= strategy.maxAttempts) {
      console.warn(`[ErrorHandler] Max recovery attempts reached for ${attemptKey}`);
      
      // Fall back to safer strategy
      if (strategy.type === 'retry') {
        strategy = {
          type: 'fallback',
          action: async () => this.reducedQualityMode()
        };
      }
    }
    
    this.recoveryAttempts.set(attemptKey, attempts + 1);
    
    try {
      console.log(`[ErrorHandler] Executing ${strategy.type} recovery`);
      await strategy.action();
      
      // Reset attempts on successful recovery
      this.recoveryAttempts.delete(attemptKey);
      
    } catch (recoveryError) {
      console.error(`[ErrorHandler] Recovery failed:`, recoveryError);
      
      // Last resort - safe mode
      this.enterSafeMode();
    }
  }
  
  private handleDeviceLoss(info: any): void {
    console.error('[ErrorHandler] GPU device lost:', info);
    
    this.handleError(
      new Error(`GPU device lost: ${info.reason}`),
      {
        component: 'gpu',
        action: 'device_lost',
        data: info,
        timestamp: Date.now()
      }
    );
  }
  
  private handlePerformanceDegradation(): void {
    console.warn('[ErrorHandler] Performance degradation detected');
    
    this.handleError(
      new Error('Performance below acceptable threshold'),
      {
        component: 'performance',
        action: 'low_fps',
        timestamp: Date.now()
      }
    );
  }
  
  private reducedQualityMode(): void {
    console.log('[ErrorHandler] Switching to reduced quality mode');
    
    // Reduce rendering quality
    window.dispatchEvent(new CustomEvent('set-quality', { 
      detail: { quality: 'minimal' } 
    }));
    
    // Disable non-essential features
    window.dispatchEvent(new CustomEvent('disable-features', {
      detail: { 
        features: ['particles', 'shadows', 'post-processing'] 
      }
    }));
    
    this.showToast('Switched to reduced quality mode for stability', 'warning');
  }
  
  private enterSafeMode(): void {
    console.log('[ErrorHandler] Entering safe mode');
    
    // Disable all advanced features
    this.fallbackMode = true;
    
    // Switch to most basic renderer
    window.dispatchEvent(new CustomEvent('enter-safe-mode'));
    
    // Show safe mode UI
    this.showSafeModeUI();
  }
  
  private clearMemory(): void {
    // Clear all caches
    if ('caches' in window) {
      caches.keys().then(names => {
        names.forEach(name => caches.delete(name));
      });
    }
    
    // Clear session storage
    sessionStorage.clear();
    
    // Clear large objects from memory
    window.dispatchEvent(new CustomEvent('clear-memory'));
    
    // Force garbage collection if available
    if ((window as any).gc) {
      (window as any).gc();
    }
  }
  
  private async saveState(): Promise<void> {
    try {
      const state = {
        timestamp: Date.now(),
        errors: this.errorLog.slice(-10), // Last 10 errors
        userId: sessionStorage.getItem('userId'),
        session: sessionStorage.getItem('session')
      };
      
      localStorage.setItem('crash_recovery_state', JSON.stringify(state));
    } catch (e) {
      console.error('[ErrorHandler] Failed to save state:', e);
    }
  }
  
  private logToBackend(error: Error, context: ErrorContext): void {
    fetch('/api/v2/hybrid/log_error', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        error: error.message,
        stack: error.stack,
        context,
        renderer: this.fallbackMode ? 'fallback' : 'normal',
        user_id: context.userId || sessionStorage.getItem('userId')
      })
    }).catch(e => {
      console.error('[ErrorHandler] Failed to log to backend:', e);
    });
  }
  
  private shouldNotifyUser(error: Error, context: ErrorContext): boolean {
    // Don't notify for minor/recoverable errors
    if (context.component === 'performance' && this.recoveryAttempts.size < 3) {
      return false;
    }
    
    // Don't notify for successfully recovered errors
    const attemptKey = `${context.component}/${context.action}`;
    if (this.recoveryAttempts.get(attemptKey) === 0) {
      return false;
    }
    
    // Notify for critical or repeated errors
    return this.errorLog.length > 5 || error.message.includes('critical');
  }
  
  private notifyUser(error: Error, context: ErrorContext, strategy: RecoveryStrategy): void {
    let message = '';
    let type: 'error' | 'warning' | 'info' = 'warning';
    
    switch (strategy.type) {
      case 'retry':
        message = 'Connection issue detected. Retrying...';
        type = 'warning';
        break;
      case 'fallback':
        message = 'Switched to compatibility mode for stability';
        type = 'warning';
        break;
      case 'reset':
        message = 'System reset to recover from error';
        type = 'warning';
        break;
      case 'reload':
        message = 'Page will reload to recover from error';
        type = 'error';
        break;
    }
    
    this.showToast(message, type);
  }
  
  private showToast(message: string, type: 'error' | 'warning' | 'info' = 'info'): void {
    const toast = document.createElement('div');
    toast.className = `error-toast error-toast-${type}`;
    toast.innerHTML = `
      <div class="error-toast-content">
        <span class="error-toast-icon">
          ${type === 'error' ? '❌' : type === 'warning' ? '⚠️' : 'ℹ️'}
        </span>
        <span class="error-toast-message">${message}</span>
        <button class="error-toast-close" onclick="this.parentElement.parentElement.remove()">✕</button>
      </div>
    `;
    
    // Add styles if not already present
    if (!document.getElementById('error-handler-styles')) {
      const styles = document.createElement('style');
      styles.id = 'error-handler-styles';
      styles.innerHTML = `
        .error-toast {
          position: fixed;
          top: 20px;
          right: 20px;
          background: rgba(20, 20, 30, 0.95);
          border: 1px solid rgba(100, 100, 150, 0.3);
          border-radius: 8px;
          padding: 12px 16px;
          color: #e0e0e0;
          font-family: -apple-system, sans-serif;
          font-size: 14px;
          z-index: 999999;
          animation: slideIn 0.3s ease;
          max-width: 400px;
          backdrop-filter: blur(10px);
        }
        
        .error-toast-error {
          border-color: rgba(255, 50, 50, 0.5);
          background: rgba(255, 50, 50, 0.1);
        }
        
        .error-toast-warning {
          border-color: rgba(255, 200, 50, 0.5);
          background: rgba(255, 200, 50, 0.1);
        }
        
        .error-toast-info {
          border-color: rgba(50, 150, 255, 0.5);
          background: rgba(50, 150, 255, 0.1);
        }
        
        .error-toast-content {
          display: flex;
          align-items: center;
          gap: 10px;
        }
        
        .error-toast-icon {
          font-size: 18px;
        }
        
        .error-toast-message {
          flex: 1;
        }
        
        .error-toast-close {
          background: none;
          border: none;
          color: inherit;
          cursor: pointer;
          padding: 0 4px;
          font-size: 18px;
          opacity: 0.7;
        }
        
        .error-toast-close:hover {
          opacity: 1;
        }
        
        @keyframes slideIn {
          from {
            transform: translateX(100%);
            opacity: 0;
          }
          to {
            transform: translateX(0);
            opacity: 1;
          }
        }
      `;
      document.head.appendChild(styles);
    }
    
    document.body.appendChild(toast);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
      toast.style.animation = 'slideOut 0.3s ease';
      setTimeout(() => toast.remove(), 300);
    }, 5000);
  }
  
  private showSafeModeUI(): void {
    const safeModeUI = document.createElement('div');
    safeModeUI.id = 'safe-mode-ui';
    safeModeUI.innerHTML = `
      <div style="
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        background: rgba(255, 200, 0, 0.1);
        border-bottom: 2px solid rgba(255, 200, 0, 0.5);
        padding: 10px;
        text-align: center;
        color: #ffcc00;
        font-family: -apple-system, sans-serif;
        z-index: 999998;
      ">
        <strong>⚠️ Safe Mode Active</strong> - 
        Running with reduced features for stability. 
        <button onclick="location.reload()" style="
          margin-left: 10px;
          padding: 4px 12px;
          background: rgba(255, 200, 0, 0.2);
          border: 1px solid rgba(255, 200, 0, 0.5);
          border-radius: 4px;
          color: inherit;
          cursor: pointer;
        ">Restart</button>
      </div>
    `;
    
    document.body.appendChild(safeModeUI);
  }
  
  private wait(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
  
  public getErrorLog(): Array<{error: Error, context: ErrorContext}> {
    return this.errorLog;
  }
  
  public clearErrorLog(): void {
    this.errorLog = [];
    this.recoveryAttempts.clear();
    this.userNotified = false;
  }
  
  public isInFallbackMode(): boolean {
    return this.fallbackMode;
  }
}

// Create global instance
const errorHandler = new BulletproofErrorHandler();

// Export for use in other modules
export default errorHandler;

// Helper function for manual error reporting
export function reportError(error: Error | string, context?: Partial<ErrorContext>) {
  const err = typeof error === 'string' ? new Error(error) : error;
  errorHandler.handleError(err, {
    component: context?.component || 'unknown',
    action: context?.action || 'manual_report',
    userId: context?.userId,
    data: context?.data,
    timestamp: Date.now()
  });
}
