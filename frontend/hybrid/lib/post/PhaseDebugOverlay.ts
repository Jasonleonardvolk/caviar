// ${IRIS_ROOT}\frontend\hybrid\lib\post\PhaseDebugOverlay.ts
/**
 * Minimal debug overlay for phase correction tuning
 * Toggle with Alt+P, shows current values + sparkline
 * Zero dependencies, pure DOM manipulation
 */

import { getPhaseConfig } from './PhaseConfig';
import { getPhaseTuner } from './PhaseTuner';

export class PhaseDebugOverlay {
  private container: HTMLDivElement | null = null;
  private visible = false;
  private config = getPhaseConfig();
  private tuner = getPhaseTuner();
  private history: number[] = [];
  private maxHistory = 60;
  
  constructor() {
    this.setupHotkey();
    this.setupStyles();
  }
  
  private setupHotkey(): void {
    window.addEventListener('keydown', (e) => {
      if (e.altKey && (e.key === 'p' || e.key === 'P')) {
        e.preventDefault();
        this.toggle();
      }
    });
  }
  
  private setupStyles(): void {
    if (document.getElementById('phase-debug-styles')) return;
    
    const style = document.createElement('style');
    style.id = 'phase-debug-styles';
    style.textContent = `
      .phase-debug-overlay {
        position: fixed;
        top: 10px;
        right: 10px;
        width: 280px;
        background: rgba(0, 0, 0, 0.85);
        color: #0f0;
        font-family: 'Courier New', monospace;
        font-size: 11px;
        padding: 10px;
        border: 1px solid #0f0;
        border-radius: 4px;
        z-index: 999999;
        pointer-events: none;
        backdrop-filter: blur(4px);
      }
      
      .phase-debug-title {
        font-size: 12px;
        font-weight: bold;
        margin-bottom: 8px;
        text-align: center;
        color: #0ff;
      }
      
      .phase-debug-row {
        display: flex;
        justify-content: space-between;
        margin: 2px 0;
        padding: 1px 0;
      }
      
      .phase-debug-label {
        color: #888;
      }
      
      .phase-debug-value {
        color: #0f0;
        font-weight: bold;
      }
      
      .phase-debug-active {
        color: #ff0;
      }
      
      .phase-debug-sparkline {
        width: 100%;
        height: 30px;
        margin-top: 8px;
        border: 1px solid #0f0;
        background: rgba(0, 255, 0, 0.05);
      }
      
      .phase-debug-help {
        margin-top: 8px;
        padding-top: 8px;
        border-top: 1px solid #333;
        font-size: 10px;
        color: #666;
      }
      
      .phase-debug-metrics {
        margin-top: 8px;
        padding-top: 8px;
        border-top: 1px solid #333;
      }
    `;
    document.head.appendChild(style);
  }
  
  toggle(): void {
    this.visible = !this.visible;
    if (this.visible) {
      this.show();
    } else {
      this.hide();
    }
  }
  
  show(): void {
    if (this.container) return;
    
    this.container = document.createElement('div');
    this.container.className = 'phase-debug-overlay';
    document.body.appendChild(this.container);
    
    // Start update loop
    this.update();
  }
  
  hide(): void {
    if (this.container) {
      this.container.remove();
      this.container = null;
    }
  }
  
  private update(): void {
    if (!this.container || !this.visible) return;
    
    const params = this.config.getAll();
    const metrics = this.tuner.getMetrics();
    
    // Update history for sparkline
    this.history.push(metrics.frameTime);
    if (this.history.length > this.maxHistory) {
      this.history.shift();
    }
    
    // Build HTML
    const methodColor = params.activeMethod === 'off' ? '#666' : '#0f0';
    
    this.container.innerHTML = `
      <div class="phase-debug-title">PHASE CORRECTION</div>
      
      <div class="phase-debug-row">
        <span class="phase-debug-label">Method:</span>
        <span class="phase-debug-value" style="color: ${methodColor}">${params.activeMethod.toUpperCase()}</span>
      </div>
      
      ${params.activeMethod === 'tv' ? `
        <div class="phase-debug-row">
          <span class="phase-debug-label">TV Lambda:</span>
          <span class="phase-debug-value">${params.tvLambda.toFixed(3)}</span>
        </div>
        <div class="phase-debug-row">
          <span class="phase-debug-label">Max Corr:</span>
          <span class="phase-debug-value">${params.tvMaxCorrection.toFixed(3)}</span>
        </div>
        <div class="phase-debug-row">
          <span class="phase-debug-label">Use Mask:</span>
          <span class="phase-debug-value">${params.tvUseMask ? 'YES' : 'NO'}</span>
        </div>
      ` : ''}
      
      ${params.activeMethod === 'zernike' ? `
        <div class="phase-debug-row">
          <span class="phase-debug-label">Defocus:</span>
          <span class="phase-debug-value">${params.zernikeDefocus.toFixed(3)}</span>
        </div>
        <div class="phase-debug-row">
          <span class="phase-debug-label">Astig 0:</span>
          <span class="phase-debug-value">${params.zernikeAstig0.toFixed(3)}</span>
        </div>
        <div class="phase-debug-row">
          <span class="phase-debug-label">Astig 45:</span>
          <span class="phase-debug-value">${params.zernikeAstig45.toFixed(3)}</span>
        </div>
      ` : ''}
      
      <div class="phase-debug-metrics">
        <div class="phase-debug-row">
          <span class="phase-debug-label">Frame Time:</span>
          <span class="phase-debug-value">${metrics.frameTime.toFixed(2)}ms</span>
        </div>
        <div class="phase-debug-row">
          <span class="phase-debug-label">Auto-Adapt:</span>
          <span class="phase-debug-value" style="color: ${params.autoAdaptEnabled ? '#0f0' : '#666'}">${params.autoAdaptEnabled ? 'ON' : 'OFF'}</span>
        </div>
        ${metrics.seamsDetected > 0 ? `
          <div class="phase-debug-row">
            <span class="phase-debug-label">Seams:</span>
            <span class="phase-debug-value" style="color: #f00">${(metrics.seamsDetected * 100).toFixed(0)}%</span>
          </div>
        ` : ''}
      </div>
      
      <canvas class="phase-debug-sparkline" width="260" height="30"></canvas>
      
      <div class="phase-debug-help">
        Alt+↑↓ Lambda | Alt+←→ MaxCorr<br>
        Alt+Z Method | Alt+A Auto | Alt+D Dump
      </div>
    `;
    
    // Draw sparkline
    this.drawSparkline();
    
    // Schedule next update
    requestAnimationFrame(() => this.update());
  }
  
  private drawSparkline(): void {
    const canvas = this.container?.querySelector('canvas');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const width = canvas.width;
    const height = canvas.height;
    
    // Clear
    ctx.clearRect(0, 0, width, height);
    
    if (this.history.length < 2) return;
    
    // Find min/max for scaling
    const min = Math.min(...this.history);
    const max = Math.max(...this.history);
    const range = Math.max(0.1, max - min);
    
    // Draw line
    ctx.strokeStyle = '#0f0';
    ctx.lineWidth = 1;
    ctx.beginPath();
    
    this.history.forEach((value, i) => {
      const x = (i / (this.maxHistory - 1)) * width;
      const y = height - ((value - min) / range) * height;
      
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    
    ctx.stroke();
    
    // Draw threshold line (16.67ms = 60fps)
    const targetMs = 16.67;
    if (targetMs >= min && targetMs <= max) {
      const targetY = height - ((targetMs - min) / range) * height;
      ctx.strokeStyle = '#ff0';
      ctx.setLineDash([2, 2]);
      ctx.beginPath();
      ctx.moveTo(0, targetY);
      ctx.lineTo(width, targetY);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }
}

// Auto-create instance
let overlayInstance: PhaseDebugOverlay | null = null;

export function getPhaseDebugOverlay(): PhaseDebugOverlay {
  if (!overlayInstance) {
    overlayInstance = new PhaseDebugOverlay();
  }
  return overlayInstance;
}

// Auto-initialize on import
if (typeof window !== 'undefined') {
  getPhaseDebugOverlay();
}
