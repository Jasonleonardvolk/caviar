/**
 * UIController.ts
 * 
 * Central UI control system for the holographic display
 * Manages all user interactions and display settings
 */

import type { HolographicPipelineIntegrator } from './HolographicPipelineIntegrator';
import type { AdaptiveRenderer } from '../adaptiveRenderer';

export interface UIConfig {
  container: HTMLElement;
  integrator: HolographicPipelineIntegrator;
  adaptiveRenderer: AdaptiveRenderer;
  onModeChange?: (mode: string) => void;
  onQualityChange?: (quality: string) => void;
}

export class UIController {
  private container: HTMLElement;
  private integrator: HolographicPipelineIntegrator;
  private adaptiveRenderer: AdaptiveRenderer;
  
  // UI Elements
  private controlPanel: HTMLElement | null = null;
  private statusBar: HTMLElement | null = null;
  private debugPanel: HTMLElement | null = null;
  
  // State
  private currentMode: 'quilt' | 'stereo' | 'hologram' = 'quilt';
  private currentQuality: 'low' | 'medium' | 'high' = 'medium';
  private debugVisible = false;
  private parallaxEnabled = true;
  
  // Keyboard shortcuts
  private shortcuts = new Map<string, () => void>();
  
  constructor(config: UIConfig) {
    this.container = config.container;
    this.integrator = config.integrator;
    this.adaptiveRenderer = config.adaptiveRenderer;
    
    this.setupUI();
    this.setupKeyboardShortcuts();
    this.startStatusUpdates();
  }
  
  /**
   * Create the UI elements
   */
  private setupUI(): void {
    // Create control panel
    this.controlPanel = this.createControlPanel();
    this.container.appendChild(this.controlPanel);
    
    // Create status bar
    this.statusBar = this.createStatusBar();
    this.container.appendChild(this.statusBar);
    
    // Create debug panel (hidden by default)
    this.debugPanel = this.createDebugPanel();
    this.debugPanel.style.display = 'none';
    this.container.appendChild(this.debugPanel);
  }
  
  /**
   * Create the main control panel
   */
  private createControlPanel(): HTMLElement {
    const panel = document.createElement('div');
    panel.className = 'holographic-controls';
    panel.style.cssText = `
      position: fixed;
      top: 10px;
      right: 10px;
      background: rgba(0, 0, 0, 0.8);
      border: 1px solid #0f0;
      border-radius: 8px;
      padding: 15px;
      color: #0f0;
      font-family: monospace;
      font-size: 12px;
      z-index: 1000;
      min-width: 200px;
    `;
    
    panel.innerHTML = `
      <h3 style="margin: 0 0 10px 0; font-size: 14px;">HOLOGRAPHIC DISPLAY</h3>
      
      <div class="control-group" style="margin-bottom: 10px;">
        <label style="display: block; margin-bottom: 5px;">Mode:</label>
        <select id="mode-select" style="width: 100%; background: #000; color: #0f0; border: 1px solid #0f0; padding: 3px;">
          <option value="quilt">Quilt (Multi-View)</option>
          <option value="stereo">Stereo (L/R)</option>
          <option value="hologram">Hologram (Phase)</option>
        </select>
      </div>
      
      <div class="control-group" style="margin-bottom: 10px;">
        <label style="display: block; margin-bottom: 5px;">Quality:</label>
        <div style="display: flex; gap: 5px;">
          <button class="quality-btn" data-quality="low" style="flex: 1; background: #000; color: #0f0; border: 1px solid #0f0; padding: 5px; cursor: pointer;">Low</button>
          <button class="quality-btn" data-quality="medium" style="flex: 1; background: #0f0; color: #000; border: 1px solid #0f0; padding: 5px; cursor: pointer;">Med</button>
          <button class="quality-btn" data-quality="high" style="flex: 1; background: #000; color: #0f0; border: 1px solid #0f0; padding: 5px; cursor: pointer;">High</button>
        </div>
      </div>
      
      <div class="control-group" style="margin-bottom: 10px;">
        <label style="display: block; margin-bottom: 5px;">Propagation Distance:</label>
        <input type="range" id="distance-slider" min="10" max="200" value="50" style="width: 100%;">
        <div id="distance-value" style="text-align: center; margin-top: 5px;">0.5m</div>
      </div>
      
      <div class="control-group" style="margin-bottom: 10px;">
        <label style="display: flex; align-items: center; cursor: pointer;">
          <input type="checkbox" id="parallax-toggle" checked style="margin-right: 5px;">
          Head Tracking
        </label>
      </div>
      
      <div style="border-top: 1px solid #0f0; margin-top: 10px; padding-top: 10px;">
        <button id="debug-toggle" style="width: 100%; background: #000; color: #0f0; border: 1px solid #0f0; padding: 5px; cursor: pointer;">
          Show Debug [D]
        </button>
      </div>
      
      <div style="margin-top: 10px; font-size: 10px; opacity: 0.7;">
        Space: Toggle Parallax<br>
        Q: Cycle Quality<br>
        M: Cycle Mode<br>
        R: Reset View
      </div>
    `;
    
    // Attach event listeners
    this.attachControlListeners(panel);
    
    return panel;
  }
  
  /**
   * Create the status bar
   */
  private createStatusBar(): HTMLElement {
    const bar = document.createElement('div');
    bar.className = 'holographic-status';
    bar.style.cssText = `
      position: fixed;
      bottom: 10px;
      left: 10px;
      background: rgba(0, 0, 0, 0.8);
      border: 1px solid #0f0;
      border-radius: 4px;
      padding: 5px 10px;
      color: #0f0;
      font-family: monospace;
      font-size: 11px;
      z-index: 1000;
    `;
    
    bar.innerHTML = `
      <span id="fps-counter">FPS: --</span> | 
      <span id="view-counter">Views: --</span> | 
      <span id="field-size">Field: --</span> |
      <span id="device-tier">Tier: --</span>
    `;
    
    return bar;
  }
  
  /**
   * Create the debug panel
   */
  private createDebugPanel(): HTMLElement {
    const panel = document.createElement('div');
    panel.className = 'holographic-debug';
    panel.style.cssText = `
      position: fixed;
      top: 10px;
      left: 10px;
      background: rgba(0, 0, 0, 0.9);
      border: 1px solid #0f0;
      border-radius: 8px;
      padding: 15px;
      color: #0f0;
      font-family: monospace;
      font-size: 11px;
      z-index: 1000;
      min-width: 300px;
    `;
    
    panel.innerHTML = `
      <h3 style="margin: 0 0 10px 0; font-size: 14px;">DEBUG INFO</h3>
      <div id="debug-content" style="line-height: 1.6;">
        <div>GPU: <span id="gpu-info">Checking...</span></div>
        <div>Adapter: <span id="adapter-info">...</span></div>
        <div>Max Texture: <span id="max-texture">...</span></div>
        <div>Workgroup Size: <span id="workgroup-size">...</span></div>
        <hr style="border: none; border-top: 1px solid #0f0; margin: 10px 0;">
        <div>Frame Time: <span id="frame-time">--</span>ms</div>
        <div>Propagation: <span id="prop-time">--</span>ms</div>
        <div>Encoding: <span id="encode-time">--</span>ms</div>
        <div>Render: <span id="render-time">--</span>ms</div>
        <hr style="border: none; border-top: 1px solid #0f0; margin: 10px 0;">
        <div>Head Pose:</div>
        <div style="margin-left: 10px;">
          X: <span id="pose-x">0.00</span><br>
          Y: <span id="pose-y">0.00</span><br>
          Z: <span id="pose-z">0.00</span><br>
          RX: <span id="pose-rx">0.00</span>°<br>
          RY: <span id="pose-ry">0.00</span>°
        </div>
      </div>
    `;
    
    return panel;
  }
  
  /**
   * Attach event listeners to control panel
   */
  private attachControlListeners(panel: HTMLElement): void {
    // Mode selector
    const modeSelect = panel.querySelector('#mode-select') as HTMLSelectElement;
    modeSelect?.addEventListener('change', () => {
      this.setMode(modeSelect.value as any);
    });
    
    // Quality buttons
    panel.querySelectorAll('.quality-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const quality = btn.getAttribute('data-quality') as any;
        this.setQuality(quality);
        this.updateQualityButtons();
      });
    });
    
    // Distance slider
    const distanceSlider = panel.querySelector('#distance-slider') as HTMLInputElement;
    const distanceValue = panel.querySelector('#distance-value') as HTMLElement;
    distanceSlider?.addEventListener('input', () => {
      const distance = parseInt(distanceSlider.value) / 100;
      this.integrator.setPropagationDistance(distance);
      if (distanceValue) {
        distanceValue.textContent = `${distance.toFixed(2)}m`;
      }
    });
    
    // Parallax toggle
    const parallaxToggle = panel.querySelector('#parallax-toggle') as HTMLInputElement;
    parallaxToggle?.addEventListener('change', () => {
      this.parallaxEnabled = parallaxToggle.checked;
      // Trigger parallax toggle in pose controller
      document.dispatchEvent(new CustomEvent('parallax:toggle', { 
        detail: { enabled: this.parallaxEnabled }
      }));
    });
    
    // Debug toggle
    const debugToggle = panel.querySelector('#debug-toggle') as HTMLButtonElement;
    debugToggle?.addEventListener('click', () => {
      this.toggleDebug();
    });
  }
  
  /**
   * Setup keyboard shortcuts
   */
  private setupKeyboardShortcuts(): void {
    // Define shortcuts
    this.shortcuts.set(' ', () => this.toggleParallax());
    this.shortcuts.set('q', () => this.cycleQuality());
    this.shortcuts.set('Q', () => this.cycleQuality());
    this.shortcuts.set('m', () => this.cycleMode());
    this.shortcuts.set('M', () => this.cycleMode());
    this.shortcuts.set('d', () => this.toggleDebug());
    this.shortcuts.set('D', () => this.toggleDebug());
    this.shortcuts.set('r', () => this.resetView());
    this.shortcuts.set('R', () => this.resetView());
    
    // Listen for keypress
    document.addEventListener('keydown', (e) => {
      const handler = this.shortcuts.get(e.key);
      if (handler && !e.ctrlKey && !e.altKey && !e.metaKey) {
        e.preventDefault();
        handler();
      }
    });
  }
  
  /**
   * Start status bar updates
   */
  private startStatusUpdates(): void {
    setInterval(() => {
      const stats = this.integrator.getStats();
      const settings = this.adaptiveRenderer.settings;
      
      // Update status bar
      const fpsElement = document.querySelector('#fps-counter');
      const viewElement = document.querySelector('#view-counter');
      const fieldElement = document.querySelector('#field-size');
      const tierElement = document.querySelector('#device-tier');
      
      if (fpsElement) fpsElement.textContent = `FPS: ${stats.fps.toFixed(0)}`;
      if (viewElement) viewElement.textContent = `Views: ${stats.viewCount}`;
      if (fieldElement) fieldElement.textContent = `Field: ${stats.fieldSize}x${stats.fieldSize}`;
      if (tierElement) {
        const tier = (this.adaptiveRenderer as any).detectDeviceTier?.() || 'unknown';
        tierElement.textContent = `Tier: ${tier}`;
      }
      
      // Update debug panel if visible
      if (this.debugVisible) {
        this.updateDebugInfo();
      }
    }, 100);
  }
  
  /**
   * Update debug panel information
   */
  private updateDebugInfo(): void {
    // This would pull real-time data from the system
    // For now, showing placeholder structure
    
    // Get pose data if available
    const poseData = (window as any).__PARALLAX_DEBUG?.getPredictedPose?.();
    if (poseData) {
      const poseX = document.querySelector('#pose-x');
      const poseY = document.querySelector('#pose-y');
      const poseZ = document.querySelector('#pose-z');
      const poseRX = document.querySelector('#pose-rx');
      const poseRY = document.querySelector('#pose-ry');
      
      if (poseX) poseX.textContent = poseData.p[0].toFixed(3);
      if (poseY) poseY.textContent = poseData.p[1].toFixed(3);
      if (poseZ) poseZ.textContent = poseData.p[2].toFixed(3);
      if (poseRX) poseRX.textContent = (poseData.r[0] * 180 / Math.PI).toFixed(1);
      if (poseRY) poseRY.textContent = (poseData.r[1] * 180 / Math.PI).toFixed(1);
    }
  }
  
  /**
   * Update quality button states
   */
  private updateQualityButtons(): void {
    document.querySelectorAll('.quality-btn').forEach(btn => {
      const quality = btn.getAttribute('data-quality');
      if (quality === this.currentQuality) {
        (btn as HTMLElement).style.background = '#0f0';
        (btn as HTMLElement).style.color = '#000';
      } else {
        (btn as HTMLElement).style.background = '#000';
        (btn as HTMLElement).style.color = '#0f0';
      }
    });
  }
  
  // Public API Methods
  
  setMode(mode: 'quilt' | 'stereo' | 'hologram'): void {
    this.currentMode = mode;
    this.integrator.setRenderMode(mode);
    console.log(`[UI] Mode changed to: ${mode}`);
  }
  
  setQuality(quality: 'low' | 'medium' | 'high'): void {
    this.currentQuality = quality;
    
    // Map quality to resolution scale
    const scaleMap = {
      low: 0.5,
      medium: 1.0,
      high: 1.5
    };
    
    this.adaptiveRenderer.settings.resolutionScale = scaleMap[quality];
    console.log(`[UI] Quality changed to: ${quality}`);
  }
  
  toggleParallax(): void {
    this.parallaxEnabled = !this.parallaxEnabled;
    const toggle = document.querySelector('#parallax-toggle') as HTMLInputElement;
    if (toggle) toggle.checked = this.parallaxEnabled;
    
    document.dispatchEvent(new CustomEvent('parallax:toggle', { 
      detail: { enabled: this.parallaxEnabled }
    }));
    
    console.log(`[UI] Parallax: ${this.parallaxEnabled ? 'ON' : 'OFF'}`);
  }
  
  toggleDebug(): void {
    this.debugVisible = !this.debugVisible;
    if (this.debugPanel) {
      this.debugPanel.style.display = this.debugVisible ? 'block' : 'none';
    }
    
    const btn = document.querySelector('#debug-toggle') as HTMLButtonElement;
    if (btn) {
      btn.textContent = this.debugVisible ? 'Hide Debug [D]' : 'Show Debug [D]';
    }
  }
  
  cycleQuality(): void {
    const qualities: Array<'low' | 'medium' | 'high'> = ['low', 'medium', 'high'];
    const currentIndex = qualities.indexOf(this.currentQuality);
    const nextIndex = (currentIndex + 1) % qualities.length;
    this.setQuality(qualities[nextIndex]);
    this.updateQualityButtons();
  }
  
  cycleMode(): void {
    const modes: Array<'quilt' | 'stereo' | 'hologram'> = ['quilt', 'stereo', 'hologram'];
    const currentIndex = modes.indexOf(this.currentMode);
    const nextIndex = (currentIndex + 1) % modes.length;
    this.setMode(modes[nextIndex]);
    
    const select = document.querySelector('#mode-select') as HTMLSelectElement;
    if (select) select.value = modes[nextIndex];
  }
  
  resetView(): void {
    // Reset propagation distance
    this.integrator.setPropagationDistance(0.5);
    const slider = document.querySelector('#distance-slider') as HTMLInputElement;
    if (slider) slider.value = '50';
    
    // Reset quality to medium
    this.setQuality('medium');
    this.updateQualityButtons();
    
    console.log('[UI] View reset to defaults');
  }
  
  /**
   * Clean up UI elements
   */
  destroy(): void {
    this.controlPanel?.remove();
    this.statusBar?.remove();
    this.debugPanel?.remove();
  }
}
