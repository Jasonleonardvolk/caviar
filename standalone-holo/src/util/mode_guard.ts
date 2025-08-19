// mode_guard.ts - Runtime mode resolution with automatic fallback
export async function resolveMode(cfg: { mode: string; calibration?: string }): Promise<'standard' | 'lenticular'> {
  // Default to standard (no hardware required)
  if (cfg.mode !== 'lenticular') return 'standard';
  
  // Only use lenticular if explicitly requested AND calibration is accessible
  try {
    const r = await fetch(cfg.calibration!, { method: 'HEAD' });
    if (!r.ok) throw new Error('Calibration not found');
    
    // Additional check: See if WebGPU supports the required features
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) {
      console.warn('[Mode Guard] WebGPU not available for lenticular mode');
      throw new Error('WebGPU required');
    }
    
    console.log('[Mode Guard] Lenticular mode confirmed - panel appears to be connected');
    return 'lenticular';
  } catch (err) {
    console.warn('[Mode Guard] Cannot use lenticular mode:', err);
    console.warn('[Mode Guard] Falling back to standard mode (no hardware required)');
    return 'standard';
  }
}

export function getModeBadgeHTML(mode: string): string {
  const isStandard = mode === 'standard';
  return `
    <div style="
      position: fixed;
      top: 8px;
      left: 8px;
      padding: 8px 12px;
      background: ${isStandard ? 'rgba(0,50,0,0.9)' : 'rgba(50,0,50,0.9)'};
      border: 1px solid ${isStandard ? '#0f0' : '#f0f'};
      border-radius: 4px;
      color: ${isStandard ? '#0f0' : '#f0f'};
      font-family: monospace;
      font-size: 12px;
      z-index: 10000;
    ">
      <div style="font-weight: bold; margin-bottom: 4px;">
        MODE: ${mode.toUpperCase()}
      </div>
      <div style="font-size: 10px; opacity: 0.8;">
        ${isStandard 
          ? '✓ NO HARDWARE REQUIRED<br>✓ Runs on ANY screen<br>✓ Head-tracked parallax' 
          : '⚡ Lenticular panel active<br>⚡ Multi-view rendering<br>⚡ No Bridge required'}
      </div>
    </div>
  `;
}

export function showModeNotification(message: string, duration: number = 3000) {
  const notif = document.createElement('div');
  notif.style.cssText = `
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    padding: 20px 30px;
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 2px solid #0f3460;
    border-radius: 10px;
    color: #fff;
    font-family: system-ui, -apple-system, sans-serif;
    font-size: 14px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.5);
    z-index: 99999;
    animation: slideIn 0.3s ease;
  `;
  
  notif.innerHTML = `
    <div style="display: flex; align-items: center; gap: 12px;">
      <div style="font-size: 24px;">ℹ️</div>
      <div>${message}</div>
    </div>
  `;
  
  // Add animation
  const style = document.createElement('style');
  style.textContent = `
    @keyframes slideIn {
      from { opacity: 0; transform: translate(-50%, -60%); }
      to { opacity: 1; transform: translate(-50%, -50%); }
    }
  `;
  document.head.appendChild(style);
  
  document.body.appendChild(notif);
  setTimeout(() => {
    notif.style.transition = 'opacity 0.3s ease';
    notif.style.opacity = '0';
    setTimeout(() => {
      notif.remove();
      style.remove();
    }, 300);
  }, duration);
}