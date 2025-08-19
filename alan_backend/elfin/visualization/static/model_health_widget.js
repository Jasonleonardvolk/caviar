/**
 * Model Health Widget for ELFIN Dashboard
 * 
 * This widget visualizes health metrics for Koopman models, including:
 * - Cross-validation metrics (train/val MSE)
 * - Eigenvalue drift visualization
 * - λ-cut and weighting adjustment controls
 */

class ModelHealthWidget {
  constructor(container, options = {}) {
    this.container = typeof container === 'string' 
      ? document.getElementById(container) 
      : container;
    
    this.options = Object.assign({
      width: 600,
      height: 400,
      systemName: 'pendulum',
      defaultCut: 0.98,
      defaultWeighting: 'uniform',
      refreshInterval: 5000  // ms
    }, options);
    
    this.data = null;
    this.cvData = null;
    
    this.init();
  }
  
  init() {
    this.createDOM();
    this.setupEventListeners();
    this.loadData();
    
    // Set up periodic refresh
    if (this.options.refreshInterval > 0) {
      this.refreshInterval = setInterval(() => {
        this.loadData();
      }, this.options.refreshInterval);
    }
    
    // Debouncing for performance
    this.debounceTimers = {};
  }
  
  // Debounce function to limit update frequency
  debounce(func, wait, id) {
    // Cancel previous timer for this ID
    if (this.debounceTimers[id]) {
      clearTimeout(this.debounceTimers[id]);
    }
    
    // Set new timer
    this.debounceTimers[id] = setTimeout(() => {
      func();
      this.debounceTimers[id] = null;
    }, wait);
  }
  
  createDOM() {
    // Clear container
    this.container.innerHTML = '';
    this.container.classList.add('model-health-widget');
    
    // Create header
    const header = document.createElement('div');
    header.className = 'widget-header';
    header.innerHTML = `
      <h3>Model Health: <span class="system-name">${this.options.systemName}</span></h3>
      <div class="widget-controls">
        <select id="system-selector">
          <option value="pendulum">Pendulum</option>
          <option value="vdp">Van der Pol</option>
        </select>
        <button id="run-cv-btn" class="btn btn-sm btn-primary">Run CV</button>
        <button id="refresh-btn" class="btn btn-sm btn-secondary">Refresh</button>
      </div>
    `;
    this.container.appendChild(header);
    
    // Create content area
    const content = document.createElement('div');
    content.className = 'widget-content';
    
    // Create metrics section
    const metrics = document.createElement('div');
    metrics.className = 'metrics-section';
    metrics.innerHTML = `
      <div class="metric-card">
        <h4>Training MSE</h4>
        <div class="metric-value" id="train-mse">-</div>
      </div>
      <div class="metric-card">
        <h4>Validation MSE</h4>
        <div class="metric-value" id="val-mse">-</div>
      </div>
      <div class="metric-card">
        <h4>Eigenvalue Drift</h4>
        <div class="metric-value" id="eigen-drift">-</div>
      </div>
      <div class="metric-card">
        <h4>Stable Modes</h4>
        <div class="metric-value" id="stable-modes">-</div>
      </div>
    `;
    content.appendChild(metrics);
    
    // Create visualization section
    const viz = document.createElement('div');
    viz.className = 'viz-section';
    viz.innerHTML = `
      <div class="viz-container">
        <div id="cv-chart" style="height: 200px;"></div>
      </div>
    `;
    content.appendChild(viz);
    
    // Create parameter tuning section
    const params = document.createElement('div');
    params.className = 'params-section';
    params.innerHTML = `
      <div class="param-control">
        <label>λ-cut: <span id="lambda-cut-value">${this.options.defaultCut}</span></label>
        <input type="range" id="lambda-cut" min="0.8" max="0.99" step="0.01" value="${this.options.defaultCut}">
      </div>
      <div class="param-control">
        <label>Weighting Strategy:</label>
        <div class="radio-group">
          <label>
            <input type="radio" name="weighting" value="uniform" ${this.options.defaultWeighting === 'uniform' ? 'checked' : ''}>
            Uniform
          </label>
          <label>
            <input type="radio" name="weighting" value="lambda" ${this.options.defaultWeighting === 'lambda' ? 'checked' : ''}>
            λ-weighted
          </label>
        </div>
      </div>
      <button id="apply-params" class="btn btn-primary">Apply Parameters</button>
    `;
    content.appendChild(params);
    
    // Create status section
    const status = document.createElement('div');
    status.className = 'status-section';
    status.innerHTML = `
      <div id="status-message"></div>
      <div class="last-updated">Last updated: <span id="last-updated">Never</span></div>
    `;
    content.appendChild(status);
    
    this.container.appendChild(content);
    
    // Add styles
    const style = document.createElement('style');
    style.textContent = `
      .model-health-widget {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background-color: #f5f8fa;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        padding: 16px;
        max-width: ${this.options.width}px;
      }
      .widget-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 1px solid #e1e4e8;
      }
      .widget-header h3 {
        margin: 0;
        font-size: 18px;
        color: #24292e;
      }
      .metrics-section {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-bottom: 16px;
      }
      .metric-card {
        background-color: white;
        border-radius: 6px;
        padding: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        text-align: center;
      }
      .metric-card h4 {
        margin: 0 0 8px 0;
        font-size: 14px;
        color: #586069;
      }
      .metric-value {
        font-size: 20px;
        font-weight: 600;
        color: #24292e;
      }
      .viz-section {
        margin-bottom: 16px;
      }
      .params-section {
        background-color: white;
        border-radius: 6px;
        padding: 16px;
        margin-bottom: 16px;
      }
      .param-control {
        margin-bottom: 12px;
      }
      .param-control label {
        display: block;
        margin-bottom: 4px;
        font-weight: 500;
      }
      .param-control input[type="range"] {
        width: 100%;
      }
      .radio-group {
        display: flex;
        gap: 16px;
      }
      .status-section {
        display: flex;
        justify-content: space-between;
        font-size: 12px;
        color: #586069;
      }
      #status-message {
        color: #0366d6;
      }
      #status-message.error {
        color: #d73a49;
      }
    `;
    document.head.appendChild(style);
  }
  
  setupEventListeners() {
    // System selector
    const systemSelector = document.getElementById('system-selector');
    systemSelector.value = this.options.systemName;
    systemSelector.addEventListener('change', () => {
      this.options.systemName = systemSelector.value;
      this.loadData();
    });
    
    // Run CV button
    const runCVBtn = document.getElementById('run-cv-btn');
    runCVBtn.addEventListener('click', () => {
      this.runCrossValidation();
    });
    
    // Refresh button
    const refreshBtn = document.getElementById('refresh-btn');
    refreshBtn.addEventListener('click', () => {
      this.loadData();
    });
    
    // Lambda cut slider
    const lambdaCutSlider = document.getElementById('lambda-cut');
    const lambdaCutValue = document.getElementById('lambda-cut-value');
    lambdaCutSlider.addEventListener('input', () => {
      lambdaCutValue.textContent = lambdaCutSlider.value;
    });
    
    // Apply parameters button
    const applyBtn = document.getElementById('apply-params');
    applyBtn.addEventListener('click', () => {
      this.applyParameters();
    });
  }
  
  loadData() {
    this.updateStatus('Loading model data...');
    
    // Load basic model info
    fetch(`/api/koopman/info?system=${this.options.systemName}`)
      .then(response => response.json())
      .then(data => {
        this.data = data;
        this.updateModelInfo();
        this.updateLastUpdated();
        this.updateStatus('Model data loaded successfully');
      })
      .catch(error => {
        console.error('Error loading model data:', error);
        this.updateStatus('Failed to load model data', true);
      });
    
    // Load CV data if available
    fetch(`/api/koopman/cv/latest?system=${this.options.systemName}`)
      .then(response => response.json())
      .then(data => {
        if (data && data.status === 'success') {
          this.cvData = data;
          this.updateCVMetrics();
          this.renderCVChart();
        }
      })
      .catch(error => {
        console.error('Error loading CV data:', error);
      });
  }
  
  updateModelInfo() {
    if (!this.data) return;
    
    // Update stable modes count
    const stableModesEl = document.getElementById('stable-modes');
    if (this.data.n_stable_modes !== undefined) {
      stableModesEl.textContent = `${this.data.n_stable_modes} / ${this.data.n_total_modes}`;
    }
  }
  
  updateCVMetrics() {
    if (!this.cvData) return;
    
    // Update metrics
    const trainMSE = document.getElementById('train-mse');
    const valMSE = document.getElementById('val-mse');
    const eigenDrift = document.getElementById('eigen-drift');
    
    trainMSE.textContent = this.cvData.train_mse.mean.toExponential(3);
    valMSE.textContent = this.cvData.val_mse.mean.toExponential(3);
    
    if (this.cvData.eigenvalues_drift) {
      eigenDrift.textContent = this.cvData.eigenvalues_drift.mean.toExponential(3);
    }
  }
  
  renderCVChart() {
    if (!this.cvData) return;
    
    // Use debouncing to limit rendering to 2 Hz for performance
    this.debounce(() => {
      // Simple chart using DOM (could be replaced with a proper charting library)
      const chartContainer = document.getElementById('cv-chart');
      chartContainer.innerHTML = '';
      
      this._renderChartContent(chartContainer);
    }, 500, 'chart-render');
  }
  
  _renderChartContent(chartContainer) {
    // Create canvas for sparkline
    const canvas = document.createElement('canvas');
    canvas.width = chartContainer.clientWidth;
    canvas.height = chartContainer.clientHeight;
    chartContainer.appendChild(canvas);
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // Draw simple sparkline of eigenvalue drift
    if (this.cvData.eigenvalues_drift && this.cvData.eigenvalues_drift.history) {
      const history = this.cvData.eigenvalues_drift.history;
      const max = Math.max(...history) * 1.1;
      
      ctx.strokeStyle = '#0366d6';
      ctx.lineWidth = 2;
      ctx.beginPath();
      
      for (let i = 0; i < history.length; i++) {
        const x = (i / (history.length - 1)) * width;
        const y = height - (history[i] / max) * height;
        
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      
      ctx.stroke();
      
      // Add labels
      ctx.fillStyle = '#586069';
      ctx.font = '12px sans-serif';
      ctx.fillText('Eigenvalue Drift History', 10, 20);
      ctx.fillText(`Max: ${max.toExponential(2)}`, width - 100, 20);
    } else {
      ctx.fillStyle = '#586069';
      ctx.font = '14px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('No eigenvalue drift history available', width / 2, height / 2);
    }
  }
  
  runCrossValidation() {
    this.updateStatus('Running cross-validation...');
    
    fetch('/api/koopman/cv', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        system: this.options.systemName,
        splits: 5
      })
    })
      .then(response => response.json())
      .then(data => {
        if (data.status === 'success') {
          this.cvData = data;
          this.updateCVMetrics();
          this.renderCVChart();
          this.updateLastUpdated();
          this.updateStatus('Cross-validation completed successfully');
        } else {
          this.updateStatus(`Cross-validation failed: ${data.message}`, true);
        }
      })
      .catch(error => {
        console.error('Error running cross-validation:', error);
        this.updateStatus('Failed to run cross-validation', true);
      });
  }
  
  applyParameters() {
    const lambdaCut = document.getElementById('lambda-cut').value;
    const weightingEls = document.querySelectorAll('input[name="weighting"]');
    let weighting = 'uniform';
    
    for (const el of weightingEls) {
      if (el.checked) {
        weighting = el.value;
        break;
      }
    }
    
    this.updateStatus(`Applying parameters: λ-cut=${lambdaCut}, weighting=${weighting}...`);
    
    fetch('/api/koopman/weighting', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        system: this.options.systemName,
        cut: parseFloat(lambdaCut),
        weighting: weighting
      })
    })
      .then(response => response.json())
      .then(data => {
        if (data.status === 'success') {
          this.data = data;
          this.updateModelInfo();
          this.updateLastUpdated();
          this.updateStatus('Parameters applied successfully');
        } else {
          this.updateStatus(`Failed to apply parameters: ${data.message}`, true);
        }
      })
      .catch(error => {
        console.error('Error applying parameters:', error);
        this.updateStatus('Failed to apply parameters', true);
      });
  }
  
  updateStatus(message, isError = false) {
    const statusEl = document.getElementById('status-message');
    statusEl.textContent = message;
    
    if (isError) {
      statusEl.classList.add('error');
    } else {
      statusEl.classList.remove('error');
    }
  }
  
  updateLastUpdated() {
    const lastUpdatedEl = document.getElementById('last-updated');
    const now = new Date();
    lastUpdatedEl.textContent = now.toLocaleTimeString();
  }
  
  destroy() {
    if (this.refreshInterval) {
      clearInterval(this.refreshInterval);
    }
  }
}

// Register the widget with the dashboard
document.addEventListener('DOMContentLoaded', () => {
  // Check if we're in the dashboard
  if (window.ELFINDashboard) {
    window.ELFINDashboard.registerWidget('model-health', (container, options) => {
      return new ModelHealthWidget(container, options);
    });
  }
});
