<script lang="ts">
  import { onMount } from 'svelte';
  
  let health: any = null;
  let analytics: any = null;
  let loading = true;
  let error = '';
  let autoRefresh = true;
  let refreshInterval: number | null = null;
  
  async function loadData() {
    try {
      // Load health status
      const healthRes = await fetch('/health', { cache: 'no-store' });
      health = await healthRes.json();
      
      // Load WOW Pack analytics
      const analyticsRes = await fetch('/api/wowpack/analytics', { cache: 'no-store' });
      analytics = await analyticsRes.json();
      
      loading = false;
    } catch (e: any) {
      error = e.message || 'Failed to load dashboard data';
      loading = false;
    }
  }
  
  function toggleAutoRefresh() {
    autoRefresh = !autoRefresh;
    if (autoRefresh) {
      startAutoRefresh();
    } else {
      stopAutoRefresh();
    }
  }
  
  function startAutoRefresh() {
    refreshInterval = window.setInterval(loadData, 5000);
  }
  
  function stopAutoRefresh() {
    if (refreshInterval) {
      clearInterval(refreshInterval);
      refreshInterval = null;
    }
  }
  
  onMount(() => {
    loadData();
    if (autoRefresh) startAutoRefresh();
    
    return () => {
      stopAutoRefresh();
    };
  });
  
  function formatSize(gb: number): string {
    if (gb < 1) return `${(gb * 1024).toFixed(1)} MB`;
    return `${gb.toFixed(2)} GB`;
  }
  
  function getStatusColor(status: boolean): string {
    return status ? '#00ffa3' : '#ff6b6b';
  }
</script>

<style>
  .dashboard {
    min-height: 100vh;
    background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
    color: #fff;
    padding: 2rem;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  }
  
  .header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 2rem;
    padding: 1rem 1.5rem;
    background: rgba(255,255,255,0.03);
    border-radius: 16px;
    backdrop-filter: blur(10px);
  }
  
  .title {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 1.5rem;
    margin-bottom: 2rem;
  }
  
  .card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 1.5rem;
    backdrop-filter: blur(10px);
    transition: all 0.3s;
  }
  
  .card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
  }
  
  .card-title {
    font-size: 0.9rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 1rem;
  }
  
  .metric {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
  }
  
  .metric-label {
    font-size: 0.85rem;
    color: #aaa;
  }
  
  .status-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin-top: 1rem;
  }
  
  .status-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.75rem;
    background: rgba(0,0,0,0.3);
    border-radius: 8px;
  }
  
  .status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
  }
  
  .progress-bar {
    height: 6px;
    background: rgba(255,255,255,0.1);
    border-radius: 3px;
    overflow: hidden;
    margin-top: 0.5rem;
  }
  
  .progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #4ac7ff, #00ffa3);
    transition: width 0.5s;
  }
  
  .button {
    padding: 0.5rem 1rem;
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 8px;
    color: white;
    cursor: pointer;
    transition: all 0.2s;
  }
  
  .button:hover {
    background: rgba(255,255,255,0.15);
  }
  
  .button.active {
    background: linear-gradient(135deg, #667eea, #764ba2);
    border: none;
  }
  
  .recommendations {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 1.5rem;
  }
  
  .rec-item {
    padding: 0.75rem;
    margin: 0.5rem 0;
    background: rgba(0,0,0,0.3);
    border-radius: 8px;
    border-left: 3px solid #667eea;
  }
  
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }
  
  .live-indicator {
    display: inline-block;
    width: 8px;
    height: 8px;
    background: #00ffa3;
    border-radius: 50%;
    margin-left: 0.5rem;
    animation: pulse 2s infinite;
  }
  
  .loading {
    text-align: center;
    padding: 4rem;
    font-size: 1.2rem;
    color: #888;
  }
  
  .error-box {
    background: rgba(255,0,0,0.1);
    border: 1px solid rgba(255,0,0,0.3);
    border-radius: 12px;
    padding: 1rem;
    margin: 1rem 0;
    color: #ff6b6b;
  }
</style>

<div class="dashboard">
  <div class="header">
    <h1 class="title">
      🚀 CAVIAR/KHA Launch Dashboard
      {#if autoRefresh}<span class="live-indicator"></span>{/if}
    </h1>
    <div>
      <button class="button" class:active={autoRefresh} on:click={toggleAutoRefresh}>
        {autoRefresh ? '⏸ Pause' : '▶ Auto Refresh'}
      </button>
      <button class="button" on:click={loadData}>🔄 Refresh</button>
    </div>
  </div>
  
  {#if loading}
    <div class="loading">Loading dashboard metrics...</div>
  {:else if error}
    <div class="error-box">{error}</div>
  {:else}
    <!-- Core Health Metrics -->
    <div class="grid">
      <div class="card">
        <div class="card-title">System Health</div>
        <div class="metric" style="color: {health?.ok ? '#00ffa3' : '#ff6b6b'}">
          {health?.ok ? 'HEALTHY' : 'DEGRADED'}
        </div>
        <div class="metric-label">
          {health?.files?.presentCount || 0}/{health?.files?.requiredTotal || 0} files ready
        </div>
        <div class="progress-bar">
          <div class="progress-fill" style="width: {((health?.files?.presentCount || 0) / (health?.files?.requiredTotal || 1)) * 100}%"></div>
        </div>
      </div>
      
      <div class="card">
        <div class="card-title">Production Readiness</div>
        <div class="metric" style="color: {health?.readiness?.demo ? '#00ffa3' : '#ff6b6b'}">
          {health?.readiness?.demo ? 'READY' : 'NOT READY'}
        </div>
        <div class="status-grid">
          <div class="status-item">
            <div class="status-dot" style="background: {health?.readiness?.demo ? '#00ffa3' : '#ff6b6b'}"></div>
            <span>Demo Mode</span>
          </div>
          <div class="status-item">
            <div class="status-dot" style="background: {health?.environment?.stripeConfigured ? '#00ffa3' : '#ff6b6b'}"></div>
            <span>Stripe</span>
          </div>
        </div>
      </div>
      
      <div class="card">
        <div class="card-title">Server Status</div>
        <div class="metric" style="color: #00ffa3">ONLINE</div>
        <div class="metric-label">Port 5173</div>
        <div class="status-grid">
          <div class="status-item">
            <div class="status-dot" style="background: #00ffa3"></div>
            <span>API Active</span>
          </div>
          <div class="status-item">
            <div class="status-dot" style="background: #00ffa3"></div>
            <span>Health OK</span>
          </div>
        </div>
      </div>
    </div>
    
    <!-- WOW Pack Analytics -->
    {#if analytics}
      <h2 style="margin: 2rem 0 1rem; font-size: 1.5rem;">🎬 WOW Pack Pipeline</h2>
      
      <div class="grid">
        <div class="card">
          <div class="card-title">ProRes Masters</div>
          <div class="metric" style="color: {analytics.storage.masters.count > 0 ? '#00ffa3' : '#ff6b6b'}">
            {analytics.storage.masters.count}
          </div>
          <div class="metric-label">
            {formatSize(analytics.storage.masters.totalSizeGB)} total
          </div>
          {#if analytics.storage.masters.files?.length > 0}
            <div style="margin-top: 1rem; font-size: 0.85rem; opacity: 0.7;">
              {#each analytics.storage.masters.files as file}
                <div>• {file.filename}</div>
              {/each}
            </div>
          {/if}
        </div>
        
        <div class="card">
          <div class="card-title">Encoded Outputs</div>
          <div class="metric" style="color: #4ac7ff">
            {analytics.storage.av1.count + analytics.storage.hdr10.count + analytics.storage.sdr.count}
          </div>
          <div class="metric-label">Total encodes</div>
          <div class="status-grid" style="margin-top: 1rem;">
            <div class="status-item">
              <span>AV1:</span>
              <span style="color: #00ffa3; margin-left: auto;">{analytics.storage.av1.count}</span>
            </div>
            <div class="status-item">
              <span>HDR10:</span>
              <span style="color: #00ffa3; margin-left: auto;">{analytics.storage.hdr10.count}</span>
            </div>
            <div class="status-item">
              <span>SDR:</span>
              <span style="color: #00ffa3; margin-left: auto;">{analytics.storage.sdr.count}</span>
            </div>
          </div>
        </div>
        
        <div class="card">
          <div class="card-title">Pipeline Status</div>
          <div class="metric" style="color: {analytics.pipeline.ffmpegInstalled ? '#00ffa3' : '#ff6b6b'}">
            {analytics.pipeline.ffmpegInstalled ? 'READY' : 'MISSING'}
          </div>
          <div class="metric-label">FFmpeg {analytics.pipeline.ffmpegInstalled ? 'installed' : 'not found'}</div>
          <div class="status-grid" style="margin-top: 1rem;">
            <div class="status-item">
              <span>Scripts:</span>
              <span style="color: #00ffa3; margin-left: auto;">{analytics.pipeline.encodingScriptsCount}</span>
            </div>
            <div class="status-item">
              <span>Videos:</span>
              <span style="color: #00ffa3; margin-left: auto;">{analytics.pipeline.totalVideos}</span>
            </div>
          </div>
        </div>
        
        <div class="card">
          <div class="card-title">Storage Usage</div>
          <div class="metric" style="color: #667eea">
            {formatSize(analytics.pipeline.totalSizeGB)}
          </div>
          <div class="metric-label">Total content size</div>
          <div class="progress-bar">
            <div class="progress-fill" style="width: {Math.min(100, (analytics.pipeline.totalSizeGB / 10) * 100)}%"></div>
          </div>
        </div>
      </div>
      
      <!-- Recommendations -->
      {#if analytics.recommendations?.length > 0}
        <div class="recommendations">
          <h3 style="margin-bottom: 1rem;">📋 Recommendations</h3>
          {#each analytics.recommendations as rec}
            <div class="rec-item">{rec}</div>
          {/each}
        </div>
      {/if}
    {/if}
    
    <!-- Quick Actions -->
    <div style="margin-top: 2rem; padding: 1.5rem; background: rgba(255,255,255,0.03); border-radius: 16px;">
      <h3 style="margin-bottom: 1rem;">🎯 Quick Actions</h3>
      <div style="display: flex; gap: 1rem; flex-wrap: wrap;">
        <a href="/hologram" class="button" style="text-decoration: none;">View Hologram</a>
        <a href="/templates" class="button" style="text-decoration: none;">Templates</a>
        <a href="/publish" class="button" style="text-decoration: none;">Publishing</a>
        <a href="/pricing" class="button" style="text-decoration: none;">Pricing</a>
        <a href="/account/manage" class="button" style="text-decoration: none;">Billing</a>
      </div>
    </div>
  {/if}
</div>