<script lang="ts">
  import { onMount } from 'svelte';
  
  let deviceInfo = {
    loading: true,
    tier: '',
    caps: null as any,
    hw: '',
    ua: '',
    minSupported: '',
    ok: false,
    reason: ''
  };
  
  onMount(async () => {
    try {
      const response = await fetch('/device/matrix');
      const data = await response.json();
      deviceInfo = { ...data, loading: false };
    } catch (error) {
      deviceInfo = {
        loading: false,
        tier: 'ERROR',
        caps: null,
        hw: '',
        ua: navigator.userAgent,
        minSupported: '',
        ok: false,
        reason: 'Failed to fetch device matrix'
      };
    }
  });
  
  function getTierColor(tier: string) {
    switch(tier) {
      case 'GOLD': return '#FFD700';
      case 'SILVER': return '#C0C0C0';
      case 'UNSUPPORTED': return '#FF4444';
      default: return '#888';
    }
  }
</script>

<div class="container">
  <h1>IRIS Device Capability Check</h1>
  
  {#if deviceInfo.loading}
    <p>Checking device capabilities...</p>
  {:else}
    <div class="status" style="background-color: {deviceInfo.ok ? '#4CAF50' : '#FF4444'}">
      {deviceInfo.ok ? '✅ Device Supported' : '❌ Device Not Supported'}
    </div>
    
    <div class="info-grid">
      <div class="info-row">
        <span class="label">Device Tier:</span>
        <span class="value" style="color: {getTierColor(deviceInfo.tier)}">{deviceInfo.tier}</span>
      </div>
      
      <div class="info-row">
        <span class="label">Hardware Model:</span>
        <span class="value">{deviceInfo.hw || '<unknown>'}</span>
      </div>
      
      <div class="info-row">
        <span class="label">Min Supported:</span>
        <span class="value">{deviceInfo.minSupported}</span>
      </div>
      
      {#if deviceInfo.caps}
        <div class="info-row">
          <span class="label">Max N:</span>
          <span class="value">{deviceInfo.caps.maxN}</span>
        </div>
        
        <div class="info-row">
          <span class="label">Zernike Modes:</span>
          <span class="value">{deviceInfo.caps.zernikeModes}</span>
        </div>
        
        <div class="info-row">
          <span class="label">Server Assist:</span>
          <span class="value">{deviceInfo.caps.serverFallback ? 'Available' : 'Not Available'}</span>
        </div>
      {/if}
      
      {#if deviceInfo.reason}
        <div class="info-row">
          <span class="label">Detection Method:</span>
          <span class="value">{deviceInfo.reason}</span>
        </div>
      {/if}
      
      <div class="info-row full-width">
        <span class="label">User Agent:</span>
        <span class="value small">{deviceInfo.ua}</span>
      </div>
    </div>
    
    {#if deviceInfo.tier === 'UNSUPPORTED'}
      <div class="warning">
        <p>⚠️ This device does not meet the minimum requirements for IRIS.</p>
        <p>iPhone 13 or newer (or equivalent iPad) is required.</p>
      </div>
    {/if}
  {/if}
</div>

<style>
  .container {
    max-width: 800px;
    margin: 2rem auto;
    padding: 2rem;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
  }
  
  h1 {
    text-align: center;
    color: #333;
  }
  
  .status {
    text-align: center;
    padding: 1rem;
    margin: 2rem 0;
    border-radius: 8px;
    color: white;
    font-size: 1.5rem;
    font-weight: bold;
  }
  
  .info-grid {
    background: #f5f5f5;
    padding: 1.5rem;
    border-radius: 8px;
    margin: 2rem 0;
  }
  
  .info-row {
    display: flex;
    justify-content: space-between;
    padding: 0.75rem 0;
    border-bottom: 1px solid #ddd;
  }
  
  .info-row:last-child {
    border-bottom: none;
  }
  
  .info-row.full-width {
    flex-direction: column;
    gap: 0.5rem;
  }
  
  .label {
    font-weight: 600;
    color: #666;
  }
  
  .value {
    color: #333;
    font-family: 'Courier New', monospace;
  }
  
  .value.small {
    font-size: 0.9rem;
    word-break: break-all;
  }
  
  .warning {
    background: #FFF3CD;
    border: 1px solid #FFC107;
    border-radius: 8px;
    padding: 1rem;
    margin-top: 2rem;
  }
  
  .warning p {
    margin: 0.5rem 0;
    color: #856404;
  }
</style>