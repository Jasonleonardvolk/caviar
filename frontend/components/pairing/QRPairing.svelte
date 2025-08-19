<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import QRCode from 'qrcode';
  import { generatePairingJWT } from '../../lib/auth/pairing';
  
  export let onPaired: (sessionId: string) => void = () => {};
  export let capabilities: string[] = ['render', 'stream', 'metrics'];
  export let expiryMinutes: number = 10;
  
  let qrCanvas: HTMLCanvasElement;
  let pairingUrl: string = '';
  let jwt: string = '';
  let sessionId: string = '';
  let status: 'waiting' | 'connected' | 'expired' = 'waiting';
  let timeRemaining: number = expiryMinutes * 60;
  
  let pollInterval: NodeJS.Timeout;
  let countdownInterval: NodeJS.Timeout;
  
  onMount(async () => {
    await generatePairingCode();
    startPolling();
    startCountdown();
  });
  
  onDestroy(() => {
    if (pollInterval) clearInterval(pollInterval);
    if (countdownInterval) clearInterval(countdownInterval);
  });
  
  async function generatePairingCode() {
    try {
      // Generate pairing JWT with requested capabilities
      const response = await fetch('/api/pairing/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          capabilities,
          expiryMinutes
        })
      });
      
      const data = await response.json();
      jwt = data.jwt;
      sessionId = data.sessionId;
      
      // Create deep link URL
      pairingUrl = `tori-holo://pair?jwt=${jwt}&session=${sessionId}`;
      
      // Generate QR code
      await QRCode.toCanvas(qrCanvas, pairingUrl, {
        width: 300,
        margin: 2,
        color: {
          dark: '#000000',
          light: '#FFFFFF'
        },
        errorCorrectionLevel: 'M'
      });
      
    } catch (error) {
      console.error('Failed to generate pairing code:', error);
    }
  }
  
  function startPolling() {
    // Poll for pairing status
    pollInterval = setInterval(async () => {
      try {
        const response = await fetch(`/api/pairing/status/${sessionId}`);
        const data = await response.json();
        
        if (data.status === 'connected') {
          status = 'connected';
          clearInterval(pollInterval);
          clearInterval(countdownInterval);
          onPaired(sessionId);
        }
      } catch (error) {
        console.error('Polling error:', error);
      }
    }, 2000); // Poll every 2 seconds
  }
  
  function startCountdown() {
    countdownInterval = setInterval(() => {
      timeRemaining--;
      
      if (timeRemaining <= 0) {
        status = 'expired';
        clearInterval(countdownInterval);
        clearInterval(pollInterval);
      }
    }, 1000);
  }
  
  function formatTime(seconds: number): string {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  }
  
  async function regenerate() {
    status = 'waiting';
    timeRemaining = expiryMinutes * 60;
    await generatePairingCode();
    startPolling();
    startCountdown();
  }
</script>

<div class="pairing-container">
  <div class="pairing-card">
    <h2>Mobile App Pairing</h2>
    
    {#if status === 'waiting'}
      <div class="qr-section">
        <canvas bind:this={qrCanvas} class="qr-code"></canvas>
        <p class="instructions">
          Scan this QR code with the TORI Hologram app to connect
        </p>
        <div class="timer">
          Expires in: <span class="time">{formatTime(timeRemaining)}</span>
        </div>
      </div>
      
      <div class="capabilities">
        <h4>Granted Capabilities:</h4>
        <ul>
          {#each capabilities as cap}
            <li class="capability-badge {cap}">
              {cap === 'render' ? '🎨' : cap === 'stream' ? '📡' : '📊'} {cap}
            </li>
          {/each}
        </ul>
      </div>
      
      <div class="manual-option">
        <details>
          <summary>Manual pairing code</summary>
          <code class="pairing-code">{sessionId.substring(0, 8)}</code>
        </details>
      </div>
    {/if}
    
    {#if status === 'connected'}
      <div class="success">
        <div class="success-icon">✅</div>
        <h3>Successfully Connected!</h3>
        <p>Your mobile device is now paired with this hologram session.</p>
      </div>
    {/if}
    
    {#if status === 'expired'}
      <div class="expired">
        <div class="expired-icon">⏰</div>
        <h3>Pairing Code Expired</h3>
        <p>The pairing code has expired for security reasons.</p>
        <button on:click={regenerate} class="regenerate-btn">
          Generate New Code
        </button>
      </div>
    {/if}
  </div>
</div>

<style>
  .pairing-container {
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 100vh;
    background: linear-gradient(135deg, #1a1a2e 0%, #0f0f1e 100%);
    color: #ffffff;
  }
  
  .pairing-card {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.1);
    max-width: 400px;
    width: 100%;
  }
  
  h2 {
    text-align: center;
    margin-bottom: 2rem;
    font-size: 1.8rem;
    background: linear-gradient(45deg, #00ffff, #ff00ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  
  .qr-section {
    text-align: center;
    margin-bottom: 2rem;
  }
  
  .qr-code {
    background: white;
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
  }
  
  .instructions {
    color: #aaa;
    font-size: 0.9rem;
    margin-bottom: 1rem;
  }
  
  .timer {
    font-size: 0.85rem;
    color: #888;
  }
  
  .time {
    color: #00ffff;
    font-weight: bold;
    font-family: monospace;
  }
  
  .capabilities {
    margin-bottom: 1.5rem;
  }
  
  .capabilities h4 {
    font-size: 0.9rem;
    color: #aaa;
    margin-bottom: 0.5rem;
  }
  
  .capabilities ul {
    list-style: none;
    padding: 0;
    display: flex;
    gap: 0.5rem;
    justify-content: center;
  }
  
  .capability-badge {
    background: rgba(255, 255, 255, 0.1);
    padding: 0.25rem 0.75rem;
    border-radius: 15px;
    font-size: 0.85rem;
    border: 1px solid rgba(255, 255, 255, 0.2);
  }
  
  .capability-badge.render {
    border-color: #00ff88;
    color: #00ff88;
  }
  
  .capability-badge.stream {
    border-color: #00ffff;
    color: #00ffff;
  }
  
  .capability-badge.metrics {
    border-color: #ff00ff;
    color: #ff00ff;
  }
  
  .manual-option {
    text-align: center;
  }
  
  .manual-option summary {
    cursor: pointer;
    color: #888;
    font-size: 0.85rem;
  }
  
  .pairing-code {
    display: block;
    margin-top: 0.5rem;
    padding: 0.5rem;
    background: rgba(0, 0, 0, 0.3);
    border-radius: 5px;
    font-size: 1.2rem;
    letter-spacing: 0.1em;
  }
  
  .success, .expired {
    text-align: center;
    padding: 2rem;
  }
  
  .success-icon, .expired-icon {
    font-size: 4rem;
    margin-bottom: 1rem;
  }
  
  .success h3 {
    color: #00ff88;
    margin-bottom: 0.5rem;
  }
  
  .expired h3 {
    color: #ff4444;
    margin-bottom: 0.5rem;
  }
  
  .regenerate-btn {
    margin-top: 1rem;
    padding: 0.75rem 1.5rem;
    background: linear-gradient(45deg, #00ffff, #ff00ff);
    border: none;
    border-radius: 25px;
    color: white;
    font-weight: bold;
    cursor: pointer;
    transition: transform 0.2s;
  }
  
  .regenerate-btn:hover {
    transform: scale(1.05);
  }
</style>
