<script>
  import { onMount } from 'svelte';
  
  let tiktokFiles = [];
  let snapFiles = [];
  let serverUrl = '';
  
  onMount(async () => {
    // Get server URL for QR codes
    serverUrl = window.location.origin;
    
    // Fetch actual files from server
    try {
      const response = await fetch('/social/files');
      const data = await response.json();
      tiktokFiles = data.tiktok || [];
      snapFiles = data.snap || [];
    } catch (error) {
      console.error('Failed to load files:', error);
      // Fallback examples
      tiktokFiles = [];
      snapFiles = [];
    }
  });
  
  function getQRUrl(platform, file) {
    return `https://api.qrserver.com/v1/create-qr-code/?size=150x150&data=${encodeURIComponent(serverUrl + '/social/' + platform + '/' + file)}`;
  }
</script>

<div class="container">
  <h1>📱 Social Pack - Device Testing</h1>
  
  <p class="info">
    Scan QR codes on your phone to test exports directly<br>
    Server: {serverUrl || 'Loading...'}
  </p>
  
  <div class="platform-section">
    <h2>🎵 TikTok Exports</h2>
    <div class="file-grid">
      {#each tiktokFiles as file}
        <div class="file-card">
          <img src={getQRUrl('tiktok', file)} alt="QR for {file}" />
          <p>{file}</p>
          <a href="/social/tiktok/{file}" target="_blank">Direct Link</a>
        </div>
      {/each}
    </div>
  </div>
  
  <div class="platform-section">
    <h2>👻 Snapchat Exports</h2>
    <div class="file-grid">
      {#each snapFiles as file}
        <div class="file-card">
          <img src={getQRUrl('snap', file)} alt="QR for {file}" />
          <p>{file}</p>
          <a href="/social/snap/{file}" target="_blank">Direct Link</a>
        </div>
      {/each}
    </div>
  </div>
  
  <div class="instructions">
    <h3>📋 Quick Upload Guide</h3>
    <ul>
      <li><strong>TikTok:</strong> Save to Camera Roll → Open TikTok → Upload</li>
      <li><strong>Snapchat:</strong> Save to Camera Roll → Open Snap → Spotlight/Story</li>
    </ul>
  </div>
</div>

<style>
  .container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
    font-family: system-ui, -apple-system, sans-serif;
  }
  
  h1 {
    color: #333;
    text-align: center;
    margin-bottom: 1rem;
  }
  
  .info {
    text-align: center;
    color: #666;
    margin-bottom: 2rem;
  }
  
  .platform-section {
    margin-bottom: 3rem;
    padding: 1.5rem;
    background: #f8f9fa;
    border-radius: 8px;
  }
  
  h2 {
    color: #444;
    margin-bottom: 1rem;
  }
  
  .file-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1.5rem;
  }
  
  .file-card {
    background: white;
    padding: 1rem;
    border-radius: 8px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
  }
  
  .file-card img {
    width: 150px;
    height: 150px;
    margin: 0 auto 0.5rem;
  }
  
  .file-card p {
    font-size: 0.875rem;
    color: #666;
    margin: 0.5rem 0;
    word-break: break-all;
  }
  
  .file-card a {
    color: #007bff;
    text-decoration: none;
    font-size: 0.875rem;
  }
  
  .file-card a:hover {
    text-decoration: underline;
  }
  
  .instructions {
    background: #e8f4f8;
    padding: 1.5rem;
    border-radius: 8px;
    margin-top: 2rem;
  }
  
  .instructions h3 {
    color: #0066cc;
    margin-bottom: 1rem;
  }
  
  .instructions ul {
    margin: 0;
    padding-left: 1.5rem;
  }
  
  .instructions li {
    margin-bottom: 0.5rem;
    color: #555;
  }
</style>
