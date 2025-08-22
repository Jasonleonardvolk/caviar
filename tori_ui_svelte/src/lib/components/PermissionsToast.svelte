<!-- PermissionsToast.svelte -->
<!-- A toast notification for requesting device permissions -->
<script lang="ts">
  import { onMount } from 'svelte';
  import { fade, slide } from 'svelte/transition';
  
  let showMotionToast = false;
  let showMicToast = false;
  let hasMotionPermission = false;
  let hasMicPermission = false;
  
  // Check permissions on mount
  onMount(() => {
    checkPermissions();
    
    // Re-check when page becomes visible
    document.addEventListener('visibilitychange', () => {
      if (!document.hidden) checkPermissions();
    });
  });
  
  async function checkPermissions() {
    // Check motion permission (iOS 13+)
    if (typeof DeviceOrientationEvent !== 'undefined' && 
        typeof (DeviceOrientationEvent as any).requestPermission === 'function') {
      try {
        const response = await (DeviceOrientationEvent as any).requestPermission();
        hasMotionPermission = response === 'granted';
        showMotionToast = !hasMotionPermission;
      } catch (e) {
        console.log('Motion permission check failed:', e);
      }
    } else {
      // Non-iOS or older iOS - motion works without permission
      hasMotionPermission = true;
    }
    
    // Check mic permission
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      hasMicPermission = true;
      showMicToast = false;
      stream.getTracks().forEach(track => track.stop()); // Clean up
    } catch (e) {
      hasMicPermission = false;
      showMicToast = true;
    }
  }
  
  async function requestMotionPermission() {
    if (typeof (DeviceOrientationEvent as any).requestPermission === 'function') {
      try {
        const response = await (DeviceOrientationEvent as any).requestPermission();
        hasMotionPermission = response === 'granted';
        showMotionToast = !hasMotionPermission;
        
        if (hasMotionPermission) {
          // Trigger a custom event so the app knows to re-init motion features
          window.dispatchEvent(new CustomEvent('motionPermissionGranted'));
        }
      } catch (e) {
        console.error('Failed to request motion permission:', e);
      }
    }
  }
  
  async function requestMicPermission() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      hasMicPermission = true;
      showMicToast = false;
      stream.getTracks().forEach(track => track.stop());
      
      // Trigger event for audio-reactive features
      window.dispatchEvent(new CustomEvent('micPermissionGranted'));
    } catch (e) {
      console.error('Failed to request mic permission:', e);
      showMicToast = true;
    }
  }
  
  function dismissToast(type: 'motion' | 'mic') {
    if (type === 'motion') showMotionToast = false;
    if (type === 'mic') showMicToast = false;
  }
</script>

<!-- Motion Permission Toast -->
{#if showMotionToast}
  <div class="toast toast-motion" transition:slide={{ duration: 300 }}>
    <div class="toast-content">
      <span class="toast-icon">📱</span>
      <span class="toast-text">Motion access needed for tilt effects</span>
      <button class="toast-action" on:click={requestMotionPermission}>
        Grant Access
      </button>
      <button class="toast-dismiss" on:click={() => dismissToast('motion')}>
        ✕
      </button>
    </div>
  </div>
{/if}

<!-- Microphone Permission Toast -->
{#if showMicToast}
  <div class="toast toast-mic" transition:slide={{ duration: 300, delay: 100 }}>
    <div class="toast-content">
      <span class="toast-icon">🎤</span>
      <span class="toast-text">Microphone access for audio-reactive visuals</span>
      <button class="toast-action" on:click={requestMicPermission}>
        Grant Access
      </button>
      <button class="toast-dismiss" on:click={() => dismissToast('mic')}>
        ✕
      </button>
    </div>
  </div>
{/if}

<style>
  .toast {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    background: linear-gradient(135deg, rgba(0, 0, 0, 0.95) 0%, rgba(30, 0, 50, 0.95) 100%);
    backdrop-filter: blur(10px);
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    z-index: 9999;
    padding: 12px 16px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
  }
  
  .toast-mic {
    top: 60px; /* Stack below motion toast if both show */
  }
  
  .toast-content {
    display: flex;
    align-items: center;
    gap: 12px;
    max-width: 600px;
    margin: 0 auto;
  }
  
  .toast-icon {
    font-size: 20px;
    flex-shrink: 0;
  }
  
  .toast-text {
    flex: 1;
    color: rgba(255, 255, 255, 0.9);
    font-size: 14px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  }
  
  .toast-action {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    transition: transform 0.2s, box-shadow 0.2s;
    white-space: nowrap;
  }
  
  .toast-action:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
  }
  
  .toast-action:active {
    transform: translateY(0);
  }
  
  .toast-dismiss {
    background: transparent;
    color: rgba(255, 255, 255, 0.5);
    border: none;
    padding: 4px 8px;
    cursor: pointer;
    font-size: 18px;
    transition: color 0.2s;
  }
  
  .toast-dismiss:hover {
    color: rgba(255, 255, 255, 0.8);
  }
  
  @media (max-width: 600px) {
    .toast {
      padding: 10px 12px;
    }
    
    .toast-content {
      gap: 8px;
    }
    
    .toast-text {
      font-size: 13px;
    }
    
    .toast-action {
      padding: 5px 12px;
      font-size: 12px;
    }
  }
</style>