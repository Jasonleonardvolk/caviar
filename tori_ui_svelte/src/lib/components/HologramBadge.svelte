<script>
    import { onMount, onDestroy } from 'svelte';
    import { hologramBridge } from '$lib/hologramBridge.js';
    
    let isConnected = false;
    let lastPing = null;
    let status = 'connecting';
    
    function updateStatus() {
        if (isConnected && lastPing) {
            const timeSinceLastPing = Date.now() - lastPing;
            if (timeSinceLastPing < 10000) { // 10 seconds
                status = 'live';
            } else {
                status = 'stale';
            }
        } else if (isConnected) {
            status = 'connected';
        } else {
            status = 'disconnected';
        }
    }
    
    onMount(() => {
        // Connect to SSE
        hologramBridge.connect();
        
        // Listen for events
        hologramBridge.on('connected', () => {
            isConnected = true;
            updateStatus();
        });
        
        hologramBridge.on('disconnected', () => {
            isConnected = false;
            updateStatus();
        });
        
        hologramBridge.on('ping', (data) => {
            lastPing = Date.now();
            updateStatus();
        });
        
        hologramBridge.on('error', () => {
            isConnected = false;
            updateStatus();
        });
        
        // Check status periodically
        const interval = setInterval(updateStatus, 1000);
        
        return () => {
            clearInterval(interval);
        };
    });
    
    onDestroy(() => {
        hologramBridge.disconnect();
    });
</script>

<div class="hologram-badge" class:live={status === 'live'} class:connected={status === 'connected'} class:disconnected={status === 'disconnected'}>
    <span class="icon">
        {#if status === 'live'}
            💓
        {:else if status === 'connected'}
            🟢
        {:else if status === 'connecting'}
            🟡
        {:else}
            🔴
        {/if}
    </span>
    <span class="label">Hologram {status}</span>
</div>

<style>
    .hologram-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .hologram-badge.live {
        background-color: #10b981;
        color: white;
    }
    
    .hologram-badge.connected {
        background-color: #3b82f6;
        color: white;
    }
    
    .hologram-badge.disconnected {
        background-color: #ef4444;
        color: white;
    }
    
    .icon {
        font-size: 1rem;
    }
    
    @keyframes pulse {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: 0.5;
        }
    }
    
    .hologram-badge.live .icon {
        animation: pulse 2s ease-in-out infinite;
    }
</style>
