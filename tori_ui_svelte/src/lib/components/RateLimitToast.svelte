<script lang="ts">
    import { fade } from 'svelte/transition';
    
    export let show = false;
    export let message = "Too many requests. Please wait a moment.";
    export let retryIn = 60; // seconds
    
    let countdown = retryIn;
    let interval: NodeJS.Timeout;
    
    $: if (show) {
        countdown = retryIn;
        startCountdown();
    } else {
        stopCountdown();
    }
    
    function startCountdown() {
        stopCountdown();
        interval = setInterval(() => {
            countdown--;
            if (countdown <= 0) {
                show = false;
            }
        }, 1000);
    }
    
    function stopCountdown() {
        if (interval) {
            clearInterval(interval);
        }
    }
    
    function close() {
        show = false;
    }
</script>

{#if show}
    <div class="rate-limit-toast" transition:fade={{ duration: 300 }}>
        <div class="toast-content">
            <div class="icon">⏳</div>
            <div class="message">
                <p>{message}</p>
                {#if countdown > 0}
                    <p class="countdown">Retry in {countdown}s</p>
                {/if}
            </div>
            <button class="close-btn" on:click={close}>×</button>
        </div>
    </div>
{/if}

<style>
    .rate-limit-toast {
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 1000;
        max-width: 400px;
    }
    
    .toast-content {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 1rem;
        background: #ff6b6b;
        color: white;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .icon {
        font-size: 2rem;
        flex-shrink: 0;
    }
    
    .message {
        flex: 1;
    }
    
    .message p {
        margin: 0;
        font-size: 0.875rem;
    }
    
    .countdown {
        font-size: 0.75rem;
        opacity: 0.9;
        margin-top: 0.25rem !important;
    }
    
    .close-btn {
        background: none;
        border: none;
        color: white;
        font-size: 1.5rem;
        cursor: pointer;
        padding: 0;
        width: 24px;
        height: 24px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        transition: background 0.2s;
    }
    
    .close-btn:hover {
        background: rgba(0, 0, 0, 0.1);
    }
</style>
