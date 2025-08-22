<script lang="ts">
    import { ghostPersona } from '$lib/stores/ghostPersona';
    import { fade, scale } from 'svelte/transition';
    import { onMount } from 'svelte';
    
    // Get persona colors
    const personaColors = {
        'Enola': '#2563eb',      // Investigation blue
        'Mentor': '#4f46e5',     // Indigo
        'Scholar': '#059669',    // Emerald
        'Explorer': '#dc2626',   // Red
        'Architect': '#7c3aed',  // Violet
        'Creator': '#ea580c'     // Orange
    };
    
    // Get persona icons
    const personaIcons = {
        'Enola': '🔍',
        'Mentor': '🧑‍🏫',
        'Scholar': '📚',
        'Explorer': '🧭',
        'Architect': '🏛️',
        'Creator': '🎨'
    };
    
    $: currentPersona = $ghostPersona.activePersona || 'Enola';
    $: personaColor = personaColors[currentPersona] || '#2563eb';
    $: personaIcon = personaIcons[currentPersona] || '👤';
    $: personaMood = $ghostPersona.mood || 'analytical';
    
    let showPulse = false;
    
    // Pulse animation when persona changes
    $: if (currentPersona) {
        showPulse = true;
        setTimeout(() => showPulse = false, 1000);
    }
</script>

<div class="hologram-persona-display">
    <!-- Main hologram container -->
    <div class="hologram-container" style="--persona-color: {personaColor}">
        <!-- Outer glow ring -->
        <div class="glow-ring" class:pulse={showPulse}></div>
        
        <!-- Inner persona display -->
        <div class="persona-core" transition:scale={{duration: 300}}>
            <!-- Persona icon -->
            <div class="persona-icon">
                {personaIcon}
            </div>
            
            <!-- Persona name -->
            <div class="persona-name">
                {currentPersona}
            </div>
            
            <!-- Mood indicator -->
            <div class="persona-mood">
                {personaMood}
            </div>
        </div>
        
        <!-- Floating particles effect -->
        <div class="particles">
            {#each Array(6) as _, i}
                <div 
                    class="particle" 
                    style="
                        --delay: {i * 0.2}s;
                        --duration: {3 + i * 0.5}s;
                        --size: {4 + i * 2}px;
                    "
                ></div>
            {/each}
        </div>
    </div>
    
    <!-- Status bar -->
    <div class="status-bar">
        <div class="status-item">
            <span class="status-label">Persona:</span>
            <span class="status-value" style="color: {personaColor}">{currentPersona}</span>
        </div>
        <div class="status-item">
            <span class="status-label">Mood:</span>
            <span class="status-value">{personaMood}</span>
        </div>
    </div>
</div>

<style>
    .hologram-persona-display {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 1rem;
        padding: 1rem;
        background: rgba(0, 0, 0, 0.02);
        border-radius: 0.75rem;
    }
    
    .hologram-container {
        position: relative;
        width: 150px;
        height: 150px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .glow-ring {
        position: absolute;
        inset: -10px;
        border: 2px solid var(--persona-color);
        border-radius: 50%;
        opacity: 0.3;
        filter: blur(4px);
        transition: all 0.3s ease;
    }
    
    .glow-ring.pulse {
        animation: hologram-pulse 1s ease-out;
    }
    
    @keyframes hologram-pulse {
        0% {
            transform: scale(1);
            opacity: 0.3;
        }
        50% {
            transform: scale(1.2);
            opacity: 0.6;
        }
        100% {
            transform: scale(1);
            opacity: 0.3;
        }
    }
    
    .persona-core {
        position: relative;
        width: 120px;
        height: 120px;
        background: linear-gradient(135deg, 
            rgba(255, 255, 255, 0.1), 
            rgba(255, 255, 255, 0.05)
        );
        border: 2px solid var(--persona-color);
        border-radius: 50%;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
        backdrop-filter: blur(10px);
        box-shadow: 
            0 0 20px rgba(var(--persona-color-rgb), 0.3),
            inset 0 0 20px rgba(var(--persona-color-rgb), 0.1);
    }
    
    .persona-icon {
        font-size: 2.5rem;
        filter: drop-shadow(0 0 10px var(--persona-color));
    }
    
    .persona-name {
        font-size: 0.875rem;
        font-weight: 600;
        color: var(--persona-color);
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    
    .persona-mood {
        font-size: 0.75rem;
        color: #6b7280;
        font-style: italic;
    }
    
    .particles {
        position: absolute;
        inset: 0;
        pointer-events: none;
    }
    
    .particle {
        position: absolute;
        width: var(--size);
        height: var(--size);
        background: var(--persona-color);
        border-radius: 50%;
        opacity: 0;
        animation: float-particle var(--duration) var(--delay) infinite ease-in-out;
    }
    
    @keyframes float-particle {
        0%, 100% {
            transform: translate(0, 0) scale(0);
            opacity: 0;
        }
        10% {
            transform: translate(0, -10px) scale(1);
            opacity: 0.6;
        }
        90% {
            transform: translate(0, -100px) scale(0.5);
            opacity: 0;
        }
    }
    
    .status-bar {
        display: flex;
        gap: 1.5rem;
        padding: 0.5rem 1rem;
        background: rgba(0, 0, 0, 0.05);
        border-radius: 9999px;
        font-size: 0.75rem;
    }
    
    .status-item {
        display: flex;
        align-items: center;
        gap: 0.25rem;
    }
    
    .status-label {
        color: #6b7280;
    }
    
    .status-value {
        font-weight: 500;
        color: #1f2937;
    }
    
    /* Dark mode support */
    :global(.dark) .hologram-persona-display {
        background: rgba(255, 255, 255, 0.02);
    }
    
    :global(.dark) .persona-core {
        background: linear-gradient(135deg, 
            rgba(0, 0, 0, 0.3), 
            rgba(0, 0, 0, 0.1)
        );
    }
    
    :global(.dark) .status-bar {
        background: rgba(255, 255, 255, 0.05);
    }
    
    :global(.dark) .status-label {
        color: #9ca3af;
    }
    
    :global(.dark) .status-value {
        color: #e5e7eb;
    }
    
    :global(.dark) .persona-mood {
        color: #9ca3af;
    }
</style>
