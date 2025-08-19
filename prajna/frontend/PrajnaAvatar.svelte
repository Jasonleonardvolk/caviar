<!--
Prajna Avatar Component - Audio/Visual Interface
==============================================

This component provides an animated avatar that represents Prajna's state
with visual feedback for speaking, listening, thinking, and processing.
-->

<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { tweened } from 'svelte/motion';
  import { cubicOut } from 'svelte/easing';
  import { spring } from 'svelte/motion';
  
  // Props
  export let state: 'idle' | 'listening' | 'thinking' | 'speaking' | 'processing' = 'idle';
  export let audioLevel: number = 0; // 0-1 for audio visualization
  export let mood: 'neutral' | 'happy' | 'confused' | 'focused' = 'neutral';
  export let size: 'small' | 'medium' | 'large' = 'medium';
  export let showAudioWaves: boolean = true;
  export let showStatusText: boolean = true;
  
  // Animation values
  const pulseScale = spring(1, { stiffness: 0.1, damping: 0.9 });
  const eyeOpenness = tweened(1, { duration: 200 });
  const mouthOpenness = tweened(0, { duration: 100 });
  const glowIntensity = tweened(0.3, { duration: 500 });
  const rotation = tweened(0, { duration: 1000, easing: cubicOut });
  
  // Avatar dimensions
  $: avatarSize = size === 'small' ? 120 : size === 'large' ? 280 : 200;
  $: svgSize = avatarSize;
  $: centerX = svgSize / 2;
  $: centerY = svgSize / 2;
  $: baseRadius = svgSize * 0.35;
  
  // Colors based on state
  $: primaryColor = getPrimaryColor(state);
  $: secondaryColor = getSecondaryColor(state);
  $: glowColor = getGlowColor(state);
  
  // Audio visualization
  let audioWavePoints: number[] = Array(12).fill(0);
  let waveAnimationId: number | null = null;
  
  // State management
  let previousState = state;
  let blinkInterval: number | null = null;
  
  // Lifecycle
  onMount(() => {
    startBlinking();
    if (showAudioWaves) {
      startAudioVisualization();
    }
  });
  
  onDestroy(() => {
    if (blinkInterval) clearInterval(blinkInterval);
    if (waveAnimationId) cancelAnimationFrame(waveAnimationId);
  });
  
  // Reactive state changes
  $: if (state !== previousState) {
    handleStateChange(previousState, state);
    previousState = state;
  }
  
  // Update mouth based on audio level when speaking
  $: if (state === 'speaking' && audioLevel > 0) {
    mouthOpenness.set(audioLevel * 0.8);
  } else if (state !== 'speaking') {
    mouthOpenness.set(0);
  }
  
  // Functions
  function getPrimaryColor(state: string): string {
    switch (state) {
      case 'listening': return '#10b981'; // green
      case 'thinking': return '#6366f1'; // indigo
      case 'speaking': return '#8b5cf6'; // purple
      case 'processing': return '#f59e0b'; // amber
      default: return '#667eea'; // default blue
    }
  }
  
  function getSecondaryColor(state: string): string {
    switch (state) {
      case 'listening': return '#34d399';
      case 'thinking': return '#818cf8';
      case 'speaking': return '#a78bfa';
      case 'processing': return '#fbbf24';
      default: return '#7c8cec';
    }
  }
  
  function getGlowColor(state: string): string {
    switch (state) {
      case 'listening': return 'rgba(16, 185, 129, 0.4)';
      case 'thinking': return 'rgba(99, 102, 241, 0.4)';
      case 'speaking': return 'rgba(139, 92, 246, 0.4)';
      case 'processing': return 'rgba(245, 158, 11, 0.4)';
      default: return 'rgba(102, 126, 234, 0.4)';
    }
  }
  
  function getStatusText(state: string): string {
    switch (state) {
      case 'listening': return 'Listening...';
      case 'thinking': return 'Thinking...';
      case 'speaking': return 'Speaking';
      case 'processing': return 'Processing...';
      default: return 'Ready';
    }
  }
  
  function handleStateChange(oldState: string, newState: string) {
    // Update animations based on state
    switch (newState) {
      case 'listening':
        pulseScale.set(1.05);
        glowIntensity.set(0.6);
        rotation.set(0);
        break;
      case 'thinking':
        pulseScale.set(0.95);
        glowIntensity.set(0.8);
        // Gentle rotation for thinking
        animateThinking();
        break;
      case 'speaking':
        pulseScale.set(1.1);
        glowIntensity.set(1);
        rotation.set(0);
        break;
      case 'processing':
        pulseScale.set(1);
        glowIntensity.set(0.7);
        // Continuous rotation for processing
        animateProcessing();
        break;
      default:
        pulseScale.set(1);
        glowIntensity.set(0.3);
        rotation.set(0);
    }
  }
  
  function startBlinking() {
    blinkInterval = setInterval(() => {
      if (state !== 'speaking' && Math.random() > 0.7) {
        blink();
      }
    }, 3000);
  }
  
  async function blink() {
    await eyeOpenness.set(0.1);
    await new Promise(resolve => setTimeout(resolve, 150));
    await eyeOpenness.set(1);
  }
  
  function animateThinking() {
    let direction = 1;
    const animate = () => {
      if (state !== 'thinking') return;
      rotation.update(r => r + direction * 5);
      if ($rotation > 10 || $rotation < -10) {
        direction *= -1;
      }
      requestAnimationFrame(animate);
    };
    animate();
  }
  
  function animateProcessing() {
    const animate = () => {
      if (state !== 'processing') return;
      rotation.update(r => (r + 2) % 360);
      requestAnimationFrame(animate);
    };
    animate();
  }
  
  function startAudioVisualization() {
    const animate = () => {
      if (state === 'speaking' || state === 'listening') {
        // Update wave points based on audio level
        audioWavePoints = audioWavePoints.map((point, i) => {
          const offset = Math.sin(Date.now() * 0.001 + i) * 0.5 + 0.5;
          return audioLevel * 20 * offset;
        });
      } else {
        // Calm waves when not speaking/listening
        audioWavePoints = audioWavePoints.map(point => point * 0.9);
      }
      waveAnimationId = requestAnimationFrame(animate);
    };
    animate();
  }
  
  function createWavePath(): string {
    const points = audioWavePoints.length;
    const angleStep = (Math.PI * 2) / points;
    
    let path = '';
    audioWavePoints.forEach((amplitude, i) => {
      const angle = i * angleStep;
      const radius = baseRadius + amplitude;
      const x = centerX + Math.cos(angle) * radius;
      const y = centerY + Math.sin(angle) * radius;
      
      if (i === 0) {
        path += `M ${x} ${y}`;
      } else {
        // Create smooth curves between points
        const prevAngle = (i - 1) * angleStep;
        const prevRadius = baseRadius + audioWavePoints[i - 1];
        const prevX = centerX + Math.cos(prevAngle) * prevRadius;
        const prevY = centerY + Math.sin(prevAngle) * prevRadius;
        
        const cp1x = prevX + Math.cos(prevAngle + Math.PI / 2) * 10;
        const cp1y = prevY + Math.sin(prevAngle + Math.PI / 2) * 10;
        const cp2x = x - Math.cos(angle + Math.PI / 2) * 10;
        const cp2y = y - Math.sin(angle + Math.PI / 2) * 10;
        
        path += ` C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${x} ${y}`;
      }
    });
    path += ' Z';
    return path;
  }
</script>

<div class="prajna-avatar-container {size}">
  <svg 
    width={svgSize} 
    height={svgSize} 
    viewBox="0 0 {svgSize} {svgSize}"
    style="transform: rotate({$rotation}deg) scale({$pulseScale})"
  >
    <!-- Glow effect -->
    <defs>
      <radialGradient id="glow-gradient">
        <stop offset="0%" style="stop-color:{glowColor};stop-opacity:1" />
        <stop offset="100%" style="stop-color:{glowColor};stop-opacity:0" />
      </radialGradient>
      
      <filter id="glow">
        <feGaussianBlur stdDeviation="4" result="coloredBlur"/>
        <feMerge>
          <feMergeNode in="coloredBlur"/>
          <feMergeNode in="SourceGraphic"/>
        </feMerge>
      </filter>
    </defs>
    
    <!-- Background glow -->
    <circle 
      cx={centerX} 
      cy={centerY} 
      r={baseRadius * 1.5}
      fill="url(#glow-gradient)"
      opacity={$glowIntensity}
    />
    
    <!-- Audio waves -->
    {#if showAudioWaves && (state === 'speaking' || state === 'listening')}
      <path
        d={createWavePath()}
        fill="none"
        stroke={secondaryColor}
        stroke-width="2"
        opacity="0.6"
        filter="url(#glow)"
      />
    {/if}
    
    <!-- Main circle -->
    <circle 
      cx={centerX} 
      cy={centerY} 
      r={baseRadius}
      fill={primaryColor}
      filter="url(#glow)"
    />
    
    <!-- Inner gradient -->
    <defs>
      <radialGradient id="inner-gradient">
        <stop offset="0%" style="stop-color:{secondaryColor};stop-opacity:0.3" />
        <stop offset="100%" style="stop-color:{primaryColor};stop-opacity:1" />
      </radialGradient>
    </defs>
    
    <circle 
      cx={centerX} 
      cy={centerY} 
      r={baseRadius * 0.9}
      fill="url(#inner-gradient)"
    />
    
    <!-- Eyes -->
    <g class="eyes">
      <!-- Left eye -->
      <ellipse 
        cx={centerX - baseRadius * 0.25} 
        cy={centerY - baseRadius * 0.1}
        rx={baseRadius * 0.08}
        ry={baseRadius * 0.15 * $eyeOpenness}
        fill="white"
      />
      <ellipse 
        cx={centerX - baseRadius * 0.25} 
        cy={centerY - baseRadius * 0.1}
        rx={baseRadius * 0.05}
        ry={baseRadius * 0.1 * $eyeOpenness}
        fill="#1a1a1a"
      />
      
      <!-- Right eye -->
      <ellipse 
        cx={centerX + baseRadius * 0.25} 
        cy={centerY - baseRadius * 0.1}
        rx={baseRadius * 0.08}
        ry={baseRadius * 0.15 * $eyeOpenness}
        fill="white"
      />
      <ellipse 
        cx={centerX + baseRadius * 0.25} 
        cy={centerY - baseRadius * 0.1}
        rx={baseRadius * 0.05}
        ry={baseRadius * 0.1 * $eyeOpenness}
        fill="#1a1a1a"
      />
    </g>
    
    <!-- Mouth -->
    <g class="mouth">
      {#if state === 'speaking'}
        <!-- Open mouth for speaking -->
        <ellipse 
          cx={centerX} 
          cy={centerY + baseRadius * 0.25}
          rx={baseRadius * 0.2}
          ry={baseRadius * 0.15 * $mouthOpenness}
          fill="#1a1a1a"
          opacity="0.8"
        />
      {:else if mood === 'happy'}
        <!-- Smile -->
        <path
          d="M {centerX - baseRadius * 0.2} {centerY + baseRadius * 0.2} 
             Q {centerX} {centerY + baseRadius * 0.35} 
             {centerX + baseRadius * 0.2} {centerY + baseRadius * 0.2}"
          fill="none"
          stroke="white"
          stroke-width="3"
          stroke-linecap="round"
        />
      {:else if mood === 'confused'}
        <!-- Wavy mouth -->
        <path
          d="M {centerX - baseRadius * 0.15} {centerY + baseRadius * 0.25} 
             Q {centerX - baseRadius * 0.05} {centerY + baseRadius * 0.2} 
             {centerX + baseRadius * 0.05} {centerY + baseRadius * 0.25}
             T {centerX + baseRadius * 0.15} {centerY + baseRadius * 0.2}"
          fill="none"
          stroke="white"
          stroke-width="3"
          stroke-linecap="round"
        />
      {:else}
        <!-- Neutral line -->
        <line
          x1={centerX - baseRadius * 0.15}
          y1={centerY + baseRadius * 0.25}
          x2={centerX + baseRadius * 0.15}
          y2={centerY + baseRadius * 0.25}
          stroke="white"
          stroke-width="3"
          stroke-linecap="round"
        />
      {/if}
    </g>
    
    <!-- Thinking dots -->
    {#if state === 'thinking'}
      <g class="thinking-dots">
        {#each [0, 1, 2] as i}
          <circle
            cx={centerX + (i - 1) * baseRadius * 0.15}
            cy={centerY + baseRadius * 0.5}
            r="3"
            fill={secondaryColor}
            opacity={0.8}
          >
            <animate
              attributeName="opacity"
              values="0.3;0.8;0.3"
              dur="1.5s"
              begin={`${i * 0.2}s`}
              repeatCount="indefinite"
            />
          </circle>
        {/each}
      </g>
    {/if}
  </svg>
  
  {#if showStatusText}
    <div class="status-text">
      {getStatusText(state)}
    </div>
  {/if}
</div>

<style>
  .prajna-avatar-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 1rem;
    padding: 1rem;
  }
  
  .prajna-avatar-container.small {
    gap: 0.5rem;
    padding: 0.5rem;
  }
  
  .prajna-avatar-container.large {
    gap: 1.5rem;
    padding: 1.5rem;
  }
  
  svg {
    transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  }
  
  .status-text {
    font-size: 1rem;
    font-weight: 500;
    color: #4b5563;
    text-align: center;
    animation: pulse 2s infinite;
  }
  
  .small .status-text {
    font-size: 0.875rem;
  }
  
  .large .status-text {
    font-size: 1.25rem;
  }
  
  @keyframes pulse {
    0%, 100% { opacity: 0.8; }
    50% { opacity: 1; }
  }
  
  /* Accessibility */
  @media (prefers-reduced-motion: reduce) {
    svg {
      transition: none;
    }
    
    .status-text {
      animation: none;
    }
  }
</style>
