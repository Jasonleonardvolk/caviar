<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { fade, scale } from 'svelte/transition';
  import { ghostPersona, Ghost } from '$lib/stores/ghostPersona';
  import HolographicDisplay from './HolographicDisplay.svelte';
  import { writable } from 'svelte/store';
  
  // State
  let dropZone: HTMLDivElement;
  let imagePreview: string | null = null;
  let selectedImage: File | null = null;
  let isMorphing = false;
  let morphProgress = 0;
  let hologramReady = false;
  
  // Ghost selection happens automatically based on image
  let activeGhost: any = null;
  let ghostInsight: string = '';
  
  // Holographic display reference
  let holographicDisplay: HolographicDisplay;
  
  // Image analysis results
  const imageAnalysis = writable({
    dominantColors: [],
    edges: [],
    patterns: [],
    concepts: []
  });
  
  // Clean, minimal UI - the magic happens behind the scenes
  async function handleImageDrop(event: DragEvent) {
    event.preventDefault();
    const file = event.dataTransfer?.files[0];
    if (file && file.type.startsWith('image/')) {
      await processImage(file);
    }
  }
  
  async function processImage(file: File) {
    selectedImage = file;
    imagePreview = URL.createObjectURL(file);
    
    // Start morphing process
    isMorphing = true;
    morphProgress = 0;
    
    // Step 1: Analyze image to select appropriate Ghost
    const imageData = await loadImageData(file);
    activeGhost = selectGhostForImage(imageData);
    
    // Step 2: Ghost analyzes the image
    ghostInsight = await activeGhost.analyzeImage(imageData);
    morphProgress = 30;
    
    // Step 3: Extract visual concepts
    const analysis = await extractVisualConcepts(imageData);
    imageAnalysis.set(analysis);
    morphProgress = 60;
    
    // Step 4: Generate holographic data
    const hologramData = await generateHologramData(imageData, analysis, activeGhost);
    morphProgress = 90;
    
    // Step 5: Feed to holographic display
    await initializeHologram(hologramData);
    morphProgress = 100;
    
    setTimeout(() => {
      isMorphing = false;
      hologramReady = true;
    }, 500);
  }
  
  async function loadImageData(file: File): Promise<ImageData> {
    return new Promise((resolve) => {
      const img = new Image();
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d')!;
      
      img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
        resolve(ctx.getImageData(0, 0, img.width, img.height));
      };
      
      img.src = URL.createObjectURL(file);
    });
  }
  
  function selectGhostForImage(imageData: ImageData): any {
    // Intelligent Ghost selection based on image characteristics
    const brightness = calculateAverageBrightness(imageData);
    const complexity = calculateVisualComplexity(imageData);
    const colorVariance = calculateColorVariance(imageData);
    
    // Map image characteristics to Ghost personas
    if (complexity > 0.7 && colorVariance > 0.6) {
      return Ghost('Creator'); // Complex, colorful images → Creator
    } else if (brightness < 0.3) {
      return Ghost('Explorer'); // Dark, mysterious images → Explorer
    } else if (complexity < 0.3) {
      return Ghost('Architect'); // Simple, structured images → Architect
    } else if (colorVariance < 0.3) {
      return Ghost('Scholar'); // Monochromatic, focused images → Scholar
    } else {
      return Ghost('Mentor'); // Balanced images → Mentor
    }
  }
  
  function calculateAverageBrightness(imageData: ImageData): number {
    let sum = 0;
    const pixels = imageData.data;
    for (let i = 0; i < pixels.length; i += 4) {
      sum += (pixels[i] + pixels[i+1] + pixels[i+2]) / 3;
    }
    return sum / (pixels.length / 4) / 255;
  }
  
  function calculateVisualComplexity(imageData: ImageData): number {
    // Simple edge detection for complexity
    let edges = 0;
    const pixels = imageData.data;
    const width = imageData.width;
    
    for (let y = 1; y < imageData.height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const idx = (y * width + x) * 4;
        const center = pixels[idx];
        const neighbors = [
          pixels[idx - width * 4], // top
          pixels[idx + width * 4], // bottom
          pixels[idx - 4], // left
          pixels[idx + 4]  // right
        ];
        
        const diff = neighbors.reduce((sum, n) => sum + Math.abs(center - n), 0);
        if (diff > 100) edges++;
      }
    }
    
    return Math.min(1, edges / (width * imageData.height * 0.1));
  }
  
  function calculateColorVariance(imageData: ImageData): number {
    const pixels = imageData.data;
    const colors = new Set();
    
    for (let i = 0; i < pixels.length; i += 4) {
      // Quantize colors to reduce set size
      const r = Math.floor(pixels[i] / 32) * 32;
      const g = Math.floor(pixels[i+1] / 32) * 32;
      const b = Math.floor(pixels[i+2] / 32) * 32;
      colors.add(`${r},${g},${b}`);
    }
    
    return Math.min(1, colors.size / 100);
  }
  
  async function extractVisualConcepts(imageData: ImageData) {
    // Extract key visual elements for holographic transformation
    const dominantColors = extractDominantColors(imageData, 5);
    const edges = detectEdges(imageData);
    const patterns = detectPatterns(imageData);
    const concepts = generateConceptsFromVisual(dominantColors, edges, patterns);
    
    return { dominantColors, edges, patterns, concepts };
  }
  
  function extractDominantColors(imageData: ImageData, count: number) {
    // K-means clustering for dominant colors
    const pixels = imageData.data;
    const samples = [];
    
    // Sample pixels
    for (let i = 0; i < pixels.length; i += 400) { // Sample every 100th pixel
      samples.push({
        r: pixels[i],
        g: pixels[i+1],
        b: pixels[i+2]
      });
    }
    
    // Simple color quantization (full k-means would be more complex)
    const colorMap = new Map();
    samples.forEach(color => {
      const key = `${Math.floor(color.r/51)*51},${Math.floor(color.g/51)*51},${Math.floor(color.b/51)*51}`;
      colorMap.set(key, (colorMap.get(key) || 0) + 1);
    });
    
    return Array.from(colorMap.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, count)
      .map(([color]) => {
        const [r, g, b] = color.split(',').map(Number);
        return { r, g, b, h: rgbToHue(r, g, b) };
      });
  }
  
  function rgbToHue(r: number, g: number, b: number): number {
    r /= 255; g /= 255; b /= 255;
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    let h = 0;
    
    if (max !== min) {
      const d = max - min;
      switch (max) {
        case r: h = ((g - b) / d + (g < b ? 6 : 0)) / 6; break;
        case g: h = ((b - r) / d + 2) / 6; break;
        case b: h = ((r - g) / d + 4) / 6; break;
      }
    }
    
    return h * 360;
  }
  
  function detectEdges(imageData: ImageData) {
    // Simplified edge detection - returns edge map
    const edges = [];
    const width = imageData.width;
    const pixels = imageData.data;
    
    for (let y = 1; y < imageData.height - 1; y += 10) {
      for (let x = 1; x < width - 1; x += 10) {
        const idx = (y * width + x) * 4;
        const gx = pixels[idx + 4] - pixels[idx - 4];
        const gy = pixels[idx + width * 4] - pixels[idx - width * 4];
        const magnitude = Math.sqrt(gx * gx + gy * gy);
        
        if (magnitude > 50) {
          edges.push({ x: x / width, y: y / imageData.height, strength: magnitude / 255 });
        }
      }
    }
    
    return edges;
  }
  
  function detectPatterns(imageData: ImageData) {
    // Detect basic patterns - circles, lines, etc.
    // This is a placeholder - real implementation would use more sophisticated algorithms
    return {
      hasCircles: false,
      hasLines: true,
      hasChaos: false,
      symmetry: 0.5
    };
  }
  
  function generateConceptsFromVisual(colors: any[], edges: any[], patterns: any) {
    const concepts = [];
    
    // Color-based concepts
    if (colors[0]?.h < 30 || colors[0]?.h > 330) concepts.push('warmth');
    if (colors[0]?.h > 180 && colors[0]?.h < 240) concepts.push('coolness');
    
    // Edge-based concepts
    if (edges.length > 100) concepts.push('complexity');
    if (edges.length < 20) concepts.push('simplicity');
    
    // Pattern-based concepts
    if (patterns.hasCircles) concepts.push('unity');
    if (patterns.hasLines) concepts.push('structure');
    if (patterns.symmetry > 0.7) concepts.push('balance');
    
    return concepts;
  }
  
  async function generateHologramData(imageData: ImageData, analysis: any, ghost: any) {
    // Transform 2D image into 3D holographic data
    const { dominantColors, edges, concepts } = analysis;
    
    // Create volumetric density field from image
    const volumetricData = imageToVolumetric(imageData, edges);
    
    // Apply Ghost persona interpretation
    const ghostInterpretation = ghost.interpretVisualData(analysis);
    
    // Generate particle system parameters
    const particles = {
      positions: edges.map(e => ({
        x: (e.x - 0.5) * 2,
        y: (e.y - 0.5) * 2,
        z: e.strength
      })),
      colors: dominantColors.map(c => ({ h: c.h, s: 80, l: 50 })),
      velocities: generateParticleVelocities(edges, ghostInterpretation)
    };
    
    // Oscillator phases based on color frequencies
    const oscillatorPhases = dominantColors.map(c => (c.h / 360) * Math.PI * 2);
    
    return {
      volumetricData,
      particles,
      oscillatorPhases,
      ghostInterpretation,
      concepts,
      morphAnimation: {
        duration: 3000,
        easing: 'cubic-bezier(0.4, 0, 0.2, 1)'
      }
    };
  }
  
  function imageToVolumetric(imageData: ImageData, edges: any[]) {
    // Convert 2D image to 3D volumetric data
    const size = 64; // Volumetric resolution
    const volume = new Float32Array(size * size * size);
    
    // Create depth layers based on brightness and edges
    const pixels = imageData.data;
    const width = imageData.width;
    const height = imageData.height;
    
    for (let z = 0; z < size; z++) {
      const depthFactor = z / size;
      
      for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
          // Map volume coordinates to image coordinates
          const imgX = Math.floor((x / size) * width);
          const imgY = Math.floor((y / size) * height);
          const idx = (imgY * width + imgX) * 4;
          
          // Base density from brightness
          const brightness = (pixels[idx] + pixels[idx+1] + pixels[idx+2]) / 3 / 255;
          
          // Add edge contribution
          const edgeContribution = edges.reduce((sum, edge) => {
            const dx = (x / size) - edge.x;
            const dy = (y / size) - edge.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            return sum + edge.strength * Math.exp(-dist * 10) * depthFactor;
          }, 0);
          
          // Combine brightness and edges for 3D effect
          volume[z * size * size + y * size + x] = brightness * (1 - depthFactor * 0.8) + edgeContribution;
        }
      }
    }
    
    return volume;
  }
  
  function generateParticleVelocities(edges: any[], ghostInterpretation: any) {
    // Generate velocities based on Ghost interpretation
    return edges.map(edge => {
      const baseVelocity = {
        x: (Math.random() - 0.5) * 0.1,
        y: (Math.random() - 0.5) * 0.1,
        z: Math.random() * 0.2
      };
      
      // Apply Ghost personality to velocities
      if (ghostInterpretation.isExplorer) {
        // Explorer: chaotic, exploratory movement
        baseVelocity.x *= 3;
        baseVelocity.y *= 3;
      } else if (ghostInterpretation.isArchitect) {
        // Architect: structured, grid-like movement
        baseVelocity.x = Math.round(baseVelocity.x * 10) / 10;
        baseVelocity.y = Math.round(baseVelocity.y * 10) / 10;
      }
      
      return baseVelocity;
    });
  }
  
  async function initializeHologram(hologramData: any) {
    // Send data to holographic display
    if (holographicDisplay) {
      await holographicDisplay.loadHologramData(hologramData);
      await holographicDisplay.startMorphAnimation();
    }
  }
  
  function reset() {
    imagePreview = null;
    selectedImage = null;
    hologramReady = false;
    activeGhost = null;
    ghostInsight = '';
    imageAnalysis.set({
      dominantColors: [],
      edges: [],
      patterns: [],
      concepts: []
    });
  }
  
  // Cleanup
  onDestroy(() => {
    if (imagePreview) {
      URL.revokeObjectURL(imagePreview);
    }
  });
</script>

<div class="photo-morph-container">
  <!-- Clean, minimal interface -->
  <div class="morph-panel">
    {#if !imagePreview}
      <!-- Drop zone -->
      <div 
        bind:this={dropZone}
        class="drop-zone"
        on:drop={handleImageDrop}
        on:dragover|preventDefault
        on:dragenter|preventDefault
        role="button"
        tabindex="0"
      >
        <div class="drop-content">
          <div class="icon">📸</div>
          <p>Drop a photo to begin morphing</p>
          <span class="hint">The Ghost will interpret your image</span>
        </div>
      </div>
    {:else}
      <!-- Image preview with Ghost interpretation -->
      <div class="preview-container" transition:scale>
        <img src={imagePreview} alt="Selected" class="preview-image" />
        
        {#if activeGhost}
          <div class="ghost-overlay" transition:fade>
            <div class="ghost-badge" style="background: {activeGhost.color}">
              {activeGhost.name}
            </div>
            {#if ghostInsight}
              <p class="ghost-insight">{ghostInsight}</p>
            {/if}
          </div>
        {/if}
        
        {#if isMorphing}
          <div class="morph-progress">
            <div class="progress-bar" style="width: {morphProgress}%"></div>
          </div>
        {/if}
      </div>
    {/if}
    
    {#if hologramReady}
      <button class="reset-button" on:click={reset}>
        New Image
      </button>
    {/if}
  </div>
  
  <!-- Holographic display -->
  <div class="hologram-container" class:active={hologramReady}>
    <HolographicDisplay
      bind:this={holographicDisplay}
      width={400}
      height={400}
      enableVideo={false}
      showStats={false}
    />
  </div>
</div>

<style>
  .photo-morph-container {
    display: flex;
    gap: 24px;
    padding: 24px;
    height: 100%;
    background: #0a0a0a;
  }
  
  .morph-panel {
    flex: 1;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    min-width: 300px;
  }
  
  .drop-zone {
    width: 100%;
    max-width: 400px;
    height: 400px;
    border: 2px dashed rgba(138, 43, 226, 0.3);
    border-radius: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: all 0.3s;
    background: rgba(138, 43, 226, 0.05);
  }
  
  .drop-zone:hover {
    border-color: rgba(138, 43, 226, 0.6);
    background: rgba(138, 43, 226, 0.1);
  }
  
  .drop-zone:focus {
    outline: none;
    box-shadow: 0 0 0 3px rgba(138, 43, 226, 0.3);
  }
  
  .drop-content {
    text-align: center;
    color: #ccc;
  }
  
  .drop-content .icon {
    font-size: 48px;
    margin-bottom: 16px;
    opacity: 0.6;
  }
  
  .drop-content p {
    margin: 0;
    font-size: 16px;
    font-weight: 500;
  }
  
  .drop-content .hint {
    display: block;
    margin-top: 8px;
    font-size: 14px;
    opacity: 0.6;
  }
  
  .preview-container {
    position: relative;
    width: 100%;
    max-width: 400px;
  }
  
  .preview-image {
    width: 100%;
    border-radius: 12px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
  }
  
  .ghost-overlay {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    padding: 20px;
    background: linear-gradient(to top, rgba(0, 0, 0, 0.9), transparent);
    border-radius: 0 0 12px 12px;
  }
  
  .ghost-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 16px;
    color: white;
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 8px;
  }
  
  .ghost-insight {
    margin: 0;
    color: #e0e0e0;
    font-size: 14px;
    line-height: 1.5;
  }
  
  .morph-progress {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: rgba(0, 0, 0, 0.3);
    border-radius: 0 0 12px 12px;
    overflow: hidden;
  }
  
  .progress-bar {
    height: 100%;
    background: linear-gradient(90deg, #8a2be2, #ff00ff);
    transition: width 0.3s ease;
  }
  
  .reset-button {
    margin-top: 20px;
    padding: 10px 24px;
    background: rgba(138, 43, 226, 0.2);
    border: 1px solid rgba(138, 43, 226, 0.4);
    color: #ccc;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.3s;
  }
  
  .reset-button:hover {
    background: rgba(138, 43, 226, 0.3);
    border-color: rgba(138, 43, 226, 0.6);
    color: white;
  }
  
  .hologram-container {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    opacity: 0.3;
    transition: opacity 0.5s;
  }
  
  .hologram-container.active {
    opacity: 1;
  }
</style>
