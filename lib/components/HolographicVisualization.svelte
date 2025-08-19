<!-- STEP 4: Holographic Memory 3D Visualization Component -->
<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { holographicMemory } from '$lib/cognitive/holographicMemory';
  
  let canvasElement: HTMLCanvasElement;
  let containerElement: HTMLElement;
  let mounted = false;
  let isRendering = false;
  
  // Visualization state
  let visualizationData: any = null;
  let unsubscribe: (() => void) | null = null;
  
  // Camera controls
  let camera = {
    position: { x: 0, y: 0, z: 20 },
    rotation: { x: 0, y: 0 },
    zoom: 1
  };
  
  // Mouse interaction
  let isDragging = false;
  let lastMousePos = { x: 0, y: 0 };
  
  // Animation frame
  let animationFrameId: number | null = null;
  
  onMount(() => {
    mounted = true;
    initializeVisualization();
    
    // Subscribe to holographic memory updates
    unsubscribe = holographicMemory.onUpdate((data) => {
      visualizationData = data;
      if (isRendering) {
        requestRender();
      }
    });
    
    // Initial data load
    visualizationData = holographicMemory.getVisualizationData();
    startRenderLoop();
    
    console.log('🌌 Holographic Memory 3D visualization initialized');
  });
  
  onDestroy(() => {
    if (unsubscribe) {
      unsubscribe();
    }
    
    if (animationFrameId) {
      cancelAnimationFrame(animationFrameId);
    }
    
    isRendering = false;
  });
  
  function initializeVisualization() {
    if (!canvasElement) return;
    
    const ctx = canvasElement.getContext('2d');
    if (!ctx) return;
    
    // Set canvas size
    resizeCanvas();
    
    // Add mouse event listeners
    canvasElement.addEventListener('mousedown', handleMouseDown);
    canvasElement.addEventListener('mousemove', handleMouseMove);
    canvasElement.addEventListener('mouseup', handleMouseUp);
    canvasElement.addEventListener('wheel', handleWheel, { passive: false });
    
    window.addEventListener('resize', resizeCanvas);
  }
  
  function resizeCanvas() {
    if (!canvasElement || !containerElement) return;
    
    const rect = containerElement.getBoundingClientRect();
    canvasElement.width = rect.width * window.devicePixelRatio;
    canvasElement.height = rect.height * window.devicePixelRatio;
    canvasElement.style.width = rect.width + 'px';
    canvasElement.style.height = rect.height + 'px';
    
    const ctx = canvasElement.getContext('2d');
    if (ctx) {
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    }
  }
  
  function startRenderLoop() {
    isRendering = true;
    render();
  }
  
  function requestRender() {
    if (animationFrameId) return;
    
    animationFrameId = requestAnimationFrame(() => {
      render();
      animationFrameId = null;
    });
  }
  
  function render() {
    if (!canvasElement || !visualizationData) return;
    
    const ctx = canvasElement.getContext('2d');
    if (!ctx) return;
    
    const width = canvasElement.width / window.devicePixelRatio;
    const height = canvasElement.height / window.devicePixelRatio;
    
    // Clear canvas
    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, width, height);
    
    // Draw background grid
    drawGrid(ctx, width, height);
    
    // Draw activation wave if present
    if (visualizationData.activationWave) {
      drawActivationWave(ctx, width, height, visualizationData.activationWave);
    }
    
    // Draw connections first (behind nodes)
    visualizationData.connections.forEach((connection: any) => {
      drawConnection(ctx, width, height, connection);
    });
    
    // Draw emergent clusters
    visualizationData.clusters.forEach((cluster: any) => {
      drawCluster(ctx, width, height, cluster);
    });
    
    // Draw concept nodes
    visualizationData.nodes.forEach((node: any) => {
      drawNode(ctx, width, height, node);
    });
    
    // Draw UI overlay
    drawUI(ctx, width, height);
    
    if (isRendering) {
      animationFrameId = requestAnimationFrame(render);
    }
  }
  
  function project3DTo2D(point3D: { x: number; y: number; z: number }, width: number, height: number) {
    // Simple 3D to 2D projection with camera controls
    const adjustedPoint = {
      x: point3D.x - camera.position.x,
      y: point3D.y - camera.position.y,
      z: point3D.z - camera.position.z
    };
    
    // Apply rotation
    const cosY = Math.cos(camera.rotation.y);
    const sinY = Math.sin(camera.rotation.y);
    const cosX = Math.cos(camera.rotation.x);
    const sinX = Math.sin(camera.rotation.x);
    
    // Rotate around Y axis
    const rotatedX = adjustedPoint.x * cosY - adjustedPoint.z * sinY;
    const rotatedZ = adjustedPoint.x * sinY + adjustedPoint.z * cosY;
    
    // Rotate around X axis
    const finalY = adjustedPoint.y * cosX - rotatedZ * sinX;
    const finalZ = adjustedPoint.y * sinX + rotatedZ * cosX;
    
    // Perspective projection
    const distance = Math.max(0.1, finalZ + 20);
    const scale = (400 / distance) * camera.zoom;
    
    return {
      x: width / 2 + rotatedX * scale,
      y: height / 2 - finalY * scale,
      z: finalZ,
      scale: scale
    };
  }
  
  function drawGrid(ctx: CanvasRenderingContext2D, width: number, height: number) {
    ctx.strokeStyle = '#1a1a1a';
    ctx.lineWidth = 0.5;
    
    const gridSize = 50;
    
    // Vertical lines
    for (let x = 0; x < width; x += gridSize) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    
    // Horizontal lines
    for (let y = 0; y < height; y += gridSize) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }
  }
  
  function drawNode(ctx: CanvasRenderingContext2D, width: number, height: number, node: any) {
    const projected = project3DTo2D(node.position, width, height);
    
    if (projected.z < -10) return; // Don't draw nodes too far behind
    
    const size = node.visual.size * projected.scale * 0.1;
    const glow = node.visual.glow;
    
    // Draw glow effect
    if (glow > 0) {
      const gradient = ctx.createRadialGradient(
        projected.x, projected.y, 0,
        projected.x, projected.y, size * 3
      );
      gradient.addColorStop(0, `${node.visual.color}${Math.floor(glow * 255).toString(16).padStart(2, '0')}`);
      gradient.addColorStop(1, 'transparent');
      
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(projected.x, projected.y, size * 3, 0, Math.PI * 2);
      ctx.fill();
    }
    
    // Draw main node
    ctx.fillStyle = node.visual.color;
    ctx.globalAlpha = node.visual.opacity;
    ctx.beginPath();
    ctx.arc(projected.x, projected.y, size, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = 1;
    
    // Draw node border
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(projected.x, projected.y, size, 0, Math.PI * 2);
    ctx.stroke();
    
    // Draw essence label
    if (size > 10) {
      ctx.fillStyle = '#ffffff';
      ctx.font = `${Math.min(12, size / 2)}px Arial`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(node.essence, projected.x, projected.y + size + 15);
    }
  }
  
  function drawConnection(ctx: CanvasRenderingContext2D, width: number, height: number, connection: any) {
    const sourceProjected = project3DTo2D(connection.source, width, height);
    const targetProjected = project3DTo2D(connection.target, width, height);
    
    if (sourceProjected.z < -10 && targetProjected.z < -10) return;
    
    const thickness = connection.visual.thickness;
    const alpha = Math.min(connection.strength, 0.8);
    
    ctx.strokeStyle = connection.visual.color;
    ctx.lineWidth = thickness;
    ctx.globalAlpha = alpha;
    
    ctx.beginPath();
    ctx.moveTo(sourceProjected.x, sourceProjected.y);
    ctx.lineTo(targetProjected.x, targetProjected.y);
    ctx.stroke();
    
    ctx.globalAlpha = 1;
    
    // Draw flow animation if animated
    if (connection.visual.animated) {
      const time = Date.now() * 0.003;
      const progress = (Math.sin(time) + 1) / 2;
      
      const flowX = sourceProjected.x + (targetProjected.x - sourceProjected.x) * progress;
      const flowY = sourceProjected.y + (targetProjected.y - sourceProjected.y) * progress;
      
      ctx.fillStyle = '#ffffff';
      ctx.beginPath();
      ctx.arc(flowX, flowY, 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  
  function drawCluster(ctx: CanvasRenderingContext2D, width: number, height: number, cluster: any) {
    const centroidProjected = project3DTo2D(cluster.centroid, width, height);
    
    if (centroidProjected.z < -10) return;
    
    const radius = 80 * centroidProjected.scale * 0.1;
    
    // Draw cluster boundary
    ctx.strokeStyle = cluster.visual.boundaryColor;
    ctx.lineWidth = 2;
    ctx.globalAlpha = cluster.visual.fillOpacity * 3;
    
    // Pulsing effect
    const time = Date.now() / cluster.visual.pulseRate;
    const pulseRadius = radius * (1 + Math.sin(time) * 0.2);
    
    ctx.beginPath();
    ctx.arc(centroidProjected.x, centroidProjected.y, pulseRadius, 0, Math.PI * 2);
    ctx.stroke();
    
    // Draw fill
    ctx.fillStyle = cluster.visual.boundaryColor;
    ctx.globalAlpha = cluster.visual.fillOpacity;
    ctx.fill();
    
    ctx.globalAlpha = 1;
    
    // Draw emergent property label
    ctx.fillStyle = '#ffffff';
    ctx.font = '10px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(
      cluster.emergentProperty.split(':')[0], 
      centroidProjected.x, 
      centroidProjected.y - pulseRadius - 10
    );
  }
  
  function drawActivationWave(ctx: CanvasRenderingContext2D, width: number, height: number, wave: any) {
    const centerProjected = project3DTo2D(wave.center, width, height);
    
    if (centerProjected.z < -10) return;
    
    const radius = wave.radius * centerProjected.scale * 0.1;
    const alpha = wave.strength;
    
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.globalAlpha = alpha;
    
    ctx.beginPath();
    ctx.arc(centerProjected.x, centerProjected.y, radius, 0, Math.PI * 2);
    ctx.stroke();
    
    ctx.globalAlpha = 1;
  }
  
  function drawUI(ctx: CanvasRenderingContext2D, width: number, height: number) {
    // Draw stats
    ctx.fillStyle = '#ffffff';
    ctx.font = '12px Arial';
    ctx.textAlign = 'left';
    
    const stats = [
      `Nodes: ${visualizationData.nodes.length}`,
      `Connections: ${visualizationData.connections.length}`,
      `Clusters: ${visualizationData.clusters.length}`,
      `Zoom: ${camera.zoom.toFixed(1)}x`
    ];
    
    stats.forEach((stat, i) => {
      ctx.fillText(stat, 10, 20 + i * 16);
    });
    
    // Draw instructions
    const instructions = [
      'Drag to rotate',
      'Scroll to zoom',
      'Watch for activation waves'
    ];
    
    ctx.textAlign = 'right';
    instructions.forEach((instruction, i) => {
      ctx.fillText(instruction, width - 10, height - 60 + i * 16);
    });
  }
  
  // Mouse interaction handlers
  function handleMouseDown(event: MouseEvent) {
    isDragging = true;
    lastMousePos = { x: event.clientX, y: event.clientY };
  }
  
  function handleMouseMove(event: MouseEvent) {
    if (!isDragging) return;
    
    const deltaX = event.clientX - lastMousePos.x;
    const deltaY = event.clientY - lastMousePos.y;
    
    camera.rotation.y += deltaX * 0.01;
    camera.rotation.x += deltaY * 0.01;
    
    // Clamp X rotation
    camera.rotation.x = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, camera.rotation.x));
    
    lastMousePos = { x: event.clientX, y: event.clientY };
    
    requestRender();
  }
  
  function handleMouseUp() {
    isDragging = false;
  }
  
  function handleWheel(event: WheelEvent) {
    event.preventDefault();
    
    const zoomFactor = event.deltaY > 0 ? 0.9 : 1.1;
    camera.zoom = Math.max(0.1, Math.min(5, camera.zoom * zoomFactor));
    
    requestRender();
  }
  
  // Test functions
  function createTestNodes() {
    const concepts = ['Learning', 'Thinking', 'Analysis', 'Creation', 'Connections'];
    const nodes: any[] = [];
    
    concepts.forEach((concept, i) => {
      const node = holographicMemory.createConceptNode(concept, 0.5 + Math.random() * 0.5);
      nodes.push(node);
    });
    
    // Create some connections
    for (let i = 0; i < nodes.length - 1; i++) {
      holographicMemory.createConnection(
        nodes[i].id,
        nodes[i + 1].id,
        0.3 + Math.random() * 0.7,
        'semantic'
      );
    }
    
    // Activate a random node
    setTimeout(() => {
      const randomNode = nodes[Math.floor(Math.random() * nodes.length)];
      holographicMemory.activateConcept(randomNode.id, 0.8);
    }, 1000);
    
    console.log('🌌 Created test holographic memory network');
  }
  
  function resetView() {
    camera = {
      position: { x: 0, y: 0, z: 20 },
      rotation: { x: 0, y: 0 },
      zoom: 1
    };
    requestRender();
  }
</script>

<div bind:this={containerElement} class="relative w-full h-full bg-black overflow-hidden rounded-lg">
  <canvas 
    bind:this={canvasElement}
    class="absolute inset-0 cursor-grab active:cursor-grabbing"
  />
  
  <!-- Controls overlay -->
  <div class="absolute top-4 left-4 space-y-2">
    <button
      on:click={createTestNodes}
      class="px-3 py-1 bg-purple-600 hover:bg-purple-700 text-white text-xs rounded transition-colors"
    >
      Create Test Network
    </button>
    
    <button
      on:click={resetView}
      class="px-3 py-1 bg-blue-600 hover:bg-blue-700 text-white text-xs rounded transition-colors"
    >
      Reset View
    </button>
    
    <button
      on:click={() => holographicMemory.clear()}
      class="px-3 py-1 bg-red-600 hover:bg-red-700 text-white text-xs rounded transition-colors"
    >
      Clear Memory
    </button>
  </div>
  
  <!-- Status indicator -->
  <div class="absolute bottom-4 left-4">
    <div class="flex items-center space-x-2 text-white text-xs">
      <div class="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
      <span>Holographic Memory Active</span>
    </div>
  </div>
  
  <!-- Node count display -->
  {#if visualizationData}
    <div class="absolute top-4 right-4 text-white text-xs">
      <div class="bg-black bg-opacity-50 rounded px-2 py-1">
        3D Memory Visualization
      </div>
    </div>
  {/if}
</div>

<style>
  canvas {
    image-rendering: pixelated;
    image-rendering: -moz-crisp-edges;
    image-rendering: crisp-edges;
  }
</style>
