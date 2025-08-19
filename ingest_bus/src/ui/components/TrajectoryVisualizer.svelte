<!--
ψTrajectory Visualizer Component for TORI

A timeline UI component that visualizes TORI's cognitive evolution, showing when
each concept was introduced, by what source, and how it grew or changed over time.

Features:
- Interactive horizontal timeline
- Concept injection markers with metadata
- Hover tooltips with detailed information
- Click-through to full concept details
- Playback slider for temporal navigation
- Color-coded drift zones and trust indicators
- Real-time updates for live ingestion

Usage:
<ψTrajectoryVisualizer 
  trajectoryData={trajectoryData}
  timeRange={[startTime, endTime]}
  onConceptClick={handleConceptClick}
  showDriftZones={true}
  enablePlayback={true}
/>
-->

<script lang="ts">
  import { onMount, createEventDispatcher } from 'svelte';
  import { writable, derived } from 'svelte/store';
  
  // Types
  interface ConceptInjection {
    id: string;
    timestamp: number;
    concept: string;
    sourceId: string;
    sourceType: 'video' | 'document' | 'conversation' | 'stream';
    confidence: number;
    conceptType: string;
    summary: string;
    ghostReflections?: GhostReflection[];
    driftScore?: number;
    trustScore?: number;
    memoryLinks: string[];
  }
  
  interface GhostReflection {
    persona: string;
    message: string;
    confidence: number;
    timestamp: number;
  }
  
  interface TrajectoryData {
    concepts: ConceptInjection[];
    timeRange: [number, number];
    metadata: {
      totalSources: number;
      conceptCount: number;
      driftEvents: number;
      trustViolations: number;
    };
  }
  
  interface DriftZone {
    startTime: number;
    endTime: number;
    severity: 'low' | 'medium' | 'high';
    affectedConcepts: string[];
  }
  
  // Props
  export let trajectoryData: TrajectoryData;
  export let timeRange: [number, number] | null = null;
  export let onConceptClick: ((concept: ConceptInjection) => void) | null = null;
  export let showDriftZones: boolean = true;
  export let enablePlayback: boolean = true;
  export let height: number = 400;
  export let autoUpdate: boolean = false;
  
  // Event dispatcher
  const dispatch = createEventDispatcher();
  
  // Stores
  const currentTime = writable(Date.now());
  const playbackTime = writable(trajectoryData.timeRange[0]);
  const isPlaying = writable(false);
  const selectedConcept = writable<ConceptInjection | null>(null);
  const hoveredConcept = writable<ConceptInjection | null>(null);
  const zoomLevel = writable(1);
  const panOffset = writable(0);
  
  // Derived stores
  const displayTimeRange = derived(
    [playbackTime, zoomLevel, panOffset],
    ([$playbackTime, $zoomLevel, $panOffset]) => {
      const range = timeRange || trajectoryData.timeRange;
      const duration = range[1] - range[0];
      const zoomedDuration = duration / $zoomLevel;
      const start = range[0] + $panOffset;
      return [start, start + zoomedDuration];
    }
  );
  
  const visibleConcepts = derived(
    [displayTimeRange],
    ([$displayTimeRange]) => {
      return trajectoryData.concepts.filter(concept =>
        concept.timestamp >= $displayTimeRange[0] &&
        concept.timestamp <= $displayTimeRange[1]
      );
    }
  );
  
  // Component state
  let svgContainer: SVGSVGElement;
  let tooltip: HTMLDivElement;
  let isDragging = false;
  let dragStartX = 0;
  let dragStartOffset = 0;
  
  // Constants
  const TIMELINE_MARGIN = { top: 40, right: 40, bottom: 60, left: 40 };
  const CONCEPT_RADIUS = 6;
  const DRIFT_ZONE_HEIGHT = 20;
  
  // Color schemes
  const SOURCE_COLORS = {
    video: '#3498db',
    document: '#2ecc71',
    conversation: '#f39c12',
    stream: '#e74c3c'
  };
  
  const TRUST_COLORS = {
    high: '#27ae60',
    medium: '#f39c12',
    low: '#e74c3c'
  };
  
  const DRIFT_COLORS = {
    low: '#f1c40f',
    medium: '#e67e22',
    high: '#c0392b'
  };
  
  // Computed properties
  $: timelineWidth = height * 2; // Aspect ratio 2:1
  $: timelineHeight = height - TIMELINE_MARGIN.top - TIMELINE_MARGIN.bottom;
  $: conceptY = TIMELINE_MARGIN.top + timelineHeight / 2;
  
  // Time scale function
  $: timeScale = (timestamp: number) => {
    const [start, end] = $displayTimeRange;
    const progress = (timestamp - start) / (end - start);
    return TIMELINE_MARGIN.left + progress * (timelineWidth - TIMELINE_MARGIN.left - TIMELINE_MARGIN.right);
  };
  
  // Trust score to color
  const trustScoreToColor = (score: number): string => {
    if (score >= 0.8) return TRUST_COLORS.high;
    if (score >= 0.5) return TRUST_COLORS.medium;
    return TRUST_COLORS.low;
  };
  
  // Generate drift zones
  $: driftZones = generateDriftZones(trajectoryData.concepts);
  
  function generateDriftZones(concepts: ConceptInjection[]): DriftZone[] {
    const zones: DriftZone[] = [];
    
    // Group concepts by time windows and detect drift patterns
    const timeWindow = 3600000; // 1 hour in milliseconds
    const conceptGroups = new Map<number, ConceptInjection[]>();
    
    concepts.forEach(concept => {
      const window = Math.floor(concept.timestamp / timeWindow);
      if (!conceptGroups.has(window)) {
        conceptGroups.set(window, []);
      }
      conceptGroups.get(window)!.push(concept);
    });
    
    // Analyze each group for drift indicators
    conceptGroups.forEach((groupConcepts, window) => {
      const driftingConcepts = groupConcepts.filter(c => (c.driftScore || 0) > 0.3);
      
      if (driftingConcepts.length > 0) {
        const startTime = window * timeWindow;
        const endTime = (window + 1) * timeWindow;
        const maxDrift = Math.max(...driftingConcepts.map(c => c.driftScore || 0));
        
        let severity: 'low' | 'medium' | 'high' = 'low';
        if (maxDrift > 0.7) severity = 'high';
        else if (maxDrift > 0.5) severity = 'medium';
        
        zones.push({
          startTime,
          endTime,
          severity,
          affectedConcepts: driftingConcepts.map(c => c.concept)
        });
      }
    });
    
    return zones;
  }
  
  // Event handlers
  function handleConceptHover(concept: ConceptInjection, event: MouseEvent) {
    hoveredConcept.set(concept);
    
    // Position tooltip
    if (tooltip) {
      tooltip.style.left = `${event.clientX + 10}px`;
      tooltip.style.top = `${event.clientY - 10}px`;
      tooltip.style.display = 'block';
    }
  }
  
  function handleConceptLeave() {
    hoveredConcept.set(null);
    if (tooltip) {
      tooltip.style.display = 'none';
    }
  }
  
  function handleConceptClickInternal(concept: ConceptInjection) {
    selectedConcept.set(concept);
    if (onConceptClick) {
      onConceptClick(concept);
    }
    dispatch('conceptClick', concept);
  }
  
  function handleTimelineClick(event: MouseEvent) {
    const rect = svgContainer.getBoundingClientRect();
    const x = event.clientX - rect.left - TIMELINE_MARGIN.left;
    const progress = x / (timelineWidth - TIMELINE_MARGIN.left - TIMELINE_MARGIN.right);
    const [start, end] = $displayTimeRange;
    const clickedTime = start + progress * (end - start);
    
    playbackTime.set(clickedTime);
    dispatch('timeClick', clickedTime);
  }
  
  function handleMouseDown(event: MouseEvent) {
    isDragging = true;
    dragStartX = event.clientX;
    dragStartOffset = $panOffset;
  }
  
  function handleMouseMove(event: MouseEvent) {
    if (!isDragging) return;
    
    const deltaX = event.clientX - dragStartX;
    const [start, end] = timeRange || trajectoryData.timeRange;
    const duration = end - start;
    const pixelToDuration = duration / timelineWidth;
    const newOffset = dragStartOffset - deltaX * pixelToDuration / $zoomLevel;
    
    // Constrain panning
    const maxOffset = duration * (1 - 1 / $zoomLevel);
    panOffset.set(Math.max(0, Math.min(maxOffset, newOffset)));
  }
  
  function handleMouseUp() {
    isDragging = false;
  }
  
  function handleWheel(event: WheelEvent) {
    event.preventDefault();
    const zoomFactor = event.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.max(1, Math.min(10, $zoomLevel * zoomFactor));
    zoomLevel.set(newZoom);
  }
  
  function togglePlayback() {
    isPlaying.update(playing => !playing);
  }
  
  function resetView() {
    zoomLevel.set(1);
    panOffset.set(0);
    playbackTime.set(trajectoryData.timeRange[0]);
  }
  
  // Playback animation
  let playbackInterval: number;
  
  $: if ($isPlaying && enablePlayback) {
    playbackInterval = setInterval(() => {
      playbackTime.update(time => {
        const [start, end] = $displayTimeRange;
        const newTime = time + (end - start) / 1000; // 1 second of real time = full range
        if (newTime >= end) {
          isPlaying.set(false);
          return start;
        }
        return newTime;
      });
    }, 16); // ~60fps
  } else if (playbackInterval) {
    clearInterval(playbackInterval);
  }
  
  // Format timestamp for display
  function formatTimestamp(timestamp: number): string {
    const date = new Date(timestamp);
    return date.toLocaleString();
  }
  
  function formatDuration(duration: number): string {
    const hours = Math.floor(duration / 3600000);
    const minutes = Math.floor((duration % 3600000) / 60000);
    const seconds = Math.floor((duration % 60000) / 1000);
    
    if (hours > 0) {
      return `${hours}h ${minutes}m ${seconds}s`;
    } else if (minutes > 0) {
      return `${minutes}m ${seconds}s`;
    } else {
      return `${seconds}s`;
    }
  }
  
  // Auto-update mechanism
  $: if (autoUpdate) {
    const updateInterval = setInterval(() => {
      currentTime.set(Date.now());
      dispatch('requestUpdate');
    }, 5000);
    
    return () => clearInterval(updateInterval);
  }
  
  onMount(() => {
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
    
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      if (playbackInterval) clearInterval(playbackInterval);
    };
  });
</script>

<div class="psi-trajectory-visualizer" style="height: {height}px;">
  <!-- Controls -->
  <div class="controls">
    <div class="playback-controls">
      {#if enablePlayback}
        <button on:click={togglePlayback} class="playback-btn">
          {$isPlaying ? '⏸️' : '▶️'}
        </button>
        <span class="playback-time">
          {formatTimestamp($playbackTime)}
        </span>
      {/if}
      <button on:click={resetView} class="reset-btn">🔄 Reset View</button>
    </div>
    
    <div class="zoom-controls">
      <button on:click={() => zoomLevel.update(z => Math.min(10, z * 1.2))}>🔍+</button>
      <span class="zoom-level">{$zoomLevel.toFixed(1)}x</span>
      <button on:click={() => zoomLevel.update(z => Math.max(1, z / 1.2))}>🔍-</button>
    </div>
    
    <div class="info">
      <span class="concept-count">{trajectoryData.concepts.length} concepts</span>
      <span class="time-range">
        {formatDuration(trajectoryData.timeRange[1] - trajectoryData.timeRange[0])}
      </span>
    </div>
  </div>
  
  <!-- Timeline SVG -->
  <svg
    bind:this={svgContainer}
    width={timelineWidth}
    height={height}
    on:click={handleTimelineClick}
    on:mousedown={handleMouseDown}
    on:wheel={handleWheel}
    class="timeline-svg"
  >
    <!-- Background -->
    <rect width="100%" height="100%" fill="#f8f9fa" />
    
    <!-- Drift zones -->
    {#if showDriftZones}
      {#each driftZones as zone}
        {@const startX = timeScale(zone.startTime)}
        {@const endX = timeScale(zone.endTime)}
        {@const width = endX - startX}
        
        <rect
          x={startX}
          y={TIMELINE_MARGIN.top - DRIFT_ZONE_HEIGHT}
          width={width}
          height={DRIFT_ZONE_HEIGHT}
          fill={DRIFT_COLORS[zone.severity]}
          opacity="0.3"
          class="drift-zone"
        >
          <title>
            Drift Zone ({zone.severity}): {zone.affectedConcepts.join(', ')}
          </title>
        </rect>
      {/each}
    {/if}
    
    <!-- Main timeline axis -->
    <line
      x1={TIMELINE_MARGIN.left}
      y1={conceptY}
      x2={timelineWidth - TIMELINE_MARGIN.right}
      y2={conceptY}
      stroke="#34495e"
      stroke-width="2"
      class="timeline-axis"
    />
    
    <!-- Time markers -->
    {#each Array.from({length: 6}, (_, i) => i) as i}
      {@const progress = i / 5}
      {@const [start, end] = $displayTimeRange}
      {@const timestamp = start + progress * (end - start)}
      {@const x = timeScale(timestamp)}
      
      <g class="time-marker">
        <line
          x1={x}
          y1={conceptY - 10}
          x2={x}
          y2={conceptY + 10}
          stroke="#7f8c8d"
          stroke-width="1"
        />
        <text
          x={x}
          y={conceptY + 25}
          text-anchor="middle"
          font-size="10"
          fill="#7f8c8d"
        >
          {new Date(timestamp).toLocaleTimeString()}
        </text>
      </g>
    {/each}
    
    <!-- Playback cursor -->
    {#if enablePlayback}
      {@const cursorX = timeScale($playbackTime)}
      <line
        x1={cursorX}
        y1={TIMELINE_MARGIN.top}
        x2={cursorX}
        y2={height - TIMELINE_MARGIN.bottom}
        stroke="#e74c3c"
        stroke-width="2"
        opacity="0.8"
        class="playback-cursor"
      />
    {/if}
    
    <!-- Concept markers -->
    {#each $visibleConcepts as concept}
      {@const x = timeScale(concept.timestamp)}
      {@const isSelected = $selectedConcept?.id === concept.id}
      {@const isHovered = $hoveredConcept?.id === concept.id}
      {@const radius = isSelected || isHovered ? CONCEPT_RADIUS * 1.5 : CONCEPT_RADIUS}
      {@const strokeWidth = isSelected ? 3 : isHovered ? 2 : 1}
      
      <g class="concept-marker">
        <!-- Concept circle -->
        <circle
          cx={x}
          cy={conceptY}
          r={radius}
          fill={SOURCE_COLORS[concept.sourceType]}
          stroke={trustScoreToColor(concept.trustScore || 1.0)}
          stroke-width={strokeWidth}
          opacity={concept.driftScore && concept.driftScore > 0.5 ? 0.7 : 1.0}
          class="concept-circle"
          on:mouseenter={(e) => handleConceptHover(concept, e)}
          on:mouseleave={handleConceptLeave}
          on:click={() => handleConceptClickInternal(concept)}
        />
        
        <!-- Confidence indicator -->
        <circle
          cx={x}
          cy={conceptY}
          r={radius * concept.confidence}
          fill="none"
          stroke="white"
          stroke-width="1"
          opacity="0.8"
          class="confidence-ring"
        />
        
        <!-- Ghost reflection indicators -->
        {#if concept.ghostReflections && concept.ghostReflections.length > 0}
          {#each concept.ghostReflections as reflection, i}
            <circle
              cx={x + (i - concept.ghostReflections.length / 2 + 0.5) * 3}
              cy={conceptY - radius - 8}
              r="2"
              fill="#9b59b6"
              opacity="0.8"
              class="ghost-indicator"
            >
              <title>{reflection.persona}: {reflection.message.substring(0, 100)}...</title>
            </circle>
          {/each}
        {/if}
        
        <!-- Drift warning -->
        {#if concept.driftScore && concept.driftScore > 0.5}
          <text
            x={x}
            y={conceptY - radius - 15}
            text-anchor="middle"
            font-size="12"
            fill={DRIFT_COLORS.high}
          >
            ⚠️
          </text>
        {/if}
      </g>
    {/each}
    
    <!-- Concept connections (memory links) -->
    {#each $visibleConcepts as concept}
      {@const x1 = timeScale(concept.timestamp)}
      {#each concept.memoryLinks as linkId}
        {@const linkedConcept = trajectoryData.concepts.find(c => c.id === linkId)}
        {#if linkedConcept && $visibleConcepts.includes(linkedConcept)}
          {@const x2 = timeScale(linkedConcept.timestamp)}
          <line
            x1={x1}
            y1={conceptY}
            x2={x2}
            y2={conceptY}
            stroke="#bdc3c7"
            stroke-width="1"
            opacity="0.3"
            stroke-dasharray="2,2"
            class="memory-link"
          />
        {/if}
      {/each}
    {/each}
  </svg>
  
  <!-- Tooltip -->
  <div bind:this={tooltip} class="tooltip" style="display: none;">
    {#if $hoveredConcept}
      <div class="tooltip-content">
        <div class="concept-header">
          <strong>{$hoveredConcept.concept}</strong>
          <span class="source-badge" style="background-color: {SOURCE_COLORS[$hoveredConcept.sourceType]}">
            {$hoveredConcept.sourceType}
          </span>
        </div>
        
        <div class="concept-details">
          <div class="detail-row">
            <span class="label">Time:</span>
            <span class="value">{formatTimestamp($hoveredConcept.timestamp)}</span>
          </div>
          <div class="detail-row">
            <span class="label">Source:</span>
            <span class="value">{$hoveredConcept.sourceId}</span>
          </div>
          <div class="detail-row">
            <span class="label">Confidence:</span>
            <span class="value">{($hoveredConcept.confidence * 100).toFixed(1)}%</span>
          </div>
          {#if $hoveredConcept.trustScore}
            <div class="detail-row">
              <span class="label">Trust:</span>
              <span class="value" style="color: {trustScoreToColor($hoveredConcept.trustScore)}">
                {($hoveredConcept.trustScore * 100).toFixed(1)}%
              </span>
            </div>
          {/if}
          {#if $hoveredConcept.driftScore}
            <div class="detail-row">
              <span class="label">Drift:</span>
              <span class="value" style="color: {DRIFT_COLORS[$hoveredConcept.driftScore > 0.7 ? 'high' : $hoveredConcept.driftScore > 0.5 ? 'medium' : 'low']}">
                {($hoveredConcept.driftScore * 100).toFixed(1)}%
              </span>
            </div>
          {/if}
        </div>
        
        <div class="concept-summary">
          {$hoveredConcept.summary}
        </div>
        
        {#if $hoveredConcept.ghostReflections && $hoveredConcept.ghostReflections.length > 0}
          <div class="ghost-reflections">
            <strong>Ghost Reflections:</strong>
            {#each $hoveredConcept.ghostReflections as reflection}
              <div class="ghost-reflection">
                <span class="ghost-name">{reflection.persona}:</span>
                <span class="ghost-message">{reflection.message}</span>
              </div>
            {/each}
          </div>
        {/if}
      </div>
    {/if}
  </div>
  
  <!-- Legend -->
  <div class="legend">
    <div class="legend-section">
      <h4>Sources</h4>
      {#each Object.entries(SOURCE_COLORS) as [type, color]}
        <div class="legend-item">
          <div class="legend-color" style="background-color: {color}"></div>
          <span>{type}</span>
        </div>
      {/each}
    </div>
    
    <div class="legend-section">
      <h4>Trust Level</h4>
      {#each Object.entries(TRUST_COLORS) as [level, color]}
        <div class="legend-item">
          <div class="legend-color" style="background-color: {color}"></div>
          <span>{level}</span>
        </div>
      {/each}
    </div>
    
    {#if showDriftZones}
      <div class="legend-section">
        <h4>Drift Zones</h4>
        {#each Object.entries(DRIFT_COLORS) as [severity, color]}
          <div class="legend-item">
            <div class="legend-color" style="background-color: {color}"></div>
            <span>{severity}</span>
          </div>
        {/each}
      </div>
    {/if}
  </div>
</div>

<style>
  .psi-trajectory-visualizer {
    position: relative;
    background: white;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    overflow: hidden;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  }
  
  .controls {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 15px;
    background: #f8f9fa;
    border-bottom: 1px solid #e0e0e0;
    flex-wrap: wrap;
    gap: 10px;
  }
  
  .playback-controls {
    display: flex;
    align-items: center;
    gap: 10px;
  }
  
  .playback-btn, .reset-btn {
    padding: 5px 10px;
    border: 1px solid #ddd;
    background: white;
    border-radius: 4px;
    cursor: pointer;
    font-size: 12px;
  }
  
  .playback-btn:hover, .reset-btn:hover {
    background: #f0f0f0;
  }
  
  .playback-time {
    font-size: 12px;
    color: #666;
    font-family: monospace;
  }
  
  .zoom-controls {
    display: flex;
    align-items: center;
    gap: 5px;
  }
  
  .zoom-controls button {
    padding: 2px 6px;
    border: 1px solid #ddd;
    background: white;
    border-radius: 3px;
    cursor: pointer;
    font-size: 11px;
  }
  
  .zoom-level {
    font-size: 11px;
    color: #666;
    min-width: 30px;
    text-align: center;
  }
  
  .info {
    display: flex;
    gap: 15px;
    font-size: 12px;
    color: #666;
  }
  
  .timeline-svg {
    cursor: grab;
    display: block;
  }
  
  .timeline-svg:active {
    cursor: grabbing;
  }
  
  .concept-circle {
    cursor: pointer;
    transition: all 0.2s ease;
  }
  
  .concept-circle:hover {
    filter: brightness(1.1);
  }
  
  .drift-zone {
    cursor: help;
  }
  
  .tooltip {
    position: fixed;
    background: rgba(0, 0, 0, 0.9);
    color: white;
    padding: 10px;
    border-radius: 6px;
    font-size: 12px;
    max-width: 300px;
    z-index: 1000;
    pointer-events: none;
  }
  
  .tooltip-content {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  
  .concept-header {
    display: flex;
    align-items: center;
    gap: 8px;
    padding-bottom: 5px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.2);
  }
  
  .source-badge {
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 10px;
    color: white;
    font-weight: bold;
  }
  
  .concept-details {
    display: flex;
    flex-direction: column;
    gap: 3px;
  }
  
  .detail-row {
    display: flex;
    justify-content: space-between;
    gap: 10px;
  }
  
  .label {
    color: #bbb;
    font-weight: 500;
  }
  
  .value {
    color: white;
    font-family: monospace;
  }
  
  .concept-summary {
    font-style: italic;
    color: #ddd;
    font-size: 11px;
    line-height: 1.3;
  }
  
  .ghost-reflections {
    border-top: 1px solid rgba(255, 255, 255, 0.2);
    padding-top: 5px;
  }
  
  .ghost-reflection {
    margin-top: 3px;
    font-size: 11px;
  }
  
  .ghost-name {
    color: #9b59b6;
    font-weight: bold;
  }
  
  .ghost-message {
    color: #ddd;
  }
  
  .legend {
    position: absolute;
    top: 50px;
    right: 10px;
    background: rgba(255, 255, 255, 0.95);
    border: 1px solid #ddd;
    border-radius: 4px;
    padding: 10px;
    font-size: 11px;
    min-width: 120px;
  }
  
  .legend-section {
    margin-bottom: 10px;
  }
  
  .legend-section:last-child {
    margin-bottom: 0;
  }
  
  .legend-section h4 {
    margin: 0 0 5px 0;
    font-size: 11px;
    font-weight: bold;
    color: #333;
  }
  
  .legend-item {
    display: flex;
    align-items: center;
    gap: 5px;
    margin-bottom: 2px;
  }
  
  .legend-color {
    width: 12px;
    height: 12px;
    border-radius: 2px;
    border: 1px solid rgba(0, 0, 0, 0.1);
  }
  
  /* Responsive design */
  @media (max-width: 768px) {
    .controls {
      flex-direction: column;
      align-items: stretch;
    }
    
    .playback-controls,
    .zoom-controls,
    .info {
      justify-content: center;
    }
    
    .legend {
      position: static;
      margin-top: 10px;
      width: 100%;
    }
  }
</style>
