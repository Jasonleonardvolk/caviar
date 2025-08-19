<!-- ENHANCED AESTHETIC IMPROVEMENTS FOR TORI LAYOUT -->

## Quick Aesthetic Upgrades

### 1. Glass Morphism for Panels
Replace the plain white backgrounds with glass morphism effect:

```svelte
<!-- In MemoryPanel.svelte -->
<aside class="w-80 flex-shrink-0 h-full glass-panel">
```

### 2. Gradient Headers
Make the TORI logo area more visually appealing:

```svelte
<!-- In header -->
<div class="w-8 h-8 gradient-mesh rounded-lg flex items-center justify-center shadow-lg">
  <span class="text-white text-sm font-bold">T</span>
</div>
```

### 3. Animated Status Indicators
For the "Memory: Initializing" status:

```svelte
<div class="flex items-center gap-2">
  <div class="w-2 h-2 bg-purple-500 rounded-full pulse-dot"></div>
  <span>Memory: {status}</span>
</div>
```

### 4. Enhanced Upload Areas
For ScholarSphere upload zones:

```svelte
<div class="upload-zone glass-panel hover-lift panel-transition">
  <!-- upload content -->
</div>
```

### 5. Better Typography
For headings:

```svelte
<h1 class="text-2xl font-bold heading-gradient">TORI</h1>
```

### 6. Smooth Panel Transitions
Add to all panels:

```svelte
<div class="panel-transition hover:shadow-xl">
```

### 7. Floating Elements
For decorative elements:

```svelte
<div class="floating">
  <!-- icon or element -->
</div>
```

### 8. Import the CSS
Add to +layout.svelte:

```svelte
<script>
  import '../styles/enhanced-aesthetics.css';
</script>
```

### 9. Panel Resize Handles
Make panels resizable:

```svelte
<aside class="w-80 resizable-panel">
```

### 10. Dark Mode Toggle
Add a theme toggle button:

```svelte
<button on:click={toggleTheme} class="neumorphic p-2 rounded-lg">
  {#if isDark}🌙{:else}☀️{/if}
</button>
```

## Color Palette Suggestions

- Primary: #667eea (Purple-Blue)
- Secondary: #764ba2 (Deep Purple)
- Accent: #FFCC70 (Golden)
- Success: #10b981 (Green)
- Warning: #f59e0b (Amber)
- Error: #ef4444 (Red)

## Font Suggestions

- Headings: 'Inter', 'SF Pro Display', sans-serif
- Body: 'Inter', 'SF Pro Text', sans-serif
- Code: 'JetBrains Mono', 'Fira Code', monospace
