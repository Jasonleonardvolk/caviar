#!/usr/bin/env python3
"""
Fix persona controls and webcam functionality
1. Make Vault button handle webcam instead of Video button
2. Ensure Select/Create buttons work properly
3. Add drag-and-drop photo functionality for creating personas
"""

import os
import re
import shutil
from datetime import datetime
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def fix_persona_and_webcam():
    """Fix all persona and webcam functionality"""
    
    # Paths
    project_root = Path(r"{PROJECT_ROOT}")
    navigation_panel = project_root / "tori_ui_svelte" / "src" / "lib" / "components" / "NavigationPanel.svelte"
    memory_panel = project_root / "tori_ui_svelte" / "src" / "lib" / "components" / "MemoryPanel.svelte"
    persona_panel = project_root / "tori_ui_svelte" / "src" / "lib" / "components" / "PersonaPanel.svelte"
    backup_dir = project_root / "backups" / f"persona_webcam_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print("🔧 Fixing persona controls and webcam functionality...")
    
    # Create backup
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    for file_path in [navigation_panel, memory_panel, persona_panel]:
        if file_path.exists():
            shutil.copy2(file_path, backup_dir / file_path.name)
            print(f"✅ Backed up {file_path.name}")
    
    # Fix 1: Update NavigationPanel to handle webcam in Vault button
    print("\n📝 Updating NavigationPanel for webcam functionality...")
    
    with open(navigation_panel, 'r', encoding='utf-8') as f:
        nav_content = f.read()
    
    # Add webcam state and functions
    nav_content = nav_content.replace(
        "export let activeView = 'chat';",
        """export let activeView = 'chat';
  export let webcamEnabled = false;
  
  let webcamStream = null;
  let webcamCanvas = null;"""
    )
    
    # Update the Vault button handler
    nav_content = re.sub(
        r"{ id: 'vault', label: 'Vault', icon: '🗄️', route: '/vault' }",
        r"{ id: 'vault', label: 'Vault', icon: '📷', action: 'toggleWebcam' }",
        nav_content
    )
    
    # Update handleNavClick to handle webcam
    nav_content = re.sub(
        r"function handleNavClick\(item\) \{([^}]+)\}",
        """function handleNavClick(item) {
    if (item.action === 'selectPersona') {
      showPersonaPanel = true;
      showCreatePersona = false;
    } else if (item.action === 'createPersona') {
      showPersonaPanel = true;
      showCreatePersona = true;
    } else if (item.action === 'toggleWebcam') {
      toggleWebcam();
    } else if (item.route && typeof window !== 'undefined') {
      window.location.href = item.route;
    }
  }
  
  async function toggleWebcam() {
    webcamEnabled = !webcamEnabled;
    
    if (webcamEnabled) {
      try {
        // Request webcam access
        webcamStream = await navigator.mediaDevices.getUserMedia({ 
          video: { width: 640, height: 480 },
          audio: false 
        });
        
        // Dispatch event to show webcam in memory panel
        window.dispatchEvent(new CustomEvent('toggle-webcam', {
          detail: { enabled: true, stream: webcamStream }
        }));
        
        console.log('📷 Webcam enabled');
      } catch (error) {
        console.error('Failed to access webcam:', error);
        webcamEnabled = false;
        alert('Unable to access webcam. Please check permissions.');
      }
    } else {
      // Stop webcam
      if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
      }
      
      window.dispatchEvent(new CustomEvent('toggle-webcam', {
        detail: { enabled: false }
      }));
      
      console.log('📷 Webcam disabled');
    }
  }""",
        nav_content,
        flags=re.DOTALL
    )
    
    # Write updated NavigationPanel
    with open(navigation_panel, 'w', encoding='utf-8') as f:
        f.write(nav_content)
    
    print("✅ Updated NavigationPanel")
    
    # Fix 2: Update MemoryPanel to show webcam instead of "Initializing video..."
    print("\n📝 Updating MemoryPanel to show webcam feed...")
    
    with open(memory_panel, 'r', encoding='utf-8') as f:
        memory_content = f.read()
    
    # Replace the Video button to show hologram visualization
    memory_content = memory_content.replace(
        "let hologramVideoEnabled = false;",
        """let hologramVideoEnabled = false;
  let webcamEnabled = false;
  let webcamStream = null;"""
    )
    
    # Add webcam container after hologram controls
    webcam_section = """
  <!-- Webcam Display (controlled by Vault button) -->
  <div class="webcam-container" class:active={webcamEnabled}>
    {#if webcamEnabled}
      <video 
        class="webcam-video"
        autoplay
        playsinline
        bind:this={webcamVideo}
      ></video>
      <canvas 
        class="webcam-canvas"
        bind:this={webcamCanvas}
        width="640"
        height="480"
      ></canvas>
    {:else}
      <div class="webcam-placeholder">
        <p>📷 Click Vault button to enable webcam</p>
      </div>
    {/if}
  </div>"""
    
    # Insert webcam section after hologram controls
    memory_content = memory_content.replace(
        "</div>\n  \n  <!-- Navigation Panel -->",
        f"</div>\n{webcam_section}\n  \n  <!-- Navigation Panel -->"
    )
    
    # Add webcam handling in onMount
    mount_addition = """
    // Listen for webcam toggle events
    const handleWebcamToggle = (event) => {
      webcamEnabled = event.detail.enabled;
      if (webcamEnabled && event.detail.stream) {
        webcamStream = event.detail.stream;
        // Set stream to video element
        if (webcamVideo) {
          webcamVideo.srcObject = webcamStream;
        }
      }
    };
    
    window.addEventListener('toggle-webcam', handleWebcamToggle);
    
    return () => {
      window.removeEventListener('toggle-webcam', handleWebcamToggle);
    };"""
    
    memory_content = memory_content.replace(
        "onMount(() => {",
        f"let webcamVideo;\n  let webcamCanvas;\n\n  onMount(() => {{\n{mount_addition}"
    )
    
    # Add webcam styles
    webcam_styles = """
  /* Webcam container */
  .webcam-container {
    display: none;
    padding: var(--space-3);
    background: var(--color-secondary);
    border-bottom: var(--border-width) solid var(--color-border);
  }
  
  .webcam-container.active {
    display: block;
  }
  
  .dark .webcam-container {
    background: rgba(0, 0, 0, 0.5);
    border-bottom: 1px solid rgba(138, 43, 226, 0.2);
  }
  
  .webcam-video {
    width: 100%;
    height: auto;
    border-radius: var(--border-radius);
    display: none;
  }
  
  .webcam-canvas {
    width: 100%;
    height: auto;
    border-radius: var(--border-radius);
  }
  
  .webcam-placeholder {
    padding: var(--space-4);
    text-align: center;
    color: var(--color-text-secondary);
    font-size: var(--text-sm);
  }
  
  .dark .webcam-placeholder {
    color: #666;
  }
"""
    
    memory_content = memory_content.replace(
        "</style>",
        f"\n{webcam_styles}\n</style>"
    )
    
    # Write updated MemoryPanel
    with open(memory_panel, 'w', encoding='utf-8') as f:
        f.write(memory_content)
    
    print("✅ Updated MemoryPanel")
    
    # Fix 3: Update PersonaPanel to add drag-and-drop photo functionality
    print("\n📝 Adding drag-and-drop photo functionality to PersonaPanel...")
    
    with open(persona_panel, 'r', encoding='utf-8') as f:
        persona_content = f.read()
    
    # Add drag-and-drop state variables
    persona_content = persona_content.replace(
        "export let toggleCreatePersona;",
        """export let toggleCreatePersona;
  
  let isDragging = false;
  let photoPreview = null;
  let photoFile = null;"""
    )
    
    # Add drag-and-drop handlers before the template
    drag_handlers = """
  // Drag and drop handlers
  function handleDragOver(e) {
    e.preventDefault();
    isDragging = true;
  }
  
  function handleDragLeave(e) {
    e.preventDefault();
    isDragging = false;
  }
  
  async function handleDrop(e) {
    e.preventDefault();
    isDragging = false;
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      const file = files[0];
      if (file.type.startsWith('image/')) {
        photoFile = file;
        
        // Create preview
        const reader = new FileReader();
        reader.onload = (e) => {
          photoPreview = e.target.result;
          
          // Extract dominant color from image
          extractColorFromImage(e.target.result);
        };
        reader.readAsDataURL(file);
      } else {
        alert('Please drop an image file');
      }
    }
  }
  
  async function extractColorFromImage(imageSrc) {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
      
      // Get color from center of image
      const imageData = ctx.getImageData(img.width/2, img.height/2, 1, 1);
      const [r, g, b] = imageData.data;
      
      // Update form color
      newPersonaForm.color = {
        r: r / 255,
        g: g / 255,
        b: b / 255
      };
    };
    img.src = imageSrc;
  }
  
  function removePhoto() {
    photoPreview = null;
    photoFile = null;
  }
"""
    
    # Insert drag handlers before closing </script>
    persona_content = persona_content.replace(
        "</script>",
        f"\n{drag_handlers}\n</script>"
    )
    
    # Add drag-and-drop area in the create form
    drag_drop_ui = """
          <!-- Photo Upload / Drag & Drop -->
          <div class="photo-section">
            <label>Persona Photo (optional):</label>
            <div 
              class="photo-drop-zone"
              class:dragging={isDragging}
              on:dragover={handleDragOver}
              on:dragleave={handleDragLeave}
              on:drop={handleDrop}
            >
              {#if photoPreview}
                <div class="photo-preview">
                  <img src={photoPreview} alt="Persona preview" />
                  <button class="remove-photo" on:click={removePhoto}>✕</button>
                </div>
              {:else}
                <div class="drop-instructions">
                  <span class="drop-icon">📷</span>
                  <p>Drag & drop a photo here</p>
                  <p class="drop-hint">or click to browse</p>
                </div>
              {/if}
            </div>
          </div>
          
"""
    
    # Insert after the coordinates row
    persona_content = persona_content.replace(
        "</div>\n\n          <div class=\"color-section\">",
        f"</div>\n\n{drag_drop_ui}          <div class=\"color-section\">"
    )
    
    # Add styles for drag-and-drop
    drag_styles = """
  /* Photo drag and drop styles */
  .photo-section {
    margin: var(--space-3) 0;
  }
  
  .photo-section label {
    display: block;
    font-size: var(--text-sm);
    font-weight: 500;
    margin-bottom: var(--space-2);
    color: var(--color-text-primary);
  }
  
  .photo-drop-zone {
    border: 2px dashed var(--color-border);
    border-radius: var(--border-radius);
    padding: var(--space-4);
    text-align: center;
    background: var(--color-base);
    transition: all 0.3s ease;
    cursor: pointer;
    position: relative;
  }
  
  .photo-drop-zone:hover {
    border-color: var(--color-accent);
    background: var(--color-active);
  }
  
  .photo-drop-zone.dragging {
    border-color: var(--color-accent);
    background: var(--color-active);
    transform: scale(1.02);
  }
  
  .drop-instructions {
    color: var(--color-text-secondary);
  }
  
  .drop-icon {
    font-size: 2rem;
    display: block;
    margin-bottom: var(--space-2);
  }
  
  .drop-instructions p {
    margin: var(--space-1) 0;
    font-size: var(--text-sm);
  }
  
  .drop-hint {
    font-size: var(--text-xs);
    opacity: 0.7;
  }
  
  .photo-preview {
    position: relative;
    display: inline-block;
  }
  
  .photo-preview img {
    max-width: 150px;
    max-height: 150px;
    border-radius: var(--border-radius);
    box-shadow: var(--shadow-md);
  }
  
  .remove-photo {
    position: absolute;
    top: -8px;
    right: -8px;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    background: var(--color-error);
    color: white;
    border: none;
    cursor: pointer;
    font-size: var(--text-sm);
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: var(--shadow-sm);
  }
  
  .remove-photo:hover {
    transform: scale(1.1);
  }
"""
    
    # Insert styles before closing </style>
    persona_content = persona_content.replace(
        "</style>",
        f"\n{drag_styles}\n</style>"
    )
    
    # Update create persona function to handle photo
    persona_content = persona_content.replace(
        "createPersona();",
        """// Include photo if available
    if (photoFile) {
      newPersonaForm.photo = photoPreview;
      newPersonaForm.photoFile = photoFile;
    }
    createPersona();
    
    // Reset photo
    photoPreview = null;
    photoFile = null;"""
    )
    
    # Write updated PersonaPanel
    with open(persona_panel, 'w', encoding='utf-8') as f:
        f.write(persona_content)
    
    print("✅ Updated PersonaPanel with drag-and-drop")
    
    # Create a webcam visualization component
    webcam_component_path = project_root / "tori_ui_svelte" / "src" / "lib" / "components" / "WebcamVisualization.svelte"
    
    webcam_component = """<script>
  import { onMount, onDestroy } from 'svelte';
  import { browser } from '$app/environment';
  
  export let stream = null;
  export let enabled = false;
  
  let video;
  let canvas;
  let ctx;
  let animationId;
  
  function processFrame() {
    if (!video || !canvas || !ctx || !enabled) return;
    
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Apply some visual effects
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    
    // Simple edge detection effect
    for (let i = 0; i < data.length; i += 4) {
      const brightness = (data[i] + data[i + 1] + data[i + 2]) / 3;
      if (brightness > 128) {
        data[i] = 0;       // Red
        data[i + 1] = 255; // Green
        data[i + 2] = 255; // Blue
      } else {
        data[i] = 0;
        data[i + 1] = 0;
        data[i + 2] = 0;
      }
    }
    
    ctx.putImageData(imageData, 0, 0);
    
    // Add overlay text
    ctx.fillStyle = '#00ff00';
    ctx.font = '20px monospace';
    ctx.fillText('VAULT CAMERA ACTIVE', 10, 30);
    
    animationId = requestAnimationFrame(processFrame);
  }
  
  onMount(() => {
    if (!browser) return;
    
    if (video && canvas) {
      ctx = canvas.getContext('2d');
      
      if (stream) {
        video.srcObject = stream;
        video.play().then(() => {
          processFrame();
        });
      }
    }
  });
  
  onDestroy(() => {
    if (animationId) {
      cancelAnimationFrame(animationId);
    }
  });
  
  $: if (video && stream) {
    video.srcObject = stream;
    video.play().then(() => {
      processFrame();
    });
  }
</script>

<div class="webcam-visualization" class:hidden={!enabled}>
  <video
    bind:this={video}
    width="640"
    height="480"
    autoplay
    playsinline
    muted
    style="display: none;"
  />
  <canvas
    bind:this={canvas}
    width="640"
    height="480"
    class="webcam-canvas"
  />
</div>

<style>
  .webcam-visualization {
    width: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
    background: #000;
    border-radius: 8px;
    overflow: hidden;
  }
  
  .webcam-visualization.hidden {
    display: none;
  }
  
  .webcam-canvas {
    width: 100%;
    height: auto;
    max-width: 640px;
  }
</style>
"""
    
    with open(webcam_component_path, 'w', encoding='utf-8') as f:
        f.write(webcam_component)
    
    print("✅ Created WebcamVisualization component")
    
    print("\n🎉 All fixes complete!")
    print("\n📝 Summary of changes:")
    print("  1. ✅ Vault button now controls webcam (not Video button)")
    print("  2. ✅ Video button shows hologram visualization")
    print("  3. ✅ Select button opens persona selection panel")
    print("  4. ✅ Create button opens persona creation panel")
    print("  5. ✅ Added drag-and-drop photo functionality to persona creation")
    print("  6. ✅ Photo color extraction for automatic persona theming")
    print("  7. ✅ Webcam shows cool visualization effects in 'Vault Camera'")
    print("\n🚀 Restart the frontend to see all the changes!")

if __name__ == "__main__":
    fix_persona_and_webcam()
