#!/usr/bin/env python3
"""
Fix persona display to be clearer
"""

import os
import re
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def fix_persona_display():
    """Make the persona display clearer with better visual indicators"""
    
    project_root = Path(r"{PROJECT_ROOT}")
    navigation_panel = project_root / "tori_ui_svelte" / "src" / "lib" / "components" / "NavigationPanel.svelte"
    
    print("🎨 Fixing persona display clarity...")
    
    with open(navigation_panel, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Update the nav header to be clearer
    new_nav_header = """  <div class="nav-header">
    <div class="persona-section">
      <h3 class="nav-title">
        <span class="title-icon">🎭</span>
        Current Persona
      </h3>
      {#if currentPersona}
        <div class="current-persona-display">
          <div class="persona-badge" style="background-color: rgb({currentPersona.color.r * 255}, {currentPersona.color.g * 255}, {currentPersona.color.b * 255})">
            {currentPersona.name.charAt(0)}
          </div>
          <div class="persona-info">
            <span class="persona-name">{currentPersona.name}</span>
            <span class="persona-description">{currentPersona.description}</span>
          </div>
        </div>
      {/if}
    </div>
  </div>"""
    
    # Replace the nav header section
    content = re.sub(
        r'<div class="nav-header">.*?</div>\s*{/if}\s*</div>',
        new_nav_header,
        content,
        flags=re.DOTALL
    )
    
    # Update styles for better clarity
    additional_styles = """
  .persona-section {
    width: 100%;
  }
  
  .nav-title {
    display: flex;
    align-items: center;
    gap: var(--space-1);
    font-size: var(--text-sm);
    font-weight: 600;
    color: var(--color-text-secondary);
    margin: 0 0 var(--space-2) 0;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  
  .title-icon {
    font-size: var(--text-lg);
  }
  
  .current-persona-display {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    padding: var(--space-2);
    background: var(--color-base);
    border: 2px solid var(--color-border);
    border-radius: var(--border-radius);
    transition: all 0.2s ease;
  }
  
  .current-persona-display:hover {
    border-color: var(--color-accent);
    transform: translateY(-1px);
    box-shadow: var(--shadow-md);
  }
  
  :global(.dark) .current-persona-display {
    background: rgba(255, 255, 255, 0.05);
    border-color: rgba(255, 255, 255, 0.1);
  }
  
  :global(.dark) .current-persona-display:hover {
    border-color: rgba(138, 43, 226, 0.3);
  }
  
  .persona-badge {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: var(--text-lg);
    font-weight: 600;
    box-shadow: var(--shadow-sm);
    flex-shrink: 0;
  }
  
  .persona-info {
    display: flex;
    flex-direction: column;
    gap: 2px;
    flex: 1;
    min-width: 0;
  }
  
  .persona-name {
    font-size: var(--text-base);
    font-weight: 600;
    color: var(--color-text-primary);
  }
  
  .persona-description {
    font-size: var(--text-xs);
    color: var(--color-text-secondary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  
  :global(.dark) .persona-name {
    color: #e0e0e0;
  }
  
  :global(.dark) .persona-description {
    color: #999;
  }"""
    
    # Find where to insert the new styles
    if ".persona-section {" not in content:
        content = content.replace(
            "/* Responsive adjustments */",
            additional_styles + "\n\n  /* Responsive adjustments */"
        )
    
    # Make the nav items clearer too
    content = content.replace(
        "{ id: 'select', label: 'Select', icon: '👤', action: 'selectPersona' },",
        "{ id: 'select', label: 'Change Persona', icon: '🔄', action: 'selectPersona' },"
    )
    
    content = content.replace(
        "{ id: 'create', label: 'Create', icon: '➕', action: 'createPersona' },",
        "{ id: 'create', label: 'New Persona', icon: '✨', action: 'createPersona' },"
    )
    
    content = content.replace(
        "{ id: 'vault', label: 'Vault', icon: '📷', action: 'toggleWebcam' }",
        "{ id: 'vault', label: 'Webcam', icon: '📷', action: 'toggleWebcam' }"
    )
    
    with open(navigation_panel, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Fixed persona display clarity")
    
    # Also update the MemoryPanel to show better hologram labels
    memory_panel = project_root / "tori_ui_svelte" / "src" / "lib" / "components" / "MemoryPanel.svelte"
    
    with open(memory_panel, 'r', encoding='utf-8') as f:
        memory_content = f.read()
    
    # Update the hologram control labels
    memory_content = memory_content.replace(
        '<span class="control-label">Audio</span>',
        '<span class="control-label">Voice Output</span>'
    )
    
    memory_content = memory_content.replace(
        '<span class="control-label">Video</span>',
        '<span class="control-label">Hologram</span>'
    )
    
    with open(memory_panel, 'w', encoding='utf-8') as f:
        f.write(memory_content)
    
    print("✅ Updated hologram button labels")
    
    print("\n🎨 UI clarity improvements complete!")
    print("\nChanges made:")
    print("  1. Added 'Current Persona' header with icon")
    print("  2. Persona display now shows name AND description")
    print("  3. Larger, clearer persona badge")
    print("  4. Hover effects for better interactivity")
    print("  5. Renamed buttons for clarity:")
    print("     - 'Select' → 'Change Persona'")
    print("     - 'Create' → 'New Persona'")
    print("     - 'Vault' → 'Webcam'")
    print("     - 'Audio' → 'Voice Output'")
    print("     - 'Video' → 'Hologram'")

if __name__ == "__main__":
    fix_persona_display()
