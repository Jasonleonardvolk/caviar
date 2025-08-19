#!/usr/bin/env python3
"""
Fix 1: Add Enola to PersonaSelector component
This script adds the Enola persona to the PersonaSelector.svelte file
"""

import os
import re
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def fix_persona_selector():
    """Add Enola to the PersonaSelector component"""
    
    # Path to the PersonaSelector component
    persona_selector_path = Path(r"{PROJECT_ROOT}\tori_ui_svelte\src\lib\components\PersonaSelector.svelte")
    
    if not persona_selector_path.exists():
        print(f"❌ PersonaSelector.svelte not found at {persona_selector_path}")
        return False
    
    # Read the current content
    with open(persona_selector_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Define Enola's persona configuration based on ENOLA_PERSONA_SPEC.md
    enola_config = """    {
      name: 'Enola',
      ψ: 'investigative',   // Cognitive mode: systematic investigation
      ε: [0.9, 0.5, 0.8],   // Emotional palette: [focused, balanced, determined]
      τ: 0.75,              // Temporal bias: methodical pacing
      φ: 2.718,             // Phase seed: e (natural harmony)
      color: '#2563eb',     // Investigation blue
      description: 'Systematic investigation and discovery',
      mood: 'analytical'
    },
"""
    
    # Find the personas array
    personas_pattern = r'const personas = \[(.*?)\];'
    personas_match = re.search(personas_pattern, content, re.DOTALL)
    
    if not personas_match:
        print("❌ Could not find personas array in PersonaSelector.svelte")
        return False
    
    # Check if Enola already exists
    if 'Enola' in content:
        print("⚠️ Enola already exists in PersonaSelector.svelte")
        return True
    
    # Insert Enola at the beginning of the personas array
    personas_content = personas_match.group(1)
    
    # Find the first persona definition
    first_persona_match = re.search(r'(\s*\{)', personas_content)
    if first_persona_match:
        insert_position = first_persona_match.start()
        
        # Insert Enola config before the first persona
        new_personas_content = (
            personas_content[:insert_position] + 
            enola_config + 
            personas_content[insert_position:]
        )
        
        # Replace the old personas array with the new one
        new_content = content.replace(
            f'const personas = [{personas_content}];',
            f'const personas = [{new_personas_content}];'
        )
        
        # Also update the default selectedPersona to Enola
        new_content = re.sub(
            r"let selectedPersona = \$ghostPersona\.activePersona \|\| '[^']*';",
            "let selectedPersona = $ghostPersona.activePersona || 'Enola';",
            new_content
        )
        
        # Write the updated content
        with open(persona_selector_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print("✅ Successfully added Enola to PersonaSelector.svelte")
        print("✅ Set Enola as the default selectedPersona")
        return True
    else:
        print("❌ Could not find insertion point for Enola")
        return False

if __name__ == "__main__":
    success = fix_persona_selector()
    if success:
        print("\n✨ Enola has been added to the PersonaSelector!")
        print("🔍 Enola's characteristics:")
        print("   - Investigative cognitive mode")
        print("   - Systematic exploration and discovery")
        print("   - Investigation blue color (#2563eb)")
        print("   - Analytical mood")
    else:
        print("\n❌ Failed to add Enola to PersonaSelector")
