#!/usr/bin/env python3
"""
Phase 4: Persona Display Verification
Check Enola avatar and UI integration
"""

import os
import json
import requests
from pathlib import Path

def verify_persona_display():
    """Verify Enola persona and avatar display"""
    
    print("🎭 PHASE 4: PERSONA DISPLAY VERIFICATION")
    print("=" * 60)
    
    issues = []
    fixes_applied = []
    
    # 1. Check PersonaSelector component
    print("\n🎨 Checking PersonaSelector component...")
    persona_selector_path = Path("tori_ui_svelte/src/lib/components/PersonaSelector.svelte")
    
    if persona_selector_path.exists():
        content = persona_selector_path.read_text()
        
        if "name: 'Enola'" in content:
            print("   ✅ Enola found in PersonaSelector")
            fixes_applied.append("Enola already in PersonaSelector")
        else:
            print("   ❌ Enola NOT in PersonaSelector")
            issues.append("Add Enola to PersonaSelector component")
    else:
        print("   ❌ PersonaSelector.svelte not found")
        issues.append("PersonaSelector component missing")
    
    # 2. Check ghost persona store
    print("\n👻 Checking ghost persona store...")
    ghost_store_path = Path("tori_ui_svelte/src/lib/stores/ghostPersona.ts")
    
    if ghost_store_path.exists():
        content = ghost_store_path.read_text()
        
        checks = [
            ("persona: 'Enola'", "Enola as default persona"),
            ("activePersona: 'Enola'", "Enola as active persona"),
            ('"Enola": new GhostPersona("Enola")', "Enola in ghost registry")
        ]
        
        for check_str, description in checks:
            if check_str in content:
                print(f"   ✅ {description}")
                fixes_applied.append(description)
            else:
                print(f"   ❌ {description} NOT found")
                issues.append(f"Missing: {description}")
    else:
        print("   ❌ ghostPersona.ts not found")
        issues.append("Ghost persona store missing")
    
    # 3. Check for avatar display component
    print("\n🎬 Checking avatar display components...")
    
    # Check if HologramPersonaDisplay exists (from our fixes)
    hologram_display_path = Path("tori_ui_svelte/src/lib/components/HologramPersonaDisplay.svelte")
    
    if hologram_display_path.exists():
        print("   ✅ HologramPersonaDisplay.svelte exists")
        fixes_applied.append("HologramPersonaDisplay component created")
    else:
        print("   ❌ HologramPersonaDisplay.svelte not found")
        issues.append("Create HologramPersonaDisplay component")
    
    # 4. Check API avatar endpoints
    print("\n🌐 Checking API avatar endpoints...")
    
    try:
        # Test avatar state endpoint
        response = requests.get("http://localhost:8002/api/avatar/state", timeout=2)
        if response.status_code == 200:
            print("   ✅ Avatar state endpoint exists")
            avatar_data = response.json()
            print(f"   📊 Current persona: {avatar_data.get('persona', 'unknown')}")
        elif response.status_code == 404:
            print("   ❌ Avatar endpoints not found")
            issues.append("Add avatar endpoints to API")
    except requests.exceptions.ConnectionError:
        print("   ⚠️  Cannot connect to API - is TORI running?")
    except Exception as e:
        print(f"   ⚠️  Avatar endpoint check failed: {e}")
    
    # 5. Check for frontend integration
    print("\n🔌 Checking frontend integration...")
    
    # Check main layout or app component
    app_paths = [
        Path("tori_ui_svelte/src/routes/+page.svelte"),
        Path("tori_ui_svelte/src/routes/+layout.svelte"),
        Path("tori_ui_svelte/src/App.svelte")
    ]
    
    hologram_integrated = False
    for app_path in app_paths:
        if app_path.exists():
            content = app_path.read_text()
            if "HologramPersonaDisplay" in content:
                print(f"   ✅ HologramPersonaDisplay integrated in {app_path.name}")
                hologram_integrated = True
                break
    
    if not hologram_integrated:
        print("   ⚠️  HologramPersonaDisplay not integrated in UI")
        issues.append("Add HologramPersonaDisplay to main layout")
    
    # 6. Create integration guide
    print("\n📝 Creating integration guide...")
    
    integration_guide = '''# Enola Avatar Integration Guide

## 1. Add HologramPersonaDisplay to your layout

In your main layout file (e.g., +layout.svelte or +page.svelte), add:

```svelte
<script>
  import HologramPersonaDisplay from '$lib/components/HologramPersonaDisplay.svelte';
</script>

<!-- Add where you want the hologram to appear -->
<div class="hologram-container">
  <HologramPersonaDisplay />
</div>
```

## 2. Ensure Enola is in PersonaSelector

The PersonaSelector should include Enola with these properties:
```javascript
{
  name: 'Enola',
  ψ: 'investigative',
  ε: [0.9, 0.5, 0.8],
  τ: 0.75,
  φ: 2.718,
  color: '#2563eb',
  description: 'Systematic investigation and discovery',
  mood: 'analytical'
}
```

## 3. Add avatar API endpoints

In prajna_api.py, add:
```python
@app.get("/api/avatar/state")
async def get_avatar_state():
    return {
        "persona": "Enola",
        "state": "idle",
        "mood": "analytical"
    }

@app.websocket("/api/avatar/updates")
async def avatar_updates(websocket: WebSocket):
    await websocket.accept()
    # Send avatar updates
```

## 4. Test the integration

1. Start TORI: `poetry run python enhanced_launcher.py`
2. Open the frontend
3. Verify Enola appears in the persona selector
4. Check that the hologram display shows Enola
'''
    
    guide_path = Path("ENOLA_INTEGRATION_GUIDE.md")
    guide_path.write_text(integration_guide)
    print("   ✅ Created ENOLA_INTEGRATION_GUIDE.md")
    
    # 7. Summary
    print("\n" + "=" * 60)
    
    if fixes_applied:
        print("✅ FIXES ALREADY APPLIED:")
        for fix in fixes_applied:
            print(f"   • {fix}")
    
    if issues:
        print("\n⚠️  ISSUES TO FIX:")
        for issue in issues:
            print(f"   • {issue}")
        
        print("\n📝 To fix these issues:")
        print("   1. Run the fix scripts from fixes_2025_01_19/")
        print("   2. Follow ENOLA_INTEGRATION_GUIDE.md")
        print("   3. Restart TORI after applying fixes")
    else:
        print("\n✅ PHASE 4 COMPLETE: Enola persona ready!")
        print("\n💡 If Enola is not visible:")
        print("   1. Check ENOLA_INTEGRATION_GUIDE.md")
        print("   2. Ensure HologramPersonaDisplay is added to your layout")
        print("   3. Clear browser cache and reload")
    
    return len(issues) == 0

if __name__ == "__main__":
    verify_persona_display()
