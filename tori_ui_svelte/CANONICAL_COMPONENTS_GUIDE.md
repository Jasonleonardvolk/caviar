# Canonical Components Guide - TORI UI
## Official Component Status Reference

---

## Holographic Display Components

### ✅ **HolographicDisplay.svelte** - CANONICAL
**Status:** Primary/Production Component  
**Path:** `src\lib\components\HolographicDisplay.svelte`  
**Evidence:**
- Drive audits: "Main UI component for hologram. Keep."
- Build logs show active HMR updates during development
- Component map routes rendering pipeline to "HolographicDisplay Rendering"
- **Decision:** This is THE holographic display component

### 🟨 **HolographicDisplayEnhanced.svelte** - EXPERIMENTAL
**Status:** Optional/Advanced (Not Default)  
**Path:** `src\lib\components\HolographicDisplayEnhanced.svelte`  
**Evidence:**
- Drive notes: "Active — in use (optional). Not yet the default, but serves as an upgraded version. Keep for now."
- **Decision:** Keep as experimental feature, but not primary implementation

### ❌ **HolographicDisplay_FIXED.svelte** - NON-CANONICAL
**Status:** Failed Fix/Abandoned  
**Path:** `src\lib\components\HolographicDisplay_FIXED.svelte`  
**Evidence:**
- Not referenced in any "Active" listings
- Appears only alongside other broken/backup files
- Currently stubbed to prevent compilation errors
- **Decision:** Archive or remove in Phase 6

---

## Phase 6 Action Plan

### 1. Clean Up Non-Canonical Files
```powershell
# Archive non-canonical files
mkdir src\lib\components\archive
move src\lib\components\HolographicDisplay_FIXED.svelte src\lib\components\archive\
move src\lib\components\HolographicDisplay.svelte.broken src\lib\components\archive\
```

### 2. Handle Experimental Components
- Keep `HolographicDisplayEnhanced.svelte` but ensure it doesn't break builds
- Add feature flag if needed to conditionally load
- Document as experimental in component header

### 3. Stub Requirements
For any component that must remain but isn't ready for production:
```svelte
<script lang="ts">
  // STUB: This component is experimental/incomplete
  export let showStats = true;
</script>

<div class="component-stub" aria-hidden="true">
  <!-- Minimal DOM to prevent errors -->
</div>
```

### 4. Import Path Standards
- Canonical components: Direct import
- Experimental: Behind feature flags or dynamic imports
- Stubs: Never imported in production paths

---

## Component Status Key
- ✅ **CANONICAL** - Production ready, actively used
- 🟨 **EXPERIMENTAL** - Optional, advanced features, not default
- ❌ **NON-CANONICAL** - Failed attempts, should be archived
- 🔧 **STUB** - Placeholder to prevent build errors

---

## Build Verification Checklist
- [ ] Canonical components compile without errors
- [ ] Experimental components don't break main build
- [ ] Non-canonical files are archived/removed
- [ ] No import paths point to non-canonical components
- [ ] HMR works with canonical components

---

## References
- Drive Audits: Component inventory and status tracking
- Build Logs: Vite/Svelte HMR activity patterns
- Component Map: Rendering pipeline architecture

---

*Last Updated: August 16, 2025*  
*Phase: 5 → 6 Transition*
