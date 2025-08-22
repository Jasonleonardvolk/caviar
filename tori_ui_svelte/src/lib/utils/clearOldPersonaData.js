/**
 * Utility to clear old persona data when upgrading to ENOLA default
 * Run this once on deployment to ensure all users get ENOLA on first load
 */

export function clearOldPersonaData() {
  if (typeof window === 'undefined') return;
  
  // Check if we've already migrated
  const migrated = localStorage.getItem('enola_migration_complete');
  if (migrated) return;
  
  console.log('🧹 Clearing old persona data for ENOLA migration...');
  
  // Clear any old persona selections
  localStorage.removeItem('selectedGhostPersona');
  localStorage.removeItem('selectedPersona');
  localStorage.removeItem('lastPersona');
  localStorage.removeItem('currentGhost');
  
  // Mark migration as complete
  localStorage.setItem('enola_migration_complete', 'true');
  localStorage.setItem('personaVersion', 'v2_enola_default');
  
  console.log('✅ ENOLA migration complete - all users will see ENOLA on next load');
}

// Auto-run on import if in browser
if (typeof window !== 'undefined') {
  clearOldPersonaData();
}