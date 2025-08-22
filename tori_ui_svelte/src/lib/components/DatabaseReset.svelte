<script lang="ts">
  import { toriStorage } from '$lib/services/toriStorage';
  
  async function resetDatabase() {
    if (confirm('This will clear all local TORI data. Continue?')) {
      try {
        await toriStorage.clearAll();
        alert('Database reset successfully!');
        window.location.reload();
      } catch (error) {
  const msg = error instanceof Error ? error.message : String(error);
        console.error('Failed to reset database:', error);
        alert('Failed to reset database. See console for details.');
      
}
    }
  }
  
  // Check for database errors on startup
  if (typeof window !== 'undefined') {
    const dbError = localStorage.getItem('tori-db-error');
    if (dbError) {
      const error = JSON.parse(dbError);
      console.warn('Previous database error detected:', error);
      // You could auto-prompt for reset here
    }
  }
</script>

<button 
  on:click={resetDatabase}
  class="text-xs text-gray-500 hover:text-gray-700 underline"
>
  Reset Database
</button>
