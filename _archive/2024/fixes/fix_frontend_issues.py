#!/usr/bin/env python3
"""
Fix TORI frontend issues:
1. IndexedDB corruption
2. Default hologram to ENOLA
"""

import json
from pathlib import Path
import re

def fix_tori_storage():
    """Add better error handling and recovery to toriStorage.ts"""
    
    storage_path = Path("tori_ui_svelte/src/lib/services/toriStorage.ts")
    
    if not storage_path.exists():
        print(f"❌ File not found: {storage_path}")
        return False
    
    with open(storage_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add a database reset method
    reset_method = '''
  /**
   * Reset the database - useful for recovery from corruption
   */
  async resetDatabase(): Promise<void> {
    console.warn('🔄 Resetting TORI database...');
    
    // Close existing connection
    if (this.db) {
      this.db.close();
      this.db = null;
    }
    
    // Delete the database
    try {
      await new Promise<void>((resolve, reject) => {
        const deleteReq = indexedDB.deleteDatabase(this.config.dbName);
        deleteReq.onsuccess = () => resolve();
        deleteReq.onerror = () => reject(deleteReq.error);
      });
      console.log('✅ Database deleted successfully');
    } catch (error) {
      console.error('Failed to delete database:', error);
    }
    
    // Reset state
    this.isInitialized = false;
    this.initializationAttempted = false;
    this.initializationPromise = null;
    
    // Reinitialize
    await this.initialize();
  }
'''
    
    # Find a good place to insert (after the attemptDatabaseRecovery method)
    recovery_end = content.find('await this.initializationPromise;')
    if recovery_end != -1:
        # Find the end of that method
        next_method = content.find('\n  private', recovery_end)
        if next_method != -1:
            # Insert the reset method
            content = content[:next_method] + '\n' + reset_method + content[next_method:]
    
    # Also improve the error handling in openDatabase
    old_error_handler = '''request.onerror = (event) => {
          console.error('Database error:', request.error);
          reject(request.error || new Error('Unknown database error'));
        };'''
    
    new_error_handler = '''request.onerror = (event) => {
          const error = request.error;
          console.error('Database error:', error);
          
          // Check for common corruption indicators
          if (error && (error.name === 'UnknownError' || 
                       error.message?.includes('Internal error'))) {
            console.warn('⚠️ Database may be corrupted. Consider resetting.');
            // Store error for potential auto-recovery
            localStorage.setItem('tori-db-error', JSON.stringify({
              error: error.name,
              message: error.message,
              timestamp: new Date().toISOString()
            }));
          }
          
          reject(error || new Error('Unknown database error'));
        };'''
    
    content = content.replace(old_error_handler, new_error_handler)
    
    # Save the updated file
    with open(storage_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Fixed database error handling in {storage_path}")
    return True


def fix_hologram_default():
    """Set ENOLA as the default hologram persona"""
    
    # First, check the hologram store
    store_path = Path("tori_ui_svelte/src/lib/stores/hologramStore.ts")
    
    if store_path.exists():
        with open(store_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Look for default persona setting
        if 'currentPersona' in content:
            # Update default persona to ENOLA
            content = re.sub(
                r"currentPersona:\s*'[^']*'",
                "currentPersona: 'Enola'",
                content
            )
            
            with open(store_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✅ Set ENOLA as default persona in {store_path}")
    
    # Also check the Hologram component
    hologram_path = Path("tori_ui_svelte/src/lib/components/Hologram.svelte")
    
    if hologram_path.exists():
        with open(hologram_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find where it says "Initializing video..."
        old_text = '"Initializing video..."'
        new_text = '"ENOLA persona loading..."'
        
        if old_text in content:
            content = content.replace(old_text, new_text)
            
            # Also ensure ENOLA is the default
            # Look for persona initialization
            if 'let currentPersona' in content:
                content = re.sub(
                    r"let currentPersona\s*=\s*'[^']*'",
                    "let currentPersona = 'Enola'",
                    content
                )
            
            with open(hologram_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✅ Updated hologram default text and persona in {hologram_path}")
    
    return True


def create_db_reset_script():
    """Create a simple database reset component"""
    
    reset_component = '''<script lang="ts">
  import { toriStorage } from '$lib/services/toriStorage';
  
  async function resetDatabase() {
    if (confirm('This will clear all local TORI data. Continue?')) {
      try {
        await toriStorage.resetDatabase();
        alert('Database reset successfully!');
        window.location.reload();
      } catch (error) {
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
'''
    
    with open('tori_ui_svelte/src/lib/components/DatabaseReset.svelte', 'w') as f:
        f.write(reset_component)
    
    print("✅ Created DatabaseReset.svelte component")
    return True


def main():
    print("🔧 Fixing TORI frontend issues...")
    print("="*60)
    
    # Fix database error handling
    print("\n1. Fixing IndexedDB error handling...")
    fix_tori_storage()
    
    # Fix hologram default
    print("\n2. Setting ENOLA as default hologram persona...")
    fix_hologram_default()
    
    # Create reset component
    print("\n3. Creating database reset component...")
    create_db_reset_script()
    
    print("\n" + "="*60)
    print("✅ Fixes applied!")
    print("\n💡 Next steps:")
    print("1. Restart the frontend: Ctrl+C and rerun")
    print("2. Clear browser data for localhost:5173")
    print("3. If issues persist, use the Database Reset button")
    print("4. ENOLA should now be the default hologram persona")


if __name__ == "__main__":
    main()
