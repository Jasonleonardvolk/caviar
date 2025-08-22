// Client-side TORI initialization
// This runs in the browser after page load to ensure all systems are active

import { browser } from '$app/environment';
import { initializeTORI, forceELFINAssignment } from '$lib/toriInit';

console.log('📡 client-init.ts loading...', { browser });

// Only run in browser context
if (browser) {
  console.log('🌐 Client-side TORI initialization starting...');
  
  // Wait for DOM to be ready
  if (document.readyState === 'loading') {
    console.log('⏳ Waiting for DOMContentLoaded...');
    document.addEventListener('DOMContentLoaded', initializeClientTORI);
  } else {
    console.log('✅ DOM already ready, initializing immediately');
    initializeClientTORI();
  }
} else {
  console.log('⚠️ Not in browser context, skipping client init');
}

function initializeClientTORI() {
  console.log('🌐 Client TORI initialization function called');
  
  // Wait a bit for all modules to load
  setTimeout(() => {
    console.log('🔧 Starting TORI system initialization...');
    
    const success = initializeTORI();
    console.log('🔧 initializeTORI result:', success);
    
    if (!success) {
      console.warn('⚠️ Initial TORI initialization failed, trying force assignment...');
      const forced = forceELFINAssignment();
      console.log('🔧 Force assignment result:', forced);
    }
    
    // Final verification with detailed debugging
    setTimeout(() => {
      if (typeof window !== 'undefined') {
        const elfinExists = !!(window as any).ELFIN;
        const toriExists = !!(window as any).TORI;
        
        console.log('🔍 Final verification:');
        console.log('  - window exists:', typeof window !== 'undefined');
        console.log('  - window.ELFIN exists:', elfinExists);
        console.log('  - window.TORI exists:', toriExists);
        
        if (elfinExists) {
          console.log('✅ ELFIN++ client initialization successful');
          console.log('🧪 Available commands:');
          console.log('  - window.ELFIN.getExecutionStats()');
          console.log('  - window.TORI.testUpload()');
          console.log('  - window.TORI.checkStats()');
          
          // Test the interface
          try {
            const stats = (window as any).ELFIN.getExecutionStats();
            console.log('🧪 ELFIN++ stats test successful:', stats);
          } catch (error) {
  const msg = error instanceof Error ? error.message : String(error);
            console.error('❌ ELFIN++ stats test failed:', error);
          
}
        } else {
          console.error('❌ ELFIN++ client initialization failed');
          console.log('🔧 Available on window:', Object.keys(window).filter(k => 
            k.toLowerCase().includes('elfin') || 
            k.toLowerCase().includes('tori')
          ));
        }
      } else {
        console.error('❌ Window object not available in verification');
      }
    }, 500);
    
  }, 100);
}

console.log('📡 client-init.ts loaded successfully');
