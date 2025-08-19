// Quick fix for Soliton Memory initialization
// Add this to your browser console to manually initialize

(async () => {
  console.log("🔧 Manual Soliton Memory Initialization");
  
  // Import the soliton memory module
  const solitonModule = await import('/src/lib/services/solitonMemory.ts');
  
  // Try to initialize with a default user
  try {
    await solitonModule.initializeUserMemory('Admin User');
    console.log("✅ Memory initialized successfully!");
    
    // Check stats
    const stats = await solitonModule.fetchMemoryStats('Admin User');
    console.log("📊 Memory stats:", stats);
    
    // Update the UI by dispatching an event
    window.dispatchEvent(new CustomEvent('soliton-initialized', {
      detail: { user: 'Admin User', stats }
    }));
    
  } catch (error) {
    console.error("❌ Initialization failed:", error);
    
    // Check if it's a backend issue
    const healthCheck = await fetch('/api/health');
    console.log("Backend health status:", healthCheck.status);
  }
})();
