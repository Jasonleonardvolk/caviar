// Quick fix for solitonMemory.ts to handle the actual backend format
// Add this to your fetchMemoryStats function

// Around line 310 in fetchMemoryStats, replace the success handling with:

if (result.success || result.stats) {
    // Handle the actual backend format
    const stats = result.stats || result;
    
    // Extract data from the different format
    const totalMemories = result.total_concepts || 
                         Object.keys(stats.concept_memories || {}).length || 
                         0;
    
    memoryStats.set({
        totalMemories: totalMemories,
        activeWaves: 0,  // Backend doesn't provide this
        averageStrength: 0,  // Backend doesn't provide this
        clusterCount: 0,  // Backend doesn't provide this
        lastUpdated: new Date()
    });
    
    return get(memoryStats);
}
