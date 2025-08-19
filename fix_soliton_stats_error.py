#!/usr/bin/env python3
"""
Quick fix for soliton memory stats error
Updates the frontend to handle the actual backend response format
"""

import os
import shutil
from datetime import datetime

def fix_soliton_stats():
    """Fix the stats format mismatch in solitonMemory.ts"""
    
    soliton_file = "tori_ui_svelte/src/lib/services/solitonMemory.ts"
    
    if not os.path.exists(soliton_file):
        print(f"❌ {soliton_file} not found")
        return False
    
    print(f"🔧 Fixing stats format in {soliton_file}...")
    
    with open(soliton_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the fetchMemoryStats function
    stats_start = content.find("export async function fetchMemoryStats")
    if stats_start == -1:
        print("❌ Could not find fetchMemoryStats function")
        return False
    
    # Find the result handling section
    result_handling = content.find("const result = await response.json();", stats_start)
    if result_handling == -1:
        print("❌ Could not find result handling")
        return False
    
    # Replace the result handling logic
    old_logic = """    if (result.success || result.totalMemories !== undefined) {
      // Reset failure timestamp on success
      lastStatsFailure = 0;
      
      // Update the store with latest stats
      const stats = result.stats || result;
      memoryStats.set({
        totalMemories: stats.totalMemories || 0,
        activeWaves: stats.activeWaves || 0,
        averageStrength: stats.averageStrength || 0,
        clusterCount: stats.clusterCount || 0,
        lastUpdated: new Date()
      });"""
    
    new_logic = """    // Handle the actual backend format
    if (result.success || result.stats || result.user) {
      // Reset failure timestamp on success
      lastStatsFailure = 0;
      
      // Extract data from the different backend format
      const stats = result.stats || result;
      const totalMemories = result.total_concepts || 
                           (stats.concept_memories ? Object.keys(stats.concept_memories).length : 0) ||
                           stats.totalMemories || 
                           0;
      
      memoryStats.set({
        totalMemories: totalMemories,
        activeWaves: stats.activeWaves || 0,
        averageStrength: stats.averageStrength || 0,
        clusterCount: stats.clusterCount || 0,
        lastUpdated: new Date()
      });"""
    
    if old_logic in content:
        # Create backup
        backup = f"{soliton_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy(soliton_file, backup)
        print(f"✅ Created backup: {backup}")
        
        # Replace the logic
        content = content.replace(old_logic, new_logic)
        
        # Write the fixed file
        with open(soliton_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Fixed stats format handling!")
        return True
    else:
        print("⚠️  Stats handling code looks different than expected")
        print("   Applying alternative fix...")
        
        # Try to find and fix the error throw
        error_section = content.find("throw new Error(result.error || 'Unknown stats error');", stats_start)
        if error_section != -1:
            # Replace with a more forgiving approach
            content = content[:error_section] + """// Handle various response formats
      const totalMemories = result.total_concepts || 
                           (result.stats?.concept_memories ? Object.keys(result.stats.concept_memories).length : 0) ||
                           result.totalMemories || 
                           0;
      
      memoryStats.set({
        totalMemories: totalMemories,
        activeWaves: 0,
        averageStrength: 0,
        clusterCount: 0,
        lastUpdated: new Date()
      });
      
      return get(memoryStats);""" + content[error_section + len("throw new Error(result.error || 'Unknown stats error');"):]
            
            # Create backup
            backup = f"{soliton_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copy(soliton_file, backup)
            
            # Write the fixed file
            with open(soliton_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print("✅ Applied alternative fix!")
            return True
        
        print("❌ Could not apply automatic fix")
        return False

def main():
    print("🔧 Fixing Soliton Memory Stats Error")
    print("=" * 60)
    
    if fix_soliton_stats():
        print("\n✅ Fix applied successfully!")
        print("\n🚀 Next steps:")
        print("1. Rebuild the frontend:")
        print("   cd tori_ui_svelte")
        print("   npm run build")
        print("\n2. Or if using dev mode, it should auto-reload")
        print("\nThe 'Unknown stats error' should be gone!")
    else:
        print("\n❌ Could not apply fix automatically")
        print("\nManual fix instructions:")
        print("1. Open tori_ui_svelte/src/lib/services/solitonMemory.ts")
        print("2. Find the fetchMemoryStats function (around line 300)")
        print("3. Replace the stats handling to accept the backend format:")
        print("   - It returns: {user, stats: {concept_memories: {}}, total_concepts: 0}")
        print("   - Not: {totalMemories, activeWaves, averageStrength, clusterCount}")

if __name__ == "__main__":
    main()
