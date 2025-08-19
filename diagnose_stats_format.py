#!/usr/bin/env python3
"""
Fix the stats response format mismatch
"""

print("🔍 ANALYZING STATS FORMAT MISMATCH")
print("=" * 60)

print("\nFRONTEND EXPECTS:")
print("""
{
  "totalMemories": number,
  "activeWaves": number,
  "averageStrength": number,
  "clusterCount": number
}
""")

print("\nBACKEND RETURNS:")
print("""
{
  "user": "adminuser",
  "stats": {
    "concept_memories": {},
    "phase_registry": {},
    "evolution_lineage": {},
    "psi_anchors": {},
    "last_updated": "2025-06-06T09:15:39.972638"
  },
  "timestamp": 1753008267.7348557,
  "total_concepts": 0
}
""")

print("\n❌ MISMATCH FOUND!")
print("The backend is returning a different format than what the frontend expects!")

print("\n🔧 SOLUTION:")
print("We need to check which soliton router is actually being used.")
print("It seems like it's using a different implementation than soliton_production.py")

print("\nLet's check what's in the main.py soliton router import...")

# Check the actual router being used
import os
import ast

# Read main.py to see which router it imports
with open("main.py", "r") as f:
    content = f.read()
    
# Find the import
if "soliton_router" in content:
    print("\n📄 main.py imports:")
    # Extract the import line
    for line in content.split('\n'):
        if 'soliton_router' in line and 'import' in line:
            print(f"   {line.strip()}")
            
print("\nNow let's check what soliton_router.py actually loads...")

# Check soliton_router.py
router_file = "api/routes/soliton_router.py"
if os.path.exists(router_file):
    with open(router_file, "r") as f:
        router_content = f.read()
    print(f"\n📄 {router_file} content:")
    print(router_content)
    
print("\n🎯 THE ISSUE:")
print("The stats endpoint is returning a different format than expected!")
print("We need to either:")
print("1. Fix the backend to return the expected format")
print("2. Fix the frontend to handle the actual format")
print("3. Find which router is actually being used")

# Create a fix
fix_code = '''
# In the actual soliton route file, the stats endpoint should return:
return {
    "totalMemories": len(stats.get("concept_memories", {})),
    "activeWaves": 0,  # Calculate from actual data
    "averageStrength": 0.0,
    "clusterCount": 0,
    "success": True
}

# Instead of the current format
'''

print("\n📝 FIX NEEDED:")
print(fix_code)
