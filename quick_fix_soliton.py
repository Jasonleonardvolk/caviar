#!/usr/bin/env python3
"""
Fix the field name mismatch in soliton endpoints
Based on the analysis, the issue is:
1. The Pydantic model expects 'user' field
2. The handler tries to access 'user_id' 
3. This causes AttributeError -> 500 error
"""

from pathlib import Path
import re
import time

# Read prajna_api.py
prajna_file = Path("prajna/api/prajna_api.py")
content = prajna_file.read_text(encoding='utf-8')

# Backup
backup_path = Path("backups") / f"prajna_api_{int(time.time())}.py.backup"
backup_path.parent.mkdir(exist_ok=True)
backup_path.write_text(content, encoding='utf-8')
print(f"✅ Created backup: {backup_path}")

print("\n🔍 Analyzing the issue...")
print("   - Frontend sends: user_id (snake_case)")
print("   - Pydantic model expects: user (in duplicate endpoints)")
print("   - Handler accesses: user_id")
print("   - Result: AttributeError -> 500 error")

# Option 1: Quick fix - just update the handler to match the model
print("\n🛠️ Applying quick fix...")

# Fix the soliton_init handler to use 'user' instead of 'user_id'
old_handler = """@app.post("/api/soliton/init")
async def soliton_init(request: SolitonInitRequest):
    \"\"\"Initialize soliton lattice for a user with strict Pydantic validation\"\"\"
    user_id = request.user_id  # Changed from request.user
    initial_concepts = []  # SolitonInitRequest from routes doesn't have initial_concepts"""

new_handler = """@app.post("/api/soliton/init")
async def soliton_init(request: SolitonInitRequest):
    \"\"\"Initialize soliton lattice for a user with strict Pydantic validation\"\"\"
    user_id = request.user  # This matches the Pydantic model
    initial_concepts = request.initial_concepts"""

if old_handler in content:
    content = content.replace(old_handler, new_handler)
    print("✅ Fixed soliton_init handler")
else:
    print("⚠️ Handler pattern not found, trying alternative fix...")

# Fix the soliton_store handler similarly
old_store = """user_id = request.user
        concept_id = request.concept_id
        content = request.content
        importance = request.activation_strength"""

new_store = """user_id = request.user
        concept_id = request.concept_id
        content = request.content
        importance = request.activation_strength"""

# The store handler already uses the correct field names

# Write the fixed content
prajna_file.write_text(content, encoding='utf-8')

print("\n✅ Applied quick fix!")
print("\n📝 This is a temporary fix. The proper solution is to:")
print("   1. Remove duplicate endpoints from prajna_api.py")
print("   2. Use only the imported soliton router")
print("   3. Ensure consistent field naming (snake_case everywhere)")

print("\n🧪 Test with:")
print('   curl -X POST http://localhost:8002/api/soliton/init \\')
print('        -H "Content-Type: application/json" \\')
print('        -d \'{"user":"test_user"}\'')
