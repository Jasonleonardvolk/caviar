#!/usr/bin/env python3
"""
Proper fix for soliton field name mismatch using Pydantic aliases
This allows the models to accept both camelCase (from old clients) and snake_case
"""

from pathlib import Path
import time

# Read prajna_api.py
prajna_file = Path("prajna/api/prajna_api.py")
content = prajna_file.read_text(encoding='utf-8')

# Backup
backup_path = Path("backups") / f"prajna_api_{int(time.time())}.py.backup"
backup_path.parent.mkdir(exist_ok=True)
backup_path.write_text(content, encoding='utf-8')
print(f"✅ Created backup: {backup_path}")

print("\n🔍 Applying the proper fix using Pydantic aliases...")

# Fix 1: Update the duplicate SolitonInitRequest model to use aliases
old_init_model = """class SolitonInitRequest(BaseModel):
    user: str = Field(..., description="User identifier")
    initial_concepts: List[str] = Field([], description="Initial concept IDs or names to initialize with")"""

new_init_model = """class SolitonInitRequest(BaseModel):
    user_id: str = Field(..., alias="userId", description="User identifier")
    initial_concepts: List[str] = Field([], description="Initial concept IDs or names to initialize with")
    
    class Config:
        populate_by_name = True  # Accept both 'user_id' and 'userId'"""

# Fix 2: Update the duplicate SolitonStoreRequest model
old_store_model = """class SolitonStoreRequest(BaseModel):
    user: str = Field(..., description="User identifier")
    concept_id: str = Field(..., description="Concept identifier")
    content: str = Field(..., description="Content of the memory")
    activation_strength: float = Field(..., description="Importance level (amplitude base)")"""

new_store_model = """class SolitonStoreRequest(BaseModel):
    user_id: str = Field(..., alias="userId", description="User identifier")
    concept_id: str = Field(..., alias="conceptId", description="Concept identifier")
    content: str = Field(..., description="Content of the memory")
    activation_strength: float = Field(..., alias="importance", description="Importance level (amplitude base)")
    
    class Config:
        populate_by_name = True  # Accept both snake_case and camelCase"""

# Apply fixes
if old_init_model in content:
    content = content.replace(old_init_model, new_init_model)
    print("✅ Fixed SolitonInitRequest model with aliases")
else:
    print("⚠️ SolitonInitRequest pattern not found")

if old_store_model in content:
    content = content.replace(old_store_model, new_store_model)
    print("✅ Fixed SolitonStoreRequest model with aliases")
else:
    print("⚠️ SolitonStoreRequest pattern not found")

# Fix 3: Update handlers to use snake_case field names
# Fix init handler
content = content.replace(
    "user_id = request.user\n    initial_concepts = request.initial_concepts",
    "user_id = request.user_id\n    initial_concepts = request.initial_concepts"
)

# Fix store handler
content = content.replace(
    "user_id = request.user",
    "user_id = request.user_id"
)

# Write the fixed content
prajna_file.write_text(content, encoding='utf-8')

print("\n✅ Applied Pydantic alias fix!")
print("\n🎯 What this does:")
print("   - Models now use snake_case internally (user_id, concept_id)")
print("   - But accept camelCase from frontend via aliases (userId, conceptId)")
print("   - Handlers use consistent snake_case")
print("   - Frontend can send either format!")

print("\n🧪 Test both formats:")
print('   # Snake case (what frontend currently sends):')
print('   curl -X POST http://localhost:8002/api/soliton/init \\')
print('        -H "Content-Type: application/json" \\')
print('        -d \'{"user_id":"test_user"}\'')
print()
print('   # Camel case (backward compatibility):')
print('   curl -X POST http://localhost:8002/api/soliton/init \\')
print('        -H "Content-Type: application/json" \\')
print('        -d \'{"userId":"test_user"}\'')

print("\n📝 Next steps:")
print("   1. Restart the server")
print("   2. Test with .\\test_soliton_powershell.ps1")
print("   3. Later: Remove duplicate endpoints entirely")
