"""
This script lists all the duplicate soliton endpoints that need to be removed
"""

endpoints_to_remove = [
    "@app.post(\"/api/soliton/init\")",
    "@app.post(\"/api/soliton/store\")",
    "@app.get(\"/api/soliton/recall/{user_id}/{concept_id}\")",
    "@app.post(\"/api/soliton/phase/{user_id}\")",
    "@app.get(\"/api/soliton/related/{user_id}/{concept_id}\")",
    "@app.post(\"/api/soliton/vault/{user_id}\")",
    "@app.get(\"/api/soliton/health\")",
    "@app.get(\"/api/soliton/stats/{user}\")"
]

print("The following duplicate endpoints need to be removed from prajna_api.py:")
print("(They are already provided by the imported soliton router)\n")

for endpoint in endpoints_to_remove:
    print(f"  - {endpoint}")

print("\nThese duplicate endpoints are causing the field name mismatch.")
print("The router expects 'user_id' but the duplicates expect 'user'.")
