"""Script to integrate the diff endpoint into the main API."""
import sys
from pathlib import Path

# Add this to your main API file (prajna_api.py or similar)
integration_code = '''
# Phase 6: Concept Mesh Diff Integration
from api.concept_mesh_diff import router as concept_mesh_router
app.include_router(concept_mesh_router)
'''

print("Add this to your main API file:")
print("=" * 50)
print(integration_code)
print("=" * 50)
print("\nThen in your ingest flow, after successful ingestion:")
print('''
import httpx

async def notify_concept_diff(record_id: str, concept_id: str):
    """Notify the diff endpoint after concept mesh update."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8002/api/concept-mesh/record_diff",
            json={
                "record_id": record_id,
                "concept_id": concept_id,
                "operation": "update"
            }
        )
        if response.status_code == 200:
            logger.info(f"✅ Diff queued: {response.json()}")
        else:
            logger.error(f"❌ Failed to queue diff: {response.text}")
''')
