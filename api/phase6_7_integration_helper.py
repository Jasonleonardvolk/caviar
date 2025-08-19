"""
Integration helper for Phase 6 - Concept Mesh Diff

Add this to your ingest flow after successful concept extraction:
"""

integration_code = '''
# Phase 6: Notify concept mesh diff endpoint
async def notify_concept_diff(record_id: str, concept_id: str, operation: str = "update"):
    """Notify the diff endpoint after concept mesh update."""
    import httpx
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:8002/api/concept-mesh/record_diff",
                json={
                    "record_id": record_id,
                    "concept_id": concept_id,
                    "operation": operation,
                    "metadata": {
                        "timestamp": time.time(),
                        "source": "ingest_pipeline"
                    }
                }
            )
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ Diff queued: {data['diff_id']}")
            else:
                logger.error(f"❌ Failed to queue diff: {response.text}")
    except Exception as e:
        logger.error(f"❌ Error notifying diff endpoint: {e}")

# In your PDF processing function, after concept extraction:
# await notify_concept_diff(document_id, concept_id, "create")
'''

print("Phase 6 Integration Code:")
print("=" * 60)
print(integration_code)
print("=" * 60)

# Also emit concept events for Phase 7
phase7_code = '''
# Phase 7: Emit concept events for oscillator feed
def emit_concept_event(concept_id: str, phase: float = None):
    """Emit concept event for oscillator lattice."""
    try:
        from python.core.fractal_soliton_events import concept_event_bus, ConceptEvent
        from datetime import datetime
        
        event = ConceptEvent(
            concept_id=concept_id,
            phase=phase or (hash(concept_id) % 1000) / 1000.0,
            operation="add",
            timestamp=datetime.utcnow()
        )
        
        # This will trigger the lattice subscriber
        import asyncio
        asyncio.create_task(concept_event_bus.emit("concept_added", event))
        
        logger.info(f"🌊 Emitted concept event: {concept_id}")
    except Exception as e:
        logger.warning(f"Could not emit concept event: {e}")

# In your concept processing:
# emit_concept_event(concept_id, phase=0.5)
'''

print("\nPhase 7 Integration Code:")
print("=" * 60)
print(phase7_code)
print("=" * 60)
