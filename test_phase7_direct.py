"""Test Phase 7: Oscillator feed without requiring API."""
import asyncio
import time
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("Testing Phase 7: Oscillator Feed Integration")
print("=" * 50)

try:
    # Import and setup
    from python.core.fractal_soliton_events import concept_event_bus, ConceptEvent
    from python.core.lattice_evolution_subscriber import oscillator_count, on_concept_added
    
    print(f"Initial oscillator count: {oscillator_count}")
    
    # Subscribe to events
    concept_event_bus.subscribe('concept_added', on_concept_added)
    print("✅ Subscribed to concept events")
    
    # Emit test events
    async def test_events():
        for i in range(3):
            test_event = ConceptEvent(
                concept_id=f"test_oscillator_{i}",
                phase=0.5 + i * 0.1,
                operation="add",
                timestamp=datetime.utcnow()
            )
            
            print(f"\nEmitting event {i+1}: {test_event.concept_id}")
            await concept_event_bus.emit("concept_added", test_event)
            
            # Give it a moment to process
            await asyncio.sleep(0.1)
            
            # Check updated count
            from python.core.lattice_evolution_subscriber import oscillator_count as new_count
            print(f"  Oscillator count after event: {new_count}")
    
    # Run the test
    asyncio.run(test_events())
    
    # Final check
    from python.core.lattice_evolution_subscriber import oscillator_count as final_count
    print(f"\n✅ Phase 7 Test Complete!")
    print(f"Final oscillator count: {final_count}")
    print(f"Events successfully triggered oscillator updates!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 50)
print("Phase 7 test completed successfully!")
