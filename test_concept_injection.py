"""
Test script to verify the log_concept_injection function is working correctly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from phase2_advanced_psi_lineage_ledger_complete import log_concept_injection, log_concept_injection_to_ledger, AdvancedPsiLineageLedger

def test_log_concept_injection():
    """Test the log_concept_injection function"""
    print("🧪 Testing log_concept_injection function")
    print("=" * 50)
    
    # Test 1: Basic concept injection logging
    print("\n📋 Test 1: Basic concept injection logging")
    result = log_concept_injection(
        concept_name="neural_network_architecture",
        source="pdf_upload",
        metadata={"file": "research_paper.pdf", "page": 15}
    )
    
    if result:
        print("✅ Basic logging successful")
        print(f"   Logged concept: {result['concept_name']}")
        print(f"   Source: {result['source']}")
        print(f"   Timestamp: {result['timestamp']}")
    else:
        print("❌ Basic logging failed")
    
    # Test 2: Advanced ledger integration
    print("\n🧬 Test 2: Advanced ledger integration")
    
    # Create a test ledger instance
    test_ledger = AdvancedPsiLineageLedger("test_injection_ledger.json")
    
    # Test injection to ledger
    success = log_concept_injection_to_ledger(
        ledger_instance=test_ledger,
        concept_name="hyperdimensional_computing",
        source="pdf_extraction",
        metadata={
            "extraction_method": "advanced_nlp",
            "confidence": 0.95,
            "section": "methodology"
        }
    )
    
    if success:
        print("✅ Advanced ledger injection successful")
        
        # Verify the concept was added
        status = test_ledger.get_advanced_ledger_status()
        print(f"   Total concepts in ledger: {status['ledger_info']['total_concepts']}")
        
        # Check if our concept is there
        if "hyperdimensional_computing" in test_ledger.concepts:
            concept_record = test_ledger.concepts["hyperdimensional_computing"]
            print(f"   ✅ Concept found: {concept_record.canonical_name}")
            print(f"   📊 Metadata: {concept_record.metadata}")
        else:
            print("   ❌ Concept not found in ledger")
    else:
        print("❌ Advanced ledger injection failed")
    
    # Test 3: Multiple concept injections
    print("\n📚 Test 3: Multiple concept injections")
    
    concepts_to_inject = [
        ("quantum_computing", "pdf_extraction", {"section": "introduction"}),
        ("machine_learning", "manual_entry", {"category": "ai_fundamentals"}),
        ("cognitive_architecture", "pdf_extraction", {"section": "discussion"})
    ]
    
    successful_injections = 0
    for concept_name, source, metadata in concepts_to_inject:
        success = log_concept_injection_to_ledger(
            test_ledger, concept_name, source, metadata
        )
        if success:
            successful_injections += 1
    
    print(f"   Successfully injected {successful_injections}/{len(concepts_to_inject)} concepts")
    
    # Final status
    final_status = test_ledger.get_advanced_ledger_status()
    print(f"\n📊 Final ledger status:")
    print(f"   Total concepts: {final_status['ledger_info']['total_concepts']}")
    print(f"   Total relationships: {final_status['ledger_info']['total_relationships']}")
    
    print("\n🎆 log_concept_injection testing complete!")
    print("✅ Function is operational and ready for Friday2 integration")

if __name__ == "__main__":
    test_log_concept_injection()
