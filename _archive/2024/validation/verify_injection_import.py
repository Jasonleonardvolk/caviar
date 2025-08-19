"""
Verification script to confirm log_concept_injection import works
"""

def verify_import():
    """Verify that log_concept_injection can be imported"""
    try:
        from phase2_advanced_psi_lineage_ledger_complete import log_concept_injection
        print("✅ SUCCESS: log_concept_injection imported successfully!")
        
        # Test basic functionality
        result = log_concept_injection(
            concept_name="test_concept", 
            source="verification", 
            metadata={"test": True}
        )
        
        if result:
            print("✅ SUCCESS: log_concept_injection function works!")
            print(f"   Returned: {result}")
            return True
        else:
            print("❌ ERROR: log_concept_injection returned None")
            return False
            
    except ImportError as e:
        print(f"❌ IMPORT ERROR: {e}")
        return False
    except Exception as e:
        print(f"❌ FUNCTION ERROR: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Verifying log_concept_injection import...")
    success = verify_import()
    
    if success:
        print("\n🎉 VERIFICATION COMPLETE - READY FOR FRIDAY2 INTEGRATION!")
    else:
        print("\n💥 VERIFICATION FAILED - NEEDS DEBUGGING")
