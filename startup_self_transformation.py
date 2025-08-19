#!/usr/bin/env python3
"""
TORI Self-Transformation System Startup
Initializes and verifies all self-transformation components
"""

import sys
import os
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_component(name, check_func):
    """Check a component and report status"""
    try:
        check_func()
        print(f"✓ {name}")
        return True
    except Exception as e:
        print(f"✗ {name}: {str(e)[:50]}...")
        return False

def startup_self_transformation():
    """Initialize and verify self-transformation system"""
    print("=" * 60)
    print("TORI Self-Transformation System Startup")
    print(f"Time: {datetime.now().isoformat()}")
    print("=" * 60)
    print()
    
    components_ok = []
    
    # Check constitution
    def check_constitution():
        from safety.constitution import Constitution
        const = Constitution(path="safety/constitution.json")
        assert const.doc["identity"]["uuid"]
    components_ok.append(check_component("Constitutional Safety", check_constitution))
    
    # Check critics
    def check_critics():
        from meta_genome.critics.aggregation import aggregate
        scores = {"test": 0.8}
        reliabilities = {"test": 0.9}
        accepted, score = aggregate(scores, reliabilities)
        assert isinstance(accepted, bool)
    components_ok.append(check_component("Critic Consensus", check_critics))
    
    # Check energy budget
    def check_energy():
        from meta.energy_budget import EnergyBudget
        energy = EnergyBudget()
        assert energy.current_energy > 0
    components_ok.append(check_component("Energy Budget", check_energy))
    
    # Check analogical transfer
    def check_transfer():
        from goals.analogical_transfer import AnalogicalTransfer
        import numpy as np
        transfer = AnalogicalTransfer()
        transfer.add_knowledge_cluster("test", ["c1"], np.array([1.0]))
    components_ok.append(check_component("Analogical Transfer", check_transfer))
    
    # Check audit
    def check_audit():
        from audit.logger import log_event
        log_event("startup", {"component": "self_transformation"})
    components_ok.append(check_component("Audit Logger", check_audit))
    
    # Summary
    print()
    print("=" * 60)
    total = len(components_ok)
    working = sum(components_ok)
    print(f"Components: {working}/{total} operational")
    
    if working == total:
        print("Status: READY FOR SELF-TRANSFORMATION")
        
        # Save startup state
        state = {
            "timestamp": datetime.now().isoformat(),
            "components_checked": total,
            "components_operational": working,
            "status": "ready"
        }
        
        with open("self_transformation_state.json", "w") as f:
            json.dump(state, f, indent=2)
            
        print(f"State saved to self_transformation_state.json")
    else:
        print("Status: DEGRADED - Some components need attention")
        return False
    
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = startup_self_transformation()
    sys.exit(0 if success else 1)
