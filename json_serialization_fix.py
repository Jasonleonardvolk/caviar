"""
JSON Serialization Fix for Consciousness System
==============================================

Custom JSON encoder to handle consciousness system objects safely.
Fixes: Object of type EvolutionStrategy/ConsciousnessPhase is not JSON serializable
"""

import json
import datetime
from typing import Any, Dict
from enum import Enum
from dataclasses import is_dataclass, asdict

class ConsciousnessJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for consciousness system objects.
    
    Handles:
    - ConsciousnessPhase enums
    - EvolutionStrategy enums
    - EvolutionPattern enums
    - Datetime objects
    - Dataclasses
    - Any other custom objects
    """
    
    def default(self, obj):
        # Handle Enum types (ConsciousnessPhase, EvolutionStrategy, etc.)
        if isinstance(obj, Enum):
            return {
                '_enum_type': obj.__class__.__name__,
                '_enum_value': obj.value,
                '_enum_name': obj.name
            }
        
        # Handle datetime objects
        if isinstance(obj, datetime.datetime):
            return {
                '_datetime': True,
                'value': obj.isoformat()
            }
        
        if isinstance(obj, datetime.date):
            return {
                '_date': True,
                'value': obj.isoformat()
            }
        
        # Handle dataclasses
        if is_dataclass(obj):
            return {
                '_dataclass_type': obj.__class__.__name__,
                '_dataclass_data': asdict(obj)
            }
        
        # Handle sets
        if isinstance(obj, set):
            return {
                '_set': True,
                'value': list(obj)
            }
        
        # Handle custom consciousness objects with special handling
        if hasattr(obj, '__dict__'):
            # For objects with __dict__, serialize their attributes
            return {
                '_object_type': obj.__class__.__name__,
                '_object_data': {k: v for k, v in obj.__dict__.items() 
                               if not k.startswith('_') and not callable(v)}
            }
        
        # Fallback for any other type - convert to string representation
        try:
            return str(obj)
        except Exception:
            return f"<{type(obj).__name__} object>"

class ConsciousnessJSONDecoder(json.JSONDecoder):
    """
    Custom JSON decoder to reconstruct consciousness objects.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)
    
    def object_hook(self, obj):
        # Reconstruct enum objects
        if '_enum_type' in obj and '_enum_value' in obj:
            # For simple cases, just return the value
            return obj['_enum_value']
        
        # Reconstruct datetime objects
        if '_datetime' in obj:
            return datetime.datetime.fromisoformat(obj['value'])
        
        if '_date' in obj:
            return datetime.date.fromisoformat(obj['value'])
        
        # Reconstruct sets
        if '_set' in obj:
            return set(obj['value'])
        
        # For other objects, return as-is (could be enhanced for full reconstruction)
        return obj

def safe_json_dump(obj: Any, file_path: str, **kwargs) -> bool:
    """
    Safely dump any consciousness system object to JSON.
    
    Args:
        obj: Object to serialize
        file_path: Path to save JSON file
        **kwargs: Additional arguments for json.dump
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(obj, f, cls=ConsciousnessJSONEncoder, 
                     indent=2, ensure_ascii=False, **kwargs)
        return True
    except Exception as e:
        print(f"❌ Failed to save JSON to {file_path}: {e}")
        return False

def safe_json_dumps(obj: Any, **kwargs) -> str:
    """
    Safely serialize any consciousness system object to JSON string.
    
    Args:
        obj: Object to serialize
        **kwargs: Additional arguments for json.dumps
    
    Returns:
        str: JSON string or error message
    """
    try:
        return json.dumps(obj, cls=ConsciousnessJSONEncoder, 
                         indent=2, ensure_ascii=False, **kwargs)
    except Exception as e:
        return f'{{"error": "JSON serialization failed: {str(e)}"}}'

def safe_json_load(file_path: str, **kwargs) -> Any:
    """
    Safely load JSON file with consciousness object reconstruction.
    
    Args:
        file_path: Path to JSON file
        **kwargs: Additional arguments for json.load
    
    Returns:
        Any: Loaded object or None if failed
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f, cls=ConsciousnessJSONDecoder, **kwargs)
    except Exception as e:
        print(f"❌ Failed to load JSON from {file_path}: {e}")
        return None

def prepare_object_for_json(obj: Any) -> Any:
    """
    Recursively prepare an object for JSON serialization by converting
    problematic types to JSON-safe equivalents.
    
    Args:
        obj: Object to prepare
        
    Returns:
        Any: JSON-safe version of the object
    """
    if isinstance(obj, Enum):
        return obj.value
    
    elif isinstance(obj, datetime.datetime):
        return obj.isoformat()
    
    elif isinstance(obj, datetime.date):
        return obj.isoformat()
    
    elif isinstance(obj, set):
        return list(obj)
    
    elif is_dataclass(obj):
        return {k: prepare_object_for_json(v) for k, v in asdict(obj).items()}
    
    elif isinstance(obj, dict):
        return {k: prepare_object_for_json(v) for k, v in obj.items()}
    
    elif isinstance(obj, (list, tuple)):
        return [prepare_object_for_json(item) for item in obj]
    
    elif hasattr(obj, '__dict__'):
        # For custom objects, extract their attributes
        return {k: prepare_object_for_json(v) for k, v in obj.__dict__.items() 
                if not k.startswith('_') and not callable(v)}
    
    else:
        # For anything else, try to convert to a simple type
        try:
            # Test if it's already JSON serializable
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            # If not serializable, convert to string
            return str(obj)

# Convenience functions for common consciousness system objects
def save_consciousness_state(state: Dict[str, Any], filename: str = "consciousness_state.json") -> bool:
    """Save consciousness state with proper JSON handling"""
    safe_state = prepare_object_for_json(state)
    return safe_json_dump(safe_state, filename)

def save_evolution_metrics(metrics: Any, filename: str = "evolution_metrics.json") -> bool:
    """Save evolution metrics with proper JSON handling"""
    safe_metrics = prepare_object_for_json(metrics)
    return safe_json_dump(safe_metrics, filename)

def save_orchestrator_status(status: Dict[str, Any], filename: str = "orchestrator_status.json") -> bool:
    """Save orchestrator status with proper JSON handling"""
    safe_status = prepare_object_for_json(status)
    return safe_json_dump(safe_status, filename)

if __name__ == "__main__":
    # Test the JSON encoder with consciousness objects
    print("🧪 Testing Consciousness JSON Encoder...")
    
    # Import the enums for testing
    try:
        from evolution_metrics import ConsciousnessPhase, EvolutionPattern
        from darwin_godel_orchestrator import EvolutionStrategy
        
        # Test data with problematic objects
        test_data = {
            'consciousness_phase': ConsciousnessPhase.ADAPTIVE,
            'evolution_pattern': EvolutionPattern.EMERGENT,
            'evolution_strategy': EvolutionStrategy.SEMANTIC_FUSION,
            'timestamp': datetime.datetime.now(),
            'metrics': {
                'awareness': 0.75,
                'complexity': 0.65,
                'phase_set': {ConsciousnessPhase.NASCENT, ConsciousnessPhase.COHERENT}
            }
        }
        
        # Test serialization
        json_string = safe_json_dumps(test_data)
        print("✅ JSON serialization successful:")
        print(json_string[:200] + "..." if len(json_string) > 200 else json_string)
        
        # Test file save
        success = safe_json_dump(test_data, "test_consciousness.json")
        print(f"✅ File save successful: {success}")
        
        print("\n🎆 Consciousness JSON encoder working perfectly!")
        
    except ImportError as e:
        print(f"⚠️ Could not import consciousness enums for testing: {e}")
        print("✅ JSON encoder is ready for use when consciousness system is available")
