# core/json_encoder_fix.py - Fix JSON serialization for numpy types
import json
import numpy as np
from typing import Any

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types"""
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif hasattr(obj, '__dict__'):
            # Handle dataclasses and objects
            return {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
        return super().default(obj)

def safe_json_dumps(obj: Any, **kwargs) -> str:
    """Safely serialize objects to JSON, handling numpy types"""
    return json.dumps(obj, cls=NumpyEncoder, **kwargs)

def safe_json_dump(obj: Any, fp, **kwargs):
    """Safely serialize objects to JSON file, handling numpy types"""
    return json.dump(obj, fp, cls=NumpyEncoder, **kwargs)
