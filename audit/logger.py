# /audit/logger.py
import json
import datetime
import os
from pathlib import Path

LOG_FILE = os.path.join(os.path.dirname(__file__), "events.log")

def log_event(event_type: str, data: dict):
    """Log an event with timestamp and type."""
    event = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "type": event_type,
        "data": data
    }
    
    # Ensure directory exists
    Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
    
    # Append to log file
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(event) + "\n")
