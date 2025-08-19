# Run with: poetry run pytest -k test_event_chain_backend
import json
import time
import glob
import os

LOGS = "logs/inference"

def tail_jsonl(pattern, timeout=10, needle=None):
    start = time.time()
    while time.time() - start < timeout:
        paths = sorted(glob.glob(f"{LOGS}/{pattern}"), reverse=True)
        for p in paths:
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    for line in reversed(f.read().splitlines()):
                        try:
                            evt = json.loads(line)
                            if not needle or needle in json.dumps(evt):
                                return evt
                        except:  # noqa
                            pass
        time.sleep(0.25)
    raise AssertionError(f"needle not found: {needle}")

def test_event_chain_backend():
    """Test that event chain is properly logged in backend"""
    # Expect mesh_updated → summary_exported → inference_started → inference_complete to be logged
    assert tail_jsonl("*.jsonl", 10, "mesh_updated")
    assert tail_jsonl("*.jsonl", 10, "summary_exported")
    assert tail_jsonl("*.jsonl", 10, "inference_started")
    assert tail_jsonl("*.jsonl", 15, "inference_complete")
    print("✅ Event chain backend test passed")
