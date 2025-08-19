"""
Seed fake adapters for testing.
Creates test adapter files with correct SHA256 hashes.
"""
import os, hashlib, json
from pathlib import Path
from secrets import token_bytes

def seed_adapters(adapter_dir=None):
    """Create test adapter files with manifest."""
    root = Path(adapter_dir or os.environ.get("TORI_ADAPTER_DIR", "tests/adapters"))
    root.mkdir(parents=True, exist_ok=True)
    
    adapters = {}
    
    # Create various test adapters
    for i in range(1, 6):
        name = f"adapter_{i}.bin"
        path = root / name
        
        # Generate unique content
        content = token_bytes(256 + i * 100)  # Different sizes
        path.write_bytes(content)
        
        # Calculate hash
        sha256 = hashlib.sha256(content).hexdigest()
        
        adapters[name] = {
            "path": name,
            "sha256": sha256,
            "size": len(content),
            "version": f"1.0.{i}"
        }
        
        print(f"Created {name}: {len(content)} bytes, SHA256: {sha256[:16]}...")
    
    # Create manifest
    manifest_path = root / "metadata.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(adapters, f, indent=2)
        f.flush()
    
    print(f"\nSeeded {len(adapters)} adapters at {root}")
    print(f"Manifest: {manifest_path}")
    
    return root, adapters

if __name__ == "__main__":
    import sys
    
    adapter_dir = sys.argv[1] if len(sys.argv) > 1 else None
    root, adapters = seed_adapters(adapter_dir)
    
    # Verify files
    for name in adapters:
        path = root / name
        if not path.exists():
            print(f"ERROR: {path} not created!")
            sys.exit(1)
    
    print("\nVerification: All adapters created successfully")
