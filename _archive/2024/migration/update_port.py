from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
#!/usr/bin/env python3
"""
🔧 PORT UPDATE - Change SvelteKit to use port 8003 instead of 8002
"""

import os
import re

def update_port():
    """Update SvelteKit server to use port 8003"""
    
    server_file = r"{PROJECT_ROOT}\tori_ui_svelte\src\routes\upload\+server.ts"
    
    if not os.path.exists(server_file):
        print(f"❌ Server file not found: {server_file}")
        return False
    
    print("🔧 Updating SvelteKit server to use port 8003...")
    
    try:
        # Read file
        with open(server_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace all localhost:8002 with localhost:8003
        updated_content = content.replace('localhost:8002', 'localhost:8003')
        
        # Write back
        with open(server_file, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        print("✅ Updated SvelteKit server to use port 8003")
        return True
        
    except Exception as e:
        print(f"❌ Failed to update port: {e}")
        return False

if __name__ == "__main__":
    success = update_port()
    
    if success:
        print("\n🚀 NOW START THE PYTHON API ON PORT 8003:")
        print("   uvicorn ingest_pdf.main:app --port 8003 --host 0.0.0.0")
        print("\n🌐 THEN RESTART SVELTEKIT:")
        print("   cd tori_ui_svelte")
        print("   npm run dev")
        print("\n✅ Upload should work on port 8003!")
    else:
        print("\n❌ Manual update needed")
