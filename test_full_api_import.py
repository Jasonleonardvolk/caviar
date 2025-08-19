#!/usr/bin/env python3
"""
Enhanced test to verify the full API import with CI integration
Features:
- Exit codes for automation
- File logging with timestamps
- Resolved file path echoing
- Optional health check
"""

import sys
import time
import requests
from pathlib import Path
from datetime import datetime

# Add current directory to Python path
script_dir = Path(__file__).parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

# Setup logging
logs_dir = script_dir / "logs"
logs_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = logs_dir / f"test_full_api_import_{timestamp}.log"

def log(msg: str):
    """Log to both console and file"""
    print(msg)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

# Start test
log("🧪 Testing Full API Import...")
log(f"📁 Log file: {log_file}")

try:
    start_time = time.time()
    from prajna.api.prajna_api import app
    import_time = time.time() - start_time
    
    log("✅ SUCCESS! Full API imported successfully")
    log(f"   Import time: {import_time:.3f} seconds")
    log(f"   App type: {type(app)}")
    log(f"   App title: {app.title if hasattr(app, 'title') else 'N/A'}")
    log(f"   App version: {app.version if hasattr(app, 'version') else 'N/A'}")
    
    # Echo resolved file path
    if hasattr(app, '__module__'):
        import importlib
        module = importlib.import_module(app.__module__)
        if hasattr(module, '__file__'):
            log(f"   Imported from: {module.__file__}")
    
    # Optional: Try a lightweight API health check
    log("\n🔍 Attempting API health check...")
    try:
        # Check common API ports
        ports_to_check = [8001, 8002, 7777]
        api_found = False
        
        for port in ports_to_check:
            try:
                response = requests.get(f"http://localhost:{port}/api/health", timeout=2)
                if response.status_code == 200:
                    log(f"✅ API is running on port {port}")
                    log(f"   Health response: {response.json()}")
                    api_found = True
                    break
            except:
                continue
        
        if not api_found:
            log("⚠️  No running API found on common ports (8001, 8002, 7777)")
            log("   This is normal if the API isn't running yet")
    
    except Exception as e:
        log(f"⚠️  Health check failed: {e}")
        log("   This is normal if the API isn't running")
    
    log("\n🎉 The import issue is fixed! You can now run:")
    log("   python enhanced_launcher.py --api full")
    log("\n✅ Test completed successfully")
    
    # Success exit code
    sys.exit(0)
    
except Exception as e:
    log(f"❌ Import failed: {e}")
    log(f"   Error type: {type(e).__name__}")
    
    # Try to provide more detailed error info
    if hasattr(e, '__traceback__'):
        import traceback
        log("\n📋 Traceback:")
        tb_lines = traceback.format_tb(e.__traceback__)
        for line in tb_lines:
            log(f"   {line.strip()}")
    
    log("\n💡 Run these commands to fix:")
    log("   python fix_prajna_init_files.py")
    log("   python diagnose_api_import.py")
    
    # Failure exit code
    sys.exit(1)
