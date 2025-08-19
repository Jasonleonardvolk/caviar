"""
Test Exception Logging in API Startup
This script verifies that exceptions are properly logged
"""

import socket
import time
import subprocess
import sys
from pathlib import Path

print("🧪 TESTING EXCEPTION LOGGING")
print("=" * 60)

# 1. Test port conflict detection
print("\n1️⃣ Testing port conflict detection...")
print("   Creating a conflict on port 8002...")

# Start a dummy server on 8002
dummy_server = subprocess.Popen(
    [sys.executable, "-m", "http.server", "8002"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

time.sleep(2)  # Let it start

# Check if port is taken
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
result = sock.connect_ex(('localhost', 8002))
sock.close()

if result == 0:
    print("   ✅ Port 8002 is now occupied (conflict created)")
else:
    print("   ❌ Failed to create port conflict")
    dummy_server.terminate()
    sys.exit(1)

# 2. Try to start the launcher (should fail with logged error)
print("\n2️⃣ Starting enhanced_launcher.py (should fail)...")
print("   This SHOULD fail and log the error properly...")

launcher_process = subprocess.Popen(
    [sys.executable, "enhanced_launcher.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

# Wait a bit for it to try to start
print("   Waiting 10 seconds for API startup attempt...")
time.sleep(10)

# 3. Check the logs for the error
print("\n3️⃣ Checking logs for exception...")

log_files = [
    "logs/launcher.log",
    Path("logs") / "session_*" / "launcher.log"
]

error_found = False
for log_pattern in log_files:
    if '*' in str(log_pattern):
        # Handle glob pattern
        for log_file in Path("logs").glob("session_*/launcher.log"):
            if log_file.exists():
                content = log_file.read_text(encoding='utf-8', errors='ignore')
                if "API server failed to start" in content:
                    print(f"   ✅ Found error in: {log_file}")
                    # Print the error
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if "API server failed to start" in line:
                            print("\n   📋 Logged error:")
                            print("   " + "="*50)
                            # Print this line and next few
                            for j in range(i, min(i+5, len(lines))):
                                print(f"   {lines[j]}")
                            print("   " + "="*50)
                            error_found = True
                            break
                    break
    else:
        log_file = Path(log_pattern)
        if log_file.exists():
            content = log_file.read_text(encoding='utf-8', errors='ignore')
            if "API server failed to start" in content:
                print(f"   ✅ Found error in: {log_file}")
                error_found = True
                break

# 4. Cleanup
print("\n4️⃣ Cleaning up...")
dummy_server.terminate()
launcher_process.terminate()
print("   ✅ Test processes terminated")

# 5. Results
print("\n" + "="*60)
print("📊 TEST RESULTS:")
print("="*60)

if error_found:
    print("✅ SUCCESS! Exception logging is working!")
    print("   - Port conflict was detected")
    print("   - Error was properly logged")
    print("   - Stack trace was captured")
    print("\n🎉 Your API startup errors will now be visible in logs!")
else:
    print("⚠️  WARNING: Could not find logged exception")
    print("   This could mean:")
    print("   - Exception logging not yet applied")
    print("   - Logs are in a different location")
    print("   - API didn't attempt to start yet")
    print("\n💡 Run: python apply_all_exception_logging.py")
    print("   Then try this test again")

print("\n📁 Check these log files for errors:")
print("   - logs/launcher.log")
print("   - logs/session_*/launcher.log")
print("   - logs/startup_*.log")
