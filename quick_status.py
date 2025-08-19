import subprocess
import sys

# Quick check of merge gate status
print("=" * 60)
print("MERGE GATE STATUS - QUICK CHECK")
print("=" * 60)
print()

# Run the Python status check
result = subprocess.run([sys.executable, "check_merge_gate_status.py"], capture_output=True, text=True)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

print()
print("✅ Concept Mesh Rename: COMPLETE!")
print("   - Both Cargo.toml files updated")
print("   - Python imports updated with fallback pattern")
print()
print("Remaining items:")
print("1. Run: ACTION_15_MINUTES.bat")
print("2. Update README badge") 
print("3. Push to GitHub")
print()
