from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# test_temp.py
"""
Test script to verify temp file operations are working
Run this BEFORE starting your FastAPI server
"""
import os
import sys

# Add project root to path
PROJECT_ROOT = r'{PROJECT_ROOT}'
sys.path.insert(0, PROJECT_ROOT)

print("=" * 60)
print("TORI TEMP FILE TEST")
print("=" * 60)

# Test 1: Check environment
print("\n1. Checking Python environment...")
print(f"Python executable: {sys.executable}")
print(f"Current working directory: {os.getcwd()}")

# Test 2: Import temp manager
print("\n2. Importing temp manager...")
try:
    from temp_manager import temp_manager, save_temp_file, cleanup_temp_file
    print("✓ Temp manager imported successfully")
except Exception as e:
    print(f"✗ Failed to import temp manager: {e}")
    sys.exit(1)

# Test 3: Check temp directory setup
print("\n3. Checking temp directory setup...")
status = temp_manager.get_status()
for key, value in status.items():
    print(f"  {key}: {value}")

# Test 4: Create a test file
print("\n4. Testing file creation...")
try:
    test_content = b"Hello, TORI! This is a test file."
    test_path = save_temp_file(test_content, prefix="test_", suffix=".txt")
    print(f"✓ Created test file: {test_path}")
    
    # Verify file exists
    if os.path.exists(test_path):
        print(f"✓ File exists on disk")
        with open(test_path, 'rb') as f:
            read_content = f.read()
        if read_content == test_content:
            print(f"✓ File content matches")
        else:
            print(f"✗ File content mismatch")
    else:
        print(f"✗ File does not exist on disk")
        
except Exception as e:
    print(f"✗ Failed to create test file: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Test PDF libraries
print("\n5. Testing PDF libraries...")
pdf_libs = []
try:
    import PyPDF2
    pdf_libs.append("PyPDF2")
except:
    pass
try:
    import pdfplumber
    pdf_libs.append("pdfplumber")
except:
    pass
try:
    import pdfminer
    pdf_libs.append("pdfminer")
except:
    pass

if pdf_libs:
    print(f"✓ Available PDF libraries: {', '.join(pdf_libs)}")
else:
    print("✗ No PDF libraries found - install with: pip install PyPDF2 pdfplumber pdfminer.six")

# Test 6: Test cleanup
print("\n6. Testing cleanup (manual mode)...")
cleanup_temp_file(test_path)
print("✓ Cleanup called (file should still exist in manual mode)")
if os.path.exists(test_path):
    print("✓ File still exists (manual mode working)")
else:
    print("✗ File was deleted (manual mode not working)")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("If all tests passed, you can start your FastAPI server!")
print("=" * 60)