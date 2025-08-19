from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
#!/usr/bin/env python3
"""
Fix numpy ABI incompatibility error for TORI/KHA
This must be resolved BEFORE implementing BPS solitons
"""

import subprocess
import sys
import os

def fix_numpy_abi_error():
    """
    Fix the numpy.dtype size mismatch error between numpy and thinc/spacy
    """
    print("=" * 60)
    print("🔧 FIXING NUMPY ABI INCOMPATIBILITY ERROR")
    print("=" * 60)
    
    # Step 1: Uninstall problematic packages
    print("\n📦 Step 1: Uninstalling packages with C extensions...")
    packages_to_reinstall = [
        'numpy',
        'scipy', 
        'scikit-learn',
        'spacy',
        'thinc',
        'blis',
        'cymem',
        'preshed',
        'murmurhash',
        'srsly'
    ]
    
    for package in packages_to_reinstall:
        print(f"   Removing {package}...")
        subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', package], 
                      capture_output=True)
    
    # Step 2: Clear pip cache
    print("\n🗑️ Step 2: Clearing pip cache...")
    subprocess.run([sys.executable, '-m', 'pip', 'cache', 'purge'], 
                  capture_output=True)
    print("   Cache cleared!")
    
    # Step 3: Install numpy first with exact version
    print("\n📥 Step 3: Installing numpy 1.26.4 (exact version)...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'numpy==1.26.4'],
                  check=True)
    
    # Step 4: Install other packages in correct order
    print("\n📥 Step 4: Installing scipy and scikit-learn...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', 
                   'scipy>=1.10.0', 'scikit-learn==1.7.0'],
                  check=True)
    
    # Step 5: Install spacy and dependencies
    print("\n📥 Step 5: Installing spacy 3.7.5 and dependencies...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', 
                   'spacy==3.7.5', '--no-cache-dir'],
                  check=True)
    
    # Step 6: Download spacy model if needed
    print("\n📥 Step 6: Ensuring spacy model is available...")
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_lg")
            print("   ✅ Spacy model already installed")
        except:
            print("   📥 Downloading en_core_web_lg model...")
            subprocess.run([sys.executable, '-m', 'spacy', 'download', 'en_core_web_lg'],
                          check=True)
    except ImportError:
        print("   ⚠️ Spacy not properly installed")
    
    # Step 7: Verify the fix
    print("\n✅ Step 7: Verifying the fix...")
    try:
        # This is where the error was occurring
        import spacy
        import thinc
        import numpy as np
        
        print(f"   numpy version: {np.__version__}")
        print(f"   spacy version: {spacy.__version__}")
        print(f"   thinc version: {thinc.__version__}")
        
        # Try to load spacy model
        nlp = spacy.load("en_core_web_lg")
        print("\n🎉 SUCCESS! The numpy ABI error has been fixed!")
        print("   You can now run enhanced_launcher.py without the error.")
        return True
        
    except Exception as e:
        print(f"\n❌ Error still present: {e}")
        print("\nAlternative fix needed - trying Poetry approach...")
        return False

def poetry_fix():
    """
    Alternative fix using Poetry
    """
    print("\n" + "=" * 60)
    print("🔧 ATTEMPTING POETRY-BASED FIX")
    print("=" * 60)
    
    print("\n📦 Reinstalling via Poetry...")
    
    # Update the lock file
    print("   Updating poetry.lock...")
    subprocess.run(['poetry', 'lock', '--no-update'], capture_output=True)
    
    # Clean install
    print("   Clean installing dependencies...")
    subprocess.run(['poetry', 'install', '--no-cache'], check=True)
    
    print("\n✅ Poetry reinstall complete!")
    print("   Try running: poetry run python enhanced_launcher.py")

if __name__ == "__main__":
    print("TORI/KHA Numpy ABI Error Fix Tool")
    print("This will fix the 'numpy.dtype size changed' error\n")
    
    # Change to the kha directory
    os.chdir(r'{PROJECT_ROOT}')
    
    # Try the pip fix first
    success = fix_numpy_abi_error()
    
    if not success:
        # Try poetry as fallback
        try:
            poetry_fix()
        except Exception as e:
            print(f"\n⚠️ Poetry fix failed: {e}")
            print("\nManual intervention may be needed.")
            print("Consider creating a fresh virtual environment.")
    
    print("\n" + "=" * 60)
    print("Fix attempt complete!")
    print("Next step: Test by running enhanced_launcher.py")
    print("=" * 60)
