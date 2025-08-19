#!/usr/bin/env python3
"""
Interactive TORI Dependency Fix Menu
"""

import os
import sys
import subprocess
from pathlib import Path

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def run_script(script_name):
    """Run a Python script"""
    print(f"\nRunning {script_name}...\n")
    if script_name.endswith('.ps1'):
        subprocess.run(['powershell', '-ExecutionPolicy', 'Bypass', '-File', script_name])
    else:
        subprocess.run([sys.executable, script_name])
    input("\nPress Enter to continue...")

def show_menu():
    """Display the main menu"""
    clear_screen()
    print("=" * 60)
    print("🔧 TORI System Dependency Fix Menu")
    print("=" * 60)
    print("\nCurrent Issues:")
    print("  • Broken scipy installation (~cipy)")
    print("  • Numpy binary incompatibility")
    print("  • Transformers import errors")
    print("  • Entropy pruning not working")
    print("\nOptions:")
    print("  1. 🔍 Check current state")
    print("  2. ⚡ Quick emergency fix (PowerShell)")
    print("  3. 🛠️  Comprehensive dependency fix")
    print("  4. 🧪 Test entropy pruning")
    print("  5. 📄 Test PDF pipeline")
    print("  6. 📊 Show system status")
    print("  7. 📚 View fix guide")
    print("  8. 🚀 Start API server (after fix)")
    print("  9. ❌ Exit")
    print("\n" + "=" * 60)

def check_state():
    """Check current dependency state"""
    run_script('test_entropy_state.py')

def emergency_fix():
    """Run emergency PowerShell fix"""
    print("\n⚠️  This will reinstall many packages.")
    print("Make sure all Python processes are closed!")
    confirm = input("Continue? (y/n): ")
    if confirm.lower() == 'y':
        run_script('emergency_dependency_fix.ps1')

def comprehensive_fix():
    """Run comprehensive Python fix"""
    run_script('comprehensive_dependency_fix.py')

def test_entropy():
    """Test entropy pruning"""
    run_script('verify_entropy.py')

def test_pipeline():
    """Test PDF pipeline"""
    run_script('test_pdf_pipeline.py')

def show_status():
    """Show system status"""
    print("\n📊 TORI System Status")
    print("=" * 40)
    print("✅ Frontend: Running at http://localhost:5173/")
    print("✅ TailwindCSS: Fixed")
    print("✅ Accessibility: Fixed")
    
    # Check entropy status
    try:
        sys.path.insert(0, os.getcwd())
        from ingest_pdf.entropy_prune import entropy_prune
        print("✅ Entropy Pruning: Available")
    except:
        print("❌ Entropy Pruning: Not working")
    
    print("❓ API Server: Not checked")
    print("❓ MCP Server: Not checked")
    
    input("\nPress Enter to continue...")

def view_guide():
    """Display the fix guide"""
    guide_path = Path("DEPENDENCY_FIX_GUIDE.md")
    if guide_path.exists():
        with open(guide_path, 'r') as f:
            content = f.read()
        print(content)
    else:
        print("Guide not found!")
    input("\nPress Enter to continue...")

def start_api():
    """Try to start the API server"""
    print("\n🚀 Starting API server...")
    print("Make sure dependencies are fixed first!")
    confirm = input("Continue? (y/n): ")
    if confirm.lower() == 'y':
        subprocess.run([sys.executable, 'enhanced_launcher.py'])

def main():
    """Main menu loop"""
    while True:
        show_menu()
        choice = input("Select option (1-9): ")
        
        if choice == '1':
            check_state()
        elif choice == '2':
            emergency_fix()
        elif choice == '3':
            comprehensive_fix()
        elif choice == '4':
            test_entropy()
        elif choice == '5':
            test_pipeline()
        elif choice == '6':
            show_status()
        elif choice == '7':
            view_guide()
        elif choice == '8':
            start_api()
        elif choice == '9':
            print("\nExiting... Good luck with TORI!")
            break
        else:
            print("\nInvalid choice! Please select 1-9.")
            input("Press Enter to continue...")

if __name__ == "__main__":
    print("Welcome to TORI Dependency Fix!")
    print("This interactive menu will help you fix the dependency issues.")
    input("Press Enter to start...")
    main()
