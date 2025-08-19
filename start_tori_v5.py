#!/usr/bin/env python3
"""
TORI/Saigon v5 Startup Manager
==============================
Choose how to launch the system
"""

import sys
import subprocess
from pathlib import Path

def print_banner():
    print("""
╔══════════════════════════════════════════════════════════╗
║             🚀 TORI/SAIGON v5.0.0 🚀                    ║
║        Production-Ready Self-Improving AI               ║
╚══════════════════════════════════════════════════════════╝
    """)

def main():
    print_banner()
    
    print("Launch Options:")
    print("1. Use NEW v5 launcher (recommended)")
    print("2. Patch existing enhanced_launcher.py with v5 support")
    print("3. Run existing enhanced_launcher.py (without v5)")
    print("4. Start API server only")
    print("5. Run interactive demo")
    print("6. Exit")
    
    choice = input("\nSelect option (1-6): ").strip()
    
    if choice == "1":
        # Use new v5 launcher
        print("\n🚀 Starting with v5 launcher...")
        if Path("enhanced_launcher_v5.py").exists():
            subprocess.run([sys.executable, "enhanced_launcher_v5.py"])
        else:
            print("❌ enhanced_launcher_v5.py not found")
    
    elif choice == "2":
        # Patch existing launcher
        print("\n🔧 Patching existing launcher...")
        subprocess.run([sys.executable, "patch_launcher_v5.py"])
        
        print("\n✅ Patch applied! Starting launcher...")
        subprocess.run([sys.executable, "enhanced_launcher.py"])
    
    elif choice == "3":
        # Run existing launcher
        print("\n🚀 Starting existing launcher...")
        subprocess.run([sys.executable, "enhanced_launcher.py"])
    
    elif choice == "4":
        # API only
        print("\n🌐 Starting API server only...")
        subprocess.run([sys.executable, "api/saigon_inference_api_v5.py"])
    
    elif choice == "5":
        # Interactive demo
        print("\n🎮 Starting interactive demo...")
        subprocess.run([
            sys.executable, 
            "scripts/demo_inference_v5.py",
            "--mode", "interactive"
        ])
    
    elif choice == "6":
        print("\n👋 Goodbye!")
        return 0
    
    else:
        print("\n❌ Invalid option")
        return 1

if __name__ == "__main__":
    sys.exit(main())
