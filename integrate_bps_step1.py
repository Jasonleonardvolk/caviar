#!/usr/bin/env python3
"""
BPS Integration Step 1: Update enhanced_launcher.py
This script patches enhanced_launcher.py to support BPS solitons
"""

import sys
from pathlib import Path

def integrate_bps_into_launcher():
    """Add BPS support to enhanced_launcher.py"""
    
    launcher_path = Path("enhanced_launcher.py")
    
    if not launcher_path.exists():
        print(f"❌ {launcher_path} not found!")
        return False
    
    print("📝 Reading enhanced_launcher.py...")
    with open(launcher_path, 'r') as f:
        content = f.read()
    
    # Check if already integrated
    if "BPSEnhancedLattice" in content:
        print("✅ BPS support already integrated!")
        return True
    
    print("🔧 Adding BPS imports...")
    
    # Find the import section (after the existing imports)
    import_marker = "from python.core.oscillator_lattice import"
    
    if import_marker not in content:
        print("⚠️ Could not find oscillator_lattice import. Adding at top of file...")
        import_section = '''
# BPS Soliton Support
from python.core.bps_config_enhanced import BPS_CONFIG, SolitonPolarity
from python.core.bps_oscillator_enhanced import BPSEnhancedLattice
from python.core.bps_soliton_memory_enhanced import BPSEnhancedSolitonMemory
from python.monitoring.bps_diagnostics import BPSDiagnostics

'''
        # Add after the initial imports
        content = content.replace("import logging", "import logging\n" + import_section)
    else:
        # Add BPS imports after the oscillator import
        bps_imports = '''from python.core.oscillator_lattice import OscillatorLattice, get_global_lattice

# BPS Soliton Support
from python.core.bps_config_enhanced import BPS_CONFIG, SolitonPolarity
from python.core.bps_oscillator_enhanced import BPSEnhancedLattice
from python.core.bps_soliton_memory_enhanced import BPSEnhancedSolitonMemory
from python.monitoring.bps_diagnostics import BPSDiagnostics
'''
        content = content.replace(
            "from python.core.oscillator_lattice import OscillatorLattice, get_global_lattice",
            bps_imports
        )
    
    print("🔧 Adding BPS lattice initialization logic...")
    
    # Find where the lattice is initialized
    lattice_init_marker = "lattice = OscillatorLattice("
    
    if lattice_init_marker in content:
        # Replace with BPS-aware initialization
        old_init = "lattice = OscillatorLattice("
        new_init = '''# Initialize lattice (BPS-enhanced if enabled)
    if "--enable-bps" in sys.argv or BPS_CONFIG.enable_bps:
        lattice = BPSEnhancedLattice(
        logger.info("🌀 BPS-Enhanced Lattice enabled")
    else:
        lattice = OscillatorLattice('''
        
        content = content.replace(old_init, new_init)
    
    print("🔧 Adding BPS diagnostics...")
    
    # Add diagnostics initialization after lattice creation
    diagnostics_code = '''
    # Initialize BPS diagnostics if using BPS lattice
    bps_diagnostics = None
    if isinstance(lattice, BPSEnhancedLattice):
        bps_diagnostics = BPSDiagnostics(lattice)
        logger.info("🔬 BPS diagnostics initialized")
'''
    
    # Find a good place to add diagnostics (after lattice creation)
    if "lattice.start()" in content and diagnostics_code not in content:
        content = content.replace(
            "lattice.start()",
            "lattice.start()" + diagnostics_code
        )
    
    print("🔧 Adding BPS command-line argument...")
    
    # Add to argument parser if it exists
    if "argparse.ArgumentParser" in content:
        # Find the parser section
        parser_marker = 'parser.add_argument("--api"'
        if parser_marker in content:
            bps_arg = '''
    parser.add_argument(
        "--enable-bps",
        action="store_true",
        help="Enable BPS (Bogomolnyi-Prasad-Sommerfield) soliton support"
    )
    parser.add_argument("--api"'''
            content = content.replace(
                'parser.add_argument("--api"',
                bps_arg
            )
    
    # Create backup
    backup_path = launcher_path.with_suffix('.py.bps_backup')
    print(f"💾 Creating backup at {backup_path}...")
    with open(backup_path, 'w') as f:
        f.write(content)
    
    # Write updated content
    print(f"💾 Writing updated {launcher_path}...")
    with open(launcher_path, 'w') as f:
        f.write(content)
    
    print("✅ BPS integration complete!")
    print("\nYou can now run:")
    print("  python enhanced_launcher.py --enable-bps")
    print("\nOr set BPS_CONFIG.enable_bps = True in bps_config_enhanced.py")
    
    return True

if __name__ == "__main__":
    success = integrate_bps_into_launcher()
    sys.exit(0 if success else 1)
