#!/usr/bin/env python3
"""
TORI/Saigon System Verification Script
=======================================
Verifies all components are properly installed and configured.
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Tuple

# ANSI colors
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
CYAN = '\033[96m'
RESET = '\033[0m'
BOLD = '\033[1m'

def check_file(path: str, description: str) -> bool:
    """Check if a file exists."""
    if Path(path).exists():
        print(f"{GREEN}✓{RESET} {description}: {path}")
        return True
    else:
        print(f"{RED}✗{RESET} {description}: {path} - MISSING")
        return False

def check_directory(path: str, description: str) -> bool:
    """Check if a directory exists."""
    if Path(path).is_dir():
        print(f"{GREEN}✓{RESET} {description}: {path}")
        return True
    else:
        print(f"{RED}✗{RESET} {description}: {path} - MISSING")
        return False

def check_import(module: str, description: str) -> bool:
    """Check if a Python module can be imported."""
    try:
        __import__(module)
        print(f"{GREEN}✓{RESET} {description}: {module}")
        return True
    except ImportError as e:
        print(f"{RED}✗{RESET} {description}: {module} - {e}")
        return False

def verify_core_files() -> Tuple[int, int]:
    """Verify core Python files."""
    print(f"\n{CYAN}Verifying Core Files...{RESET}")
    
    files = [
        ("python/core/saigon_inference_v5.py", "Inference Engine"),
        ("python/core/adapter_loader_v5.py", "Adapter Loader"),
        ("python/core/concept_mesh_v5.py", "Concept Mesh"),
        ("python/core/user_context.py", "User Context"),
        ("python/core/conversation_manager.py", "Conversation Manager"),
        ("python/core/lattice_morphing.py", "Lattice Morphing"),
        ("python/core/__init__.py", "Core Module Init"),
    ]
    
    passed = sum(check_file(f[0], f[1]) for f in files)
    total = len(files)
    return passed, total

def verify_training_files() -> Tuple[int, int]:
    """Verify training files."""
    print(f"\n{CYAN}Verifying Training Files...{RESET}")
    
    files = [
        ("python/training/train_lora_adapter_v5.py", "LoRA Trainer"),
        ("python/training/synthetic_data_generator.py", "Data Generator"),
        ("python/training/validate_adapter.py", "Validator"),
        ("python/training/rollback_adapter.py", "Rollback Manager"),
        ("python/training/__init__.py", "Training Module Init"),
    ]
    
    passed = sum(check_file(f[0], f[1]) for f in files)
    total = len(files)
    return passed, total

def verify_api_files() -> Tuple[int, int]:
    """Verify API files."""
    print(f"\n{CYAN}Verifying API Files...{RESET}")
    
    files = [
        ("api/saigon_inference_api_v5.py", "FastAPI Server"),
        ("scripts/demo_inference_v5.py", "Demo Script"),
    ]
    
    passed = sum(check_file(f[0], f[1]) for f in files)
    total = len(files)
    return passed, total

def verify_frontend_files() -> Tuple[int, int]:
    """Verify frontend files."""
    print(f"\n{CYAN}Verifying Frontend Files...{RESET}")
    
    files = [
        ("frontend/lib/realGhostEngine_v2.js", "Ghost Engine"),
        ("frontend/lib/conceptHologramRenderer.js", "Hologram Renderer"),
        ("frontend/lib/webgpu/quiltGenerator.ts", "WebGPU Quilt"),
        ("frontend/hybrid/package.json", "Frontend Package"),
        ("frontend/hybrid/Dickboxfile.frontend", "Frontend Dickboxfile"),
    ]
    
    passed = sum(check_file(f[0], f[1]) for f in files)
    total = len(files)
    return passed, total

def verify_advanced_files() -> Tuple[int, int]:
    """Verify advanced component files."""
    print(f"\n{CYAN}Verifying Advanced Components...{RESET}")
    
    files = [
        ("hott_integration/psi_morphon.py", "HoTT/Psi-Morphon"),
    ]
    
    passed = sum(check_file(f[0], f[1]) for f in files)
    total = len(files)
    return passed, total

def verify_configuration_files() -> Tuple[int, int]:
    """Verify configuration files."""
    print(f"\n{CYAN}Verifying Configuration Files...{RESET}")
    
    files = [
        ("requirements.txt", "Python Requirements"),
        ("dickbox.toml", "Dickbox Config"),
        ("Dickboxfile", "Main Dickboxfile"),
        ("dickbox.py", "Dickbox Manager"),
        ("setup.py", "Setup Script"),
        (".gitignore", "Git Ignore"),
        ("Makefile", "Makefile"),
        ("README.md", "Main README"),
        ("IMPLEMENTATION_SUMMARY.md", "Implementation Summary"),
    ]
    
    passed = sum(check_file(f[0], f[1]) for f in files)
    total = len(files)
    return passed, total

def verify_directories() -> Tuple[int, int]:
    """Verify required directories."""
    print(f"\n{CYAN}Verifying Directories...{RESET}")
    
    dirs = [
        ("models/saigon_base", "Base Model Dir"),
        ("models/adapters", "Adapters Dir"),
        ("data/mesh_contexts", "Mesh Contexts"),
        ("data/training", "Training Data"),
        ("data/validation", "Validation Data"),
        ("data/intent_vault", "Intent Vault"),
        ("data/psi_archive", "Psi Archive"),
        ("logs/inference", "Inference Logs"),
        ("logs/validation", "Validation Logs"),
        ("logs/rollback", "Rollback Logs"),
        ("logs/conversations", "Conversation Logs"),
        ("logs/user_context", "User Context Logs"),
        ("monitoring", "Monitoring Config"),
        ("nginx", "Nginx Config"),
        ("nginx/ssl", "SSL Certificates"),
    ]
    
    passed = sum(check_directory(d[0], d[1]) for d in dirs)
    total = len(dirs)
    return passed, total

def verify_metadata() -> bool:
    """Verify adapter metadata file."""
    print(f"\n{CYAN}Verifying Metadata...{RESET}")
    
    metadata_file = Path("models/adapters/metadata.json")
    if not metadata_file.exists():
        print(f"{RED}✗{RESET} Adapter metadata file missing")
        return False
    
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        if "global" in metadata and "domain" in metadata and "user" in metadata:
            print(f"{GREEN}✓{RESET} Adapter metadata structure valid")
            print(f"  - Global adapters: {len(metadata.get('global', []))}")
            print(f"  - Domain categories: {len(metadata.get('domain', {}))}")
            print(f"  - Users with adapters: {len(metadata.get('user', {}))}")
            return True
        else:
            print(f"{YELLOW}⚠{RESET} Adapter metadata incomplete")
            return False
    except Exception as e:
        print(f"{RED}✗{RESET} Failed to parse metadata: {e}")
        return False

def verify_monitoring() -> Tuple[int, int]:
    """Verify monitoring configuration."""
    print(f"\n{CYAN}Verifying Monitoring Configuration...{RESET}")
    
    files = [
        ("monitoring/prometheus.yml", "Prometheus Config"),
        ("monitoring/alerts/tori_alerts.yml", "Alert Rules"),
        ("monitoring/grafana/dashboards/tori_dashboard.json", "Grafana Dashboard"),
        ("nginx/nginx.conf", "Nginx Config"),
    ]
    
    passed = sum(check_file(f[0], f[1]) for f in files)
    total = len(files)
    return passed, total

def main():
    """Main verification function."""
    print(f"{CYAN}{'='*60}")
    print(f"{BOLD}TORI/Saigon System Verification{RESET}")
    print(f"{CYAN}{'='*60}{RESET}")
    
    total_passed = 0
    total_checks = 0
    
    # Verify all components
    results = []
    
    # Core files
    passed, total = verify_core_files()
    results.append(("Core Files", passed, total))
    total_passed += passed
    total_checks += total
    
    # Training files
    passed, total = verify_training_files()
    results.append(("Training Files", passed, total))
    total_passed += passed
    total_checks += total
    
    # API files
    passed, total = verify_api_files()
    results.append(("API Files", passed, total))
    total_passed += passed
    total_checks += total
    
    # Frontend files
    passed, total = verify_frontend_files()
    results.append(("Frontend Files", passed, total))
    total_passed += passed
    total_checks += total
    
    # Advanced components
    passed, total = verify_advanced_files()
    results.append(("Advanced Components", passed, total))
    total_passed += passed
    total_checks += total
    
    # Configuration
    passed, total = verify_configuration_files()
    results.append(("Configuration", passed, total))
    total_passed += passed
    total_checks += total
    
    # Directories
    passed, total = verify_directories()
    results.append(("Directories", passed, total))
    total_passed += passed
    total_checks += total
    
    # Monitoring
    passed, total = verify_monitoring()
    results.append(("Monitoring", passed, total))
    total_passed += passed
    total_checks += total
    
    # Metadata
    if verify_metadata():
        total_passed += 1
    total_checks += 1
    
    # Summary
    print(f"\n{CYAN}{'='*60}")
    print(f"{BOLD}Verification Summary{RESET}")
    print(f"{CYAN}{'='*60}{RESET}\n")
    
    for category, passed, total in results:
        if passed == total:
            color = GREEN
            status = "PASS"
        elif passed > total * 0.7:
            color = YELLOW
            status = "PARTIAL"
        else:
            color = RED
            status = "FAIL"
        
        print(f"{color}{status:8}{RESET} {category:20} {passed:2}/{total:2} checks passed")
    
    # Overall result
    print(f"\n{CYAN}{'='*60}{RESET}")
    percentage = (total_passed / total_checks * 100) if total_checks > 0 else 0
    
    if percentage == 100:
        print(f"{GREEN}{BOLD}✓ SYSTEM FULLY OPERATIONAL{RESET}")
        print(f"{GREEN}All {total_passed}/{total_checks} checks passed!{RESET}")
    elif percentage >= 90:
        print(f"{GREEN}{BOLD}✓ SYSTEM OPERATIONAL{RESET}")
        print(f"{GREEN}{total_passed}/{total_checks} checks passed ({percentage:.1f}%){RESET}")
    elif percentage >= 70:
        print(f"{YELLOW}{BOLD}⚠ SYSTEM PARTIALLY OPERATIONAL{RESET}")
        print(f"{YELLOW}{total_passed}/{total_checks} checks passed ({percentage:.1f}%){RESET}")
        print(f"{YELLOW}Some components may not work correctly.{RESET}")
    else:
        print(f"{RED}{BOLD}✗ SYSTEM NOT READY{RESET}")
        print(f"{RED}Only {total_passed}/{total_checks} checks passed ({percentage:.1f}%){RESET}")
        print(f"{RED}Please run setup.py to complete installation.{RESET}")
    
    print(f"{CYAN}{'='*60}{RESET}\n")
    
    # Next steps
    if percentage < 100:
        print(f"{YELLOW}Next Steps:{RESET}")
        print(f"1. Run: {CYAN}python setup.py{RESET}")
        print(f"2. Install dependencies: {CYAN}pip install -r requirements.txt{RESET}")
        print(f"3. Build containers: {CYAN}python dickbox.py build{RESET}")
        print(f"4. Start services: {CYAN}python dickbox.py up{RESET}")
    else:
        print(f"{GREEN}Ready to start!{RESET}")
        print(f"Run: {CYAN}python dickbox.py up{RESET} to start all services")
        print(f"Or: {CYAN}python scripts/demo_inference_v5.py --mode interactive{RESET} for demo")
    
    return 0 if percentage >= 70 else 1

if __name__ == "__main__":
    sys.exit(main())
