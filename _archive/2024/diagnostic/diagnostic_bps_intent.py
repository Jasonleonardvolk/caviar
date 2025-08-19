#!/usr/bin/env python3
"""
Comprehensive BPS and Intent Module Diagnostic
Generates detailed log for decision making
"""

import subprocess
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
import traceback

# Create logs directory if it doesn't exist
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Setup logging
log_file = log_dir / f"bps_intent_diagnostic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_module_implementation(module_path: Path) -> dict:
    """Check if a module is properly implemented or using stubs"""
    result = {
        "path": str(module_path),
        "exists": module_path.exists(),
        "has_stubs": False,
        "has_implementation": False,
        "classes": [],
        "functions": [],
        "stub_references": [],
        "issues": []
    }
    
    if not module_path.exists():
        result["issues"].append(f"File does not exist: {module_path}")
        return result
    
    try:
        with open(module_path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
        
        # Check for stub indicators
        stub_keywords = ['stub', 'mock', 'placeholder', 'not implemented', 'TODO', 'FIXME', 'pass  # stub']
        for i, line in enumerate(lines):
            for keyword in stub_keywords:
                if keyword.lower() in line.lower():
                    result["stub_references"].append({
                        "line": i + 1,
                        "content": line.strip()[:100],
                        "keyword": keyword
                    })
                    result["has_stubs"] = True
        
        # Check for actual implementations
        import_count = content.count('import ')
        class_count = content.count('class ')
        def_count = content.count('def ')
        
        # Extract class names
        for line in lines:
            if line.strip().startswith('class '):
                class_name = line.split('class ')[1].split('(')[0].split(':')[0].strip()
                result["classes"].append(class_name)
        
        # Extract function names (top-level only)
        for line in lines:
            if line.startswith('def '):
                func_name = line.split('def ')[1].split('(')[0].strip()
                result["functions"].append(func_name)
        
        # Determine if it has real implementation
        if class_count > 0 or def_count > 5:  # More than 5 functions suggests real implementation
            result["has_implementation"] = True
        
        # Check for warning logs about stubs
        if "logger.warning" in content and "stub" in content.lower():
            result["issues"].append("Contains warning about stub usage")
        
    except Exception as e:
        result["issues"].append(f"Error reading file: {str(e)}")
    
    return result

def main():
    logger.info("="*60)
    logger.info("BPS AND INTENT MODULE COMPREHENSIVE DIAGNOSTIC")
    logger.info("="*60)
    
    diagnostic_data = {
        "timestamp": datetime.now().isoformat(),
        "bps_integration": {},
        "intent_module": {},
        "stub_analysis": {},
        "recommendations": []
    }
    
    # 1. Check BPS Integration
    logger.info("\n1. CHECKING BPS INTEGRATION")
    logger.info("-"*40)
    
    launcher_path = Path("enhanced_launcher.py")
    if launcher_path.exists():
        with open(launcher_path, 'r', encoding='utf-8') as f:
            launcher_content = f.read()
        
        bps_checks = {
            "--enable-bps argument": "--enable-bps" in launcher_content,
            "BPSEnhancedLattice import": "BPSEnhancedLattice" in launcher_content,
            "BPS_CONFIG import": "BPS_CONFIG" in launcher_content,
            "BPSDiagnostics import": "BPSDiagnostics" in launcher_content,
            "BPS lattice initialization": "if args.enable_bps" in launcher_content or "if \"--enable-bps\"" in launcher_content
        }
        
        for check, status in bps_checks.items():
            status_str = "✅ FOUND" if status else "❌ MISSING"
            logger.info(f"  {check}: {status_str}")
            diagnostic_data["bps_integration"][check] = status
        
        # Overall BPS status
        bps_ready = all(bps_checks.values())
        diagnostic_data["bps_integration"]["ready"] = bps_ready
        
        if bps_ready:
            logger.info("  🎉 BPS FULLY INTEGRATED")
        else:
            logger.warning("  ⚠️ BPS PARTIALLY INTEGRATED")
            diagnostic_data["recommendations"].append("Complete BPS integration in enhanced_launcher.py")
    else:
        logger.error("  ❌ enhanced_launcher.py not found!")
        diagnostic_data["bps_integration"]["ready"] = False
    
    # 2. Check Intent-Driven Reasoning Module
    logger.info("\n2. CHECKING INTENT-DRIVEN REASONING MODULE")
    logger.info("-"*40)
    
    intent_modules = [
        Path("python/core/intent_driven_reasoning.py"),
        Path("python/core/intent_router.py"),
        Path("python/core/intent_router_lib/__init__.py")
    ]
    
    for module_path in intent_modules:
        logger.info(f"\n  Analyzing: {module_path}")
        analysis = check_module_implementation(module_path)
        diagnostic_data["intent_module"][module_path.name] = analysis
        
        if analysis["exists"]:
            logger.info(f"    Classes found: {len(analysis['classes'])}")
            if analysis['classes']:
                logger.info(f"      {', '.join(analysis['classes'][:3])}")
            
            logger.info(f"    Functions found: {len(analysis['functions'])}")
            if analysis['functions']:
                logger.info(f"      {', '.join(analysis['functions'][:3])}")
            
            if analysis["has_stubs"]:
                logger.warning(f"    ⚠️ STUB REFERENCES FOUND: {len(analysis['stub_references'])}")
                for ref in analysis['stub_references'][:3]:
                    logger.warning(f"      Line {ref['line']}: {ref['content'][:60]}")
            
            if analysis["has_implementation"]:
                logger.info(f"    ✅ HAS REAL IMPLEMENTATION")
            else:
                logger.warning(f"    ⚠️ APPEARS TO BE STUB/MINIMAL")
        else:
            logger.error(f"    ❌ FILE DOES NOT EXIST")
    
    # 3. Stub Analysis Across Core Modules
    logger.info("\n3. STUB ANALYSIS ACROSS CORE MODULES")
    logger.info("-"*40)
    
    core_dir = Path("python/core")
    stub_count = 0
    modules_with_stubs = []
    
    for py_file in core_dir.glob("*.py"):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for stub warning
            if "using stubs" in content.lower() or "not available" in content and "stub" in content.lower():
                stub_count += 1
                modules_with_stubs.append(py_file.name)
                
                # Find the specific line
                for i, line in enumerate(content.split('\n')):
                    if "using stubs" in line.lower():
                        logger.warning(f"  {py_file.name}:{i+1} - {line.strip()[:80]}")
                        break
        except Exception as e:
            logger.error(f"  Error reading {py_file.name}: {e}")
    
    diagnostic_data["stub_analysis"]["total_modules_with_stubs"] = stub_count
    diagnostic_data["stub_analysis"]["modules"] = modules_with_stubs
    
    logger.info(f"\n  Total modules with stub references: {stub_count}")
    if modules_with_stubs:
        logger.warning(f"  Modules: {', '.join(modules_with_stubs[:5])}")
    
    # 4. Test Imports
    logger.info("\n4. TESTING MODULE IMPORTS")
    logger.info("-"*40)
    
    test_imports = [
        ("BPS Config", "from python.core.bps_config_enhanced import BPS_CONFIG"),
        ("BPS Lattice", "from python.core.bps_oscillator_enhanced import BPSEnhancedLattice"),
        ("Intent Router", "from python.core.intent_router import IntentRouter"),
        ("Intent Reasoning", "from python.core.intent_driven_reasoning import IntentDrivenReasoning")
    ]
    
    for name, import_stmt in test_imports:
        try:
            exec(import_stmt)
            logger.info(f"  ✅ {name}: Import successful")
            diagnostic_data["intent_module"][f"{name}_import"] = "success"
        except ImportError as e:
            logger.error(f"  ❌ {name}: Import failed - {e}")
            diagnostic_data["intent_module"][f"{name}_import"] = f"failed: {str(e)}"
        except Exception as e:
            logger.error(f"  ❌ {name}: Unexpected error - {e}")
            diagnostic_data["intent_module"][f"{name}_import"] = f"error: {str(e)}"
    
    # 5. Recommendations
    logger.info("\n5. ANALYSIS & RECOMMENDATIONS")
    logger.info("-"*40)
    
    # Determine deliverable status
    bps_ready = diagnostic_data["bps_integration"].get("ready", False)
    intent_has_impl = any(
        m.get("has_implementation", False) 
        for m in diagnostic_data["intent_module"].values() 
        if isinstance(m, dict)
    )
    intent_has_stubs = any(
        m.get("has_stubs", False) 
        for m in diagnostic_data["intent_module"].values() 
        if isinstance(m, dict)
    )
    
    # Time estimates
    if bps_ready:
        logger.info("  ✅ BPS: READY TO TEST (0 hours)")
        diagnostic_data["recommendations"].append("BPS is ready for immediate testing")
    else:
        logger.info("  ⏱️ BPS: Needs minor fixes (~0.5 hours)")
        diagnostic_data["recommendations"].append("Fix BPS integration in enhanced_launcher.py")
    
    if intent_has_impl and not intent_has_stubs:
        logger.info("  ✅ INTENT: Fully implemented (0 hours)")
        diagnostic_data["recommendations"].append("Intent module is ready to use")
    elif intent_has_impl and intent_has_stubs:
        logger.info("  ⏱️ INTENT: Partially implemented (~2-4 hours to complete)")
        diagnostic_data["recommendations"].append("Remove stub references from intent module")
        diagnostic_data["recommendations"].append("Complete partial implementations")
    else:
        logger.info("  ⏱️ INTENT: Needs full implementation (~8-16 hours)")
        diagnostic_data["recommendations"].append("Intent module needs complete implementation")
    
    # Priority recommendation
    logger.info("\n  📊 PRIORITY RECOMMENDATION:")
    if bps_ready:
        logger.info("    1. TEST BPS immediately (it's ready)")
        logger.info("    2. Fix intent module stubs in parallel")
        diagnostic_data["recommendations"].insert(0, "PRIORITY: Test BPS system immediately")
    else:
        logger.info("    1. Quick fix for BPS integration (30 min)")
        logger.info("    2. Test BPS system")
        logger.info("    3. Then address intent module")
        diagnostic_data["recommendations"].insert(0, "PRIORITY: Fix BPS integration first (quick win)")
    
    # 6. Save detailed JSON report
    json_file = log_dir / f"diagnostic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_file, 'w') as f:
        json.dump(diagnostic_data, f, indent=2, default=str)
    
    logger.info(f"\n📊 Detailed JSON report saved to: {json_file}")
    logger.info(f"📝 Log file saved to: {log_file}")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("EXECUTIVE SUMMARY")
    logger.info("="*60)
    logger.info(f"BPS Integration: {'READY' if bps_ready else 'NEEDS FIXES'}")
    logger.info(f"Intent Module: {'IMPLEMENTED' if intent_has_impl else 'NEEDS WORK'}")
    logger.info(f"Stub Issues: {stub_count} modules with stub references")
    logger.info(f"Recommended Next Step: {diagnostic_data['recommendations'][0]}")

if __name__ == "__main__":
    main()
