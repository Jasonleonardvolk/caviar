#!/usr/bin/env python3
"""
beyond_diagnostics.py - Health check and diagnostic tool for Beyond Metacognition

Checks:
- Component availability
- File integrity
- Performance metrics
- Resource usage
- Common issues
"""

import sys
import os
import json
import psutil
import time
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any

class BeyondDiagnostics:
    """Diagnostic tool for Beyond Metacognition system"""
    
    def __init__(self):
        self.root = Path(__file__).parent
        self.report = {
            "timestamp": datetime.now().isoformat(),
            "checks": {},
            "issues": [],
            "recommendations": []
        }
        
    def check_file_structure(self) -> Tuple[bool, List[str]]:
        """Check all required files exist"""
        print("\n📁 Checking file structure...")
        
        required_files = {
            # Core components
            "alan_backend/origin_sentry.py": "OriginSentry component",
            "alan_backend/braid_aggregator.py": "Braid Aggregator",
            "python/core/braid_buffers.py": "Temporal Braiding Engine",
            "python/core/observer_synthesis.py": "Observer-Observed Synthesis",
            "python/core/creative_feedback.py": "Creative Feedback Loop",
            "python/core/topology_tracker.py": "Topology tracking stub",
            
            # Patched files
            "alan_backend/eigensentry_guard.py": "EigenSentry (needs patching)",
            "python/core/chaos_control_layer.py": "Chaos Control Layer (needs patching)",
            "tori_master.py": "TORI Master (needs patching)",
            "services/metrics_ws.py": "WebSocket service (needs patching)",
            
            # Tools
            "apply_beyond_patches.py": "Patch application script",
            "verify_beyond_integration.py": "Verification script",
            "beyond_demo.py": "Demo scenarios",
            "torictl.py": "CLI tool"
        }
        
        missing = []
        for file_path, description in required_files.items():
            full_path = self.root / file_path
            if not full_path.exists():
                missing.append(f"{file_path} ({description})")
                print(f"  ❌ Missing: {file_path}")
            else:
                print(f"  ✅ Found: {file_path}")
        
        success = len(missing) == 0
        self.report['checks']['file_structure'] = {
            'success': success,
            'missing_files': missing
        }
        
        if missing:
            self.report['issues'].append(f"Missing {len(missing)} required files")
            self.report['recommendations'].append(
                "Re-run the Beyond Metacognition setup to create missing files"
            )
        
        return success, missing
    
    def check_patches_applied(self) -> Tuple[bool, Dict[str, bool]]:
        """Check if patches have been applied"""
        print("\n🔧 Checking patch status...")
        
        patch_indicators = {
            "alan_backend/eigensentry_guard.py": [
                "from alan_backend.origin_sentry import OriginSentry",
                "self.origin_sentry = OriginSentry()"
            ],
            "python/core/chaos_control_layer.py": [
                "from python.core.braid_buffers import get_braiding_engine",
                "self.braiding_engine = get_braiding_engine()"
            ],
            "tori_master.py": [
                "from alan_backend.braid_aggregator import BraidAggregator",
                "self.braid_aggregator = BraidAggregator()"
            ],
            "services/metrics_ws.py": [
                "beyond_metacognition",
                "origin_sentry"
            ]
        }
        
        patch_status = {}
        all_patched = True
        
        for file_path, indicators in patch_indicators.items():
            full_path = self.root / file_path
            if full_path.exists():
                try:
                    content = full_path.read_text(encoding='utf-8')
                    patched = all(indicator in content for indicator in indicators)
                    patch_status[file_path] = patched
                    
                    status = "✅ Patched" if patched else "❌ Not patched"
                    print(f"  {status}: {file_path}")
                    
                    if not patched:
                        all_patched = False
                except Exception as e:
                    patch_status[file_path] = False
                    print(f"  ❌ Error reading {file_path}: {e}")
                    all_patched = False
            else:
                patch_status[file_path] = False
                all_patched = False
        
        self.report['checks']['patches'] = {
            'success': all_patched,
            'file_status': patch_status
        }
        
        if not all_patched:
            self.report['issues'].append("Some files are not patched")
            self.report['recommendations'].append(
                "Run: python apply_beyond_patches.py --verify"
            )
        
        return all_patched, patch_status
    
    def check_imports(self) -> Tuple[bool, List[str]]:
        """Check if all components can be imported"""
        print("\n📦 Checking component imports...")
        
        components = [
            ("alan_backend.origin_sentry", "OriginSentry"),
            ("alan_backend.braid_aggregator", "BraidAggregator"),
            ("python.core.braid_buffers", "TemporalBraidingEngine"),
            ("python.core.observer_synthesis", "ObserverObservedSynthesis"),
            ("python.core.creative_feedback", "CreativeSingularityFeedback"),
            ("python.core.topology_tracker", "compute_betti_numbers")
        ]
        
        import_errors = []
        
        # Add root to path
        sys.path.insert(0, str(self.root))
        
        for module_name, class_name in components:
            try:
                module = __import__(module_name, fromlist=[class_name])
                if hasattr(module, class_name):
                    print(f"  ✅ Imported: {module_name}.{class_name}")
                else:
                    import_errors.append(f"{module_name} missing {class_name}")
                    print(f"  ❌ Missing class: {class_name} in {module_name}")
            except ImportError as e:
                import_errors.append(f"{module_name}: {str(e)}")
                print(f"  ❌ Import error: {module_name}")
                print(f"     {str(e)}")
        
        success = len(import_errors) == 0
        self.report['checks']['imports'] = {
            'success': success,
            'errors': import_errors
        }
        
        if import_errors:
            self.report['issues'].append(f"Import errors: {len(import_errors)}")
            
        return success, import_errors
    
    def check_data_files(self) -> Dict[str, Any]:
        """Check data files and storage"""
        print("\n💾 Checking data files...")
        
        data_locations = {
            "spectral_db.json": "OriginSentry spectral database",
            "lyapunov_watchlist.json": "Lyapunov exponent history",
            "braid_buffers/micro_buffer.json": "Micro-scale temporal buffer",
            "braid_buffers/meso_buffer.json": "Meso-scale temporal buffer",
            "braid_buffers/macro_buffer.json": "Macro-scale temporal buffer"
        }
        
        data_status = {}
        total_size = 0
        
        for file_path, description in data_locations.items():
            full_path = self.root / file_path
            if full_path.exists():
                size = full_path.stat().st_size
                total_size += size
                
                # Check if valid JSON
                try:
                    with open(full_path) as f:
                        data = json.load(f)
                    entries = len(data) if isinstance(data, list) else 1
                    data_status[file_path] = {
                        'exists': True,
                        'size': size,
                        'entries': entries,
                        'valid': True
                    }
                    print(f"  ✅ {file_path}: {size/1024:.1f}KB, {entries} entries")
                except:
                    data_status[file_path] = {
                        'exists': True,
                        'size': size,
                        'valid': False
                    }
                    print(f"  ⚠️ {file_path}: Invalid JSON")
            else:
                data_status[file_path] = {'exists': False}
                print(f"  ℹ️ {file_path}: Not created yet")
        
        self.report['checks']['data_files'] = {
            'files': data_status,
            'total_size_kb': total_size / 1024
        }
        
        print(f"\n  Total data size: {total_size/1024:.1f}KB")
        
        return data_status
    
    def check_performance(self) -> Dict[str, Any]:
        """Check system performance metrics"""
        print("\n⚡ Checking performance...")
        
        # Get system resources
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Test component initialization time
        init_times = {}
        
        try:
            # Time OriginSentry init
            start = time.time()
            from alan_backend.origin_sentry import OriginSentry
            origin = OriginSentry()
            init_times['OriginSentry'] = time.time() - start
            
            # Time classification
            start = time.time()
            import numpy as np
            test_eigenvalues = np.random.randn(10) * 0.02
            origin.classify(test_eigenvalues)
            init_times['Classification'] = time.time() - start
            
        except Exception as e:
            print(f"  ❌ Performance test error: {e}")
        
        perf_data = {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'init_times_ms': {k: v*1000 for k, v in init_times.items()}
        }
        
        self.report['checks']['performance'] = perf_data
        
        print(f"  CPU Usage: {cpu_percent}%")
        print(f"  Memory Usage: {memory.percent}%")
        print(f"  Available Memory: {memory.available/(1024**3):.1f}GB")
        
        if init_times:
            print("\n  Component init times:")
            for component, time_sec in init_times.items():
                print(f"    {component}: {time_sec*1000:.1f}ms")
        
        # Performance recommendations
        if cpu_percent > 80:
            self.report['recommendations'].append(
                "High CPU usage detected. Consider reducing aggregation frequency."
            )
        
        if memory.percent > 80:
            self.report['recommendations'].append(
                "High memory usage. Consider reducing buffer sizes or enabling eviction."
            )
        
        return perf_data
    
    def check_recent_activity(self) -> Dict[str, Any]:
        """Check for recent Beyond Metacognition activity"""
        print("\n📊 Checking recent activity...")
        
        activity = {}
        
        # Check status file
        status_file = self.root / "BEYOND_METACOGNITION_STATUS.json"
        if status_file.exists():
            try:
                with open(status_file) as f:
                    status = json.load(f)
                
                last_update = status.get('timestamp', 'Unknown')
                activity['last_patch'] = last_update
                print(f"  Last patch: {last_update}")
                
                # Check if recent (within 24 hours)
                try:
                    update_time = datetime.fromisoformat(last_update)
                    age = datetime.now() - update_time
                    if age > timedelta(days=1):
                        self.report['recommendations'].append(
                            "No recent patches. Consider re-running patches if you've updated TORI."
                        )
                except:
                    pass
                    
            except Exception as e:
                print(f"  ❌ Error reading status: {e}")
        
        # Check for test results
        test_results = self.root / "beyond_integration_test_results.json"
        if test_results.exists():
            try:
                with open(test_results) as f:
                    results = json.load(f)
                
                timestamp = results.get('timestamp', 'Unknown')
                summary = results.get('summary', {})
                
                activity['last_test'] = timestamp
                activity['test_results'] = summary
                
                print(f"  Last test: {timestamp}")
                print(f"  Test results: {summary.get('passed', 0)}/{summary.get('total_tests', 0)} passed")
                
            except Exception as e:
                print(f"  ❌ Error reading test results: {e}")
        
        self.report['checks']['activity'] = activity
        
        return activity
    
    def generate_report(self):
        """Generate final diagnostic report"""
        print("\n" + "="*60)
        print("DIAGNOSTIC SUMMARY")
        print("="*60)
        
        # Count successes
        successes = sum(1 for check in self.report['checks'].values() 
                       if isinstance(check, dict) and check.get('success', True))
        total_checks = len(self.report['checks'])
        
        print(f"\n✅ Passed: {successes}/{total_checks} checks")
        
        if self.report['issues']:
            print(f"\n⚠️ Issues found: {len(self.report['issues'])}")
            for issue in self.report['issues']:
                print(f"  - {issue}")
        
        if self.report['recommendations']:
            print(f"\n💡 Recommendations:")
            for rec in self.report['recommendations']:
                print(f"  - {rec}")
        
        # Overall health
        if successes == total_checks and not self.report['issues']:
            print("\n✅ SYSTEM HEALTHY")
            print("Beyond Metacognition is fully operational!")
        elif successes >= total_checks - 1:
            print("\n⚠️ MINOR ISSUES")
            print("System is mostly functional but has minor issues.")
        else:
            print("\n❌ MAJOR ISSUES")
            print("System requires attention before it can function properly.")
        
        # Save report
        report_file = self.root / "beyond_diagnostics_report.json"
        with open(report_file, 'w') as f:
            json.dump(self.report, f, indent=2)
        
        print(f"\n📝 Full report saved to: {report_file}")

def main():
    """Run diagnostics"""
    print("🔍 Beyond Metacognition Diagnostics")
    print("="*60)
    
    diagnostics = BeyondDiagnostics()
    
    # Run all checks
    diagnostics.check_file_structure()
    diagnostics.check_patches_applied()
    diagnostics.check_imports()
    diagnostics.check_data_files()
    diagnostics.check_performance()
    diagnostics.check_recent_activity()
    
    # Generate report
    diagnostics.generate_report()
    
    # Quick fix suggestions
    print("\n🛠️ Quick fixes:")
    print("  1. Missing files? Re-run the Beyond Metacognition setup")
    print("  2. Not patched? Run: python apply_beyond_patches.py --verify")
    print("  3. Import errors? Check that all component files were created")
    print("  4. Performance issues? Adjust buffer sizes and aggregation intervals")

if __name__ == "__main__":
    main()
