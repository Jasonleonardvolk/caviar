#!/usr/bin/env python3
"""
TORI Pre-Flight Validation System
Checks all prerequisites before starting TORI to prevent common failures
"""

import subprocess
import sys
import socket
import json
from pathlib import Path
from datetime import datetime

class PreFlightValidator:
    def __init__(self):
        self.errors = {"CRITICAL": [], "HIGH": [], "MEDIUM": [], "LOW": []}
        self.required_packages = {
            'torch': {'priority': 'CRITICAL', 'install': 'pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu'},
            'deepdiff': {'priority': 'CRITICAL', 'install': 'pip install deepdiff'},
            'sympy': {'priority': 'HIGH', 'install': 'pip install sympy'},
            'PyPDF2': {'priority': 'HIGH', 'install': 'pip install PyPDF2'},
            'websockets': {'priority': 'CRITICAL', 'install': 'pip install websockets'},
            'fastapi': {'priority': 'CRITICAL', 'install': 'pip install fastapi[all]'},
            'celery': {'priority': 'HIGH', 'install': 'pip install celery[redis]'},
            'redis': {'priority': 'HIGH', 'install': 'pip install redis'},
            'aiohttp': {'priority': 'CRITICAL', 'install': 'pip install aiohttp'},
            'uvicorn': {'priority': 'CRITICAL', 'install': 'pip install uvicorn[standard]'},
            'psutil': {'priority': 'MEDIUM', 'install': 'pip install psutil'}
        }
        
    def check_python_packages(self):
        """Validate all required Python packages"""
        print("\n📦 Checking Python packages...")
        for package, info in self.required_packages.items():
            try:
                __import__(package)
                print(f"  ✅ {package} - OK")
            except ImportError:
                self.errors[info['priority']].append({
                    'type': 'missing_package',
                    'package': package,
                    'fix': info['install']
                })
                print(f"  ❌ {package} - MISSING")
    
    def check_ports(self):
        """Check if required ports are available"""
        print("\n🔌 Checking port availability...")
        ports = {
            8002: "API Server",
            8100: "MCP Metacognitive Server", 
            8765: "Audio Bridge",
            8766: "Concept Mesh Bridge",
            5173: "Frontend Dev Server",
            6379: "Redis"
        }
        
        for port, service in ports.items():
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            
            if result == 0:
                self.errors['HIGH'].append({
                    'type': 'port_conflict',
                    'port': port,
                    'service': service,
                    'fix': f'Kill process on port {port} or use alternate port'
                })
                print(f"  ❌ Port {port} ({service}) - IN USE")
            else:
                print(f"  ✅ Port {port} ({service}) - AVAILABLE")
    
    def check_system_resources(self):
        """Validate system has sufficient resources"""
        print("\n💻 Checking system resources...")
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Check memory
            memory_gb = memory.available / (1024**3)
            if memory_gb < 8:
                self.errors['HIGH'].append({
                    'type': 'insufficient_memory',
                    'available': f"{memory_gb:.1f}GB",
                    'fix': 'Close other applications or add more RAM'
                })
                print(f"  ⚠️  Available memory: {memory_gb:.1f}GB (recommended: 8GB+)")
            else:
                print(f"  ✅ Available memory: {memory_gb:.1f}GB")
            
            # Check disk space
            disk_gb = disk.free / (1024**3)
            if disk_gb < 10:
                self.errors['MEDIUM'].append({
                    'type': 'low_disk_space',
                    'available': f"{disk_gb:.1f}GB",
                    'fix': 'Free up disk space'
                })
                print(f"  ⚠️  Free disk space: {disk_gb:.1f}GB (recommended: 10GB+)")
            else:
                print(f"  ✅ Free disk space: {disk_gb:.1f}GB")
                
        except ImportError:
            print("  ⚠️  psutil not installed - skipping resource checks")
    
    def check_configuration_files(self):
        """Check for required configuration files"""
        print("\n📄 Checking configuration files...")
        
        required_files = {
            "vite.config.js": "tori_ui_svelte/vite.config.js",
            "package.json": "tori_ui_svelte/package.json",
            ".env": ".env"
        }
        
        for name, path in required_files.items():
            full_path = Path(path)
            if full_path.exists():
                print(f"  ✅ {name} exists")
            else:
                self.errors['MEDIUM'].append({
                    'type': 'missing_config',
                    'file': name,
                    'path': path,
                    'fix': f'Create {path} file'
                })
                print(f"  ❌ {name} missing")
    
    def generate_report(self):
        """Generate prioritized error report with fixes"""
        if not any(self.errors.values()):
            print("\n✅ All pre-flight checks passed!")
            return True
        
        print("\n" + "="*60)
        print("❌ PRE-FLIGHT VALIDATION FAILED!")
        print("="*60)
        
        total_issues = sum(len(errors) for errors in self.errors.values())
        print(f"\nTotal issues found: {total_issues}")
        
        for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
            if self.errors[severity]:
                print(f"\n{severity} ERRORS ({len(self.errors[severity])}):")
                print("-" * 40)
                for error in self.errors[severity]:
                    print(f"  • {error['type']}: {error.get('package', error.get('port', error.get('file', 'N/A')))}")
                    print(f"    Fix: {error['fix']}")
        
        # Save detailed report
        report_path = f"preflight_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'errors': self.errors,
                'total_issues': total_issues
            }, f, indent=2)
        
        print(f"\n📊 Detailed report saved to: {report_path}")
        
        return False
    
    def auto_fix_option(self):
        """Offer to automatically fix issues"""
        if not any(self.errors.values()):
            return
            
        print("\n" + "="*60)
        print("🔧 AUTO-FIX OPTION")
        print("="*60)
        
        # Count fixable issues
        fixable = []
        for severity in ['CRITICAL', 'HIGH']:
            for error in self.errors[severity]:
                if error['type'] == 'missing_package':
                    fixable.append(error)
        
        if fixable:
            print(f"\n{len(fixable)} issues can be automatically fixed:")
            for error in fixable:
                print(f"  • Install {error['package']}")
            
            response = input("\nWould you like to auto-fix these issues? (y/n): ")
            if response.lower() == 'y':
                print("\n🔧 Applying automatic fixes...")
                for error in fixable:
                    print(f"\n📦 Installing {error['package']}...")
                    try:
                        subprocess.check_call(error['fix'].split(), stdout=subprocess.DEVNULL)
                        print(f"  ✅ {error['package']} installed successfully")
                    except subprocess.CalledProcessError:
                        print(f"  ❌ Failed to install {error['package']}")

def main():
    print("🚀 TORI Pre-Flight Validation System")
    print("=" * 60)
    print("Checking system requirements before startup...")
    
    validator = PreFlightValidator()
    
    # Run all checks
    validator.check_python_packages()
    validator.check_ports()
    validator.check_system_resources()
    validator.check_configuration_files()
    
    # Generate report
    if validator.generate_report():
        print("\n✅ System ready to start TORI!")
        sys.exit(0)
    else:
        # Offer auto-fix
        validator.auto_fix_option()
        
        print("\n⚠️  Fix the above issues before starting TORI!")
        print("Run 'python tori_emergency_fix.py' for comprehensive fixes")
        sys.exit(1)

if __name__ == "__main__":
    main()
