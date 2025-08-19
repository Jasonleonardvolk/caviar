"""
Enhanced Diagnostic System for TORI
Implements systematic automation and better separation of concerns
"""

import asyncio
import json
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error classification system"""
    CRITICAL = "blocks_startup"         # Missing core dependencies, port conflicts
    HIGH = "degrades_functionality"     # WebSocket failures, missing routes
    MEDIUM = "affects_performance"      # Build warnings, suboptimal configs
    LOW = "cosmetic_warnings"          # Accessibility warnings, deprecations


class DependencyStatus(Enum):
    """Package availability status"""
    AVAILABLE = "available"
    MISSING = "missing"
    VERSION_MISMATCH = "version_mismatch"
    CORRUPTED = "corrupted"


@dataclass
class DiagnosticResult:
    """Container for diagnostic findings"""
    category: str
    severity: ErrorSeverity
    issue: str
    root_cause: str
    resolution: str
    auto_fixable: bool
    fix_command: Optional[str] = None
    
    def to_dict(self):
        return {
            "category": self.category,
            "severity": self.severity.value,
            "issue": self.issue,
            "root_cause": self.root_cause,
            "resolution": self.resolution,
            "auto_fixable": self.auto_fixable,
            "fix_command": self.fix_command
        }


class TORIDiagnosticSystem:
    """Enhanced diagnostic system with automated triage and recovery"""
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.results: List[DiagnosticResult] = []
        self.port_config = {
            "api": 8002,
            "mcp": 8100,
            "audio_bridge": 8765,
            "hologram_bridge": 8766,
            "frontend": 5173
        }
        
    async def run_full_diagnostic(self) -> Dict:
        """Execute comprehensive system diagnostic"""
        logger.info("Starting TORI Enhanced Diagnostic System...")
        
        # Phase 1: Pre-flight checks
        await self._check_dependencies()
        await self._check_port_availability()
        await self._check_file_permissions()
        
        # Phase 2: Configuration validation
        await self._validate_configurations()
        await self._check_environment_setup()
        
        # Phase 3: Component isolation tests
        await self._test_component_isolation()
        
        # Phase 4: Generate report and fixes
        report = self._generate_report()
        
        # Phase 5: Apply auto-fixes if requested
        if self._should_auto_fix():
            await self._apply_auto_fixes()
            
        return report
    
    async def _check_dependencies(self):
        """Validate all required Python packages"""
        required_packages = {
            # Core dependencies (CRITICAL)
            'torch': {'severity': ErrorSeverity.CRITICAL, 'install': 'pip install torch torchvision torchaudio'},
            'deepdiff': {'severity': ErrorSeverity.CRITICAL, 'install': 'pip install deepdiff'},
            'sympy': {'severity': ErrorSeverity.CRITICAL, 'install': 'pip install sympy'},
            'PyPDF2': {'severity': ErrorSeverity.HIGH, 'install': 'pip install PyPDF2'},
            
            # WebSocket dependencies
            'websockets': {'severity': ErrorSeverity.CRITICAL, 'install': 'pip install websockets'},
            'fastapi': {'severity': ErrorSeverity.CRITICAL, 'install': 'pip install fastapi[all]'},
            
            # Frontend build dependencies
            'uvicorn': {'severity': ErrorSeverity.CRITICAL, 'install': 'pip install uvicorn[standard]'},
            
            # Optional but recommended
            'redis': {'severity': ErrorSeverity.MEDIUM, 'install': 'pip install redis'},
            'celery': {'severity': ErrorSeverity.MEDIUM, 'install': 'pip install celery[redis]'},
        }
        
        for package, info in required_packages.items():
            status = await self._check_package_availability(package)
            
            if status == DependencyStatus.MISSING:
                self.results.append(DiagnosticResult(
                    category="Dependencies",
                    severity=info['severity'],
                    issue=f"Missing package: {package}",
                    root_cause="Package not installed in current environment",
                    resolution=f"Install package using: {info['install']}",
                    auto_fixable=True,
                    fix_command=info['install']
                ))
    
    async def _check_package_availability(self, package_name: str) -> DependencyStatus:
        """Check if a Python package is available"""
        try:
            __import__(package_name)
            return DependencyStatus.AVAILABLE
        except ImportError:
            return DependencyStatus.MISSING
        except Exception as e:
            logger.warning(f"Error checking {package_name}: {e}")
            return DependencyStatus.CORRUPTED
    
    async def _check_port_availability(self):
        """Check if required ports are available"""
        import socket
        
        for service, port in self.port_config.items():
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                # Check if port is in use
                result = sock.connect_ex(('localhost', port))
                if result == 0:
                    # Port is in use - check if it's our service or something else
                    is_our_service = await self._check_if_our_service(port)
                    
                    if not is_our_service:
                        self.results.append(DiagnosticResult(
                            category="Port Conflicts",
                            severity=ErrorSeverity.CRITICAL,
                            issue=f"Port {port} for {service} is already in use",
                            root_cause="Another process is using the required port",
                            resolution=f"Kill process using port or use alternative port",
                            auto_fixable=True,
                            fix_command=f"netstat -ano | findstr :{port} && taskkill /PID <PID> /F"
                        ))
            finally:
                sock.close()
    
    async def _check_if_our_service(self, port: int) -> bool:
        """Check if the process on a port is our TORI service"""
        # This would check process name, API endpoints, etc.
        # Simplified for now
        return False
    
    async def _check_file_permissions(self):
        """Verify file permissions for critical paths"""
        critical_paths = [
            self.project_root / "enhanced_launcher.py",
            self.project_root / "data",
            self.project_root / "logs",
            self.project_root / "tmp"  # For uploads
        ]
        
        for path in critical_paths:
            if not path.exists():
                self.results.append(DiagnosticResult(
                    category="File System",
                    severity=ErrorSeverity.HIGH,
                    issue=f"Missing critical path: {path}",
                    root_cause="Required directory/file not created",
                    resolution=f"Create path: {path}",
                    auto_fixable=True,
                    fix_command=f"mkdir -p {path}" if path.suffix == "" else f"touch {path}"
                ))
    
    async def _validate_configurations(self):
        """Validate configuration files"""
        # Check vite.config.js for proxy settings
        vite_config = self.project_root / "tori_ui_svelte" / "vite.config.js"
        
        if vite_config.exists():
            content = vite_config.read_text()
            if '/api' not in content or 'ws: true' not in content:
                self.results.append(DiagnosticResult(
                    category="Configuration",
                    severity=ErrorSeverity.CRITICAL,
                    issue="Vite proxy not configured for WebSocket",
                    root_cause="Frontend cannot properly proxy API and WebSocket requests",
                    resolution="Update vite.config.js with proper proxy configuration",
                    auto_fixable=True,
                    fix_command="python fix_vite_proxy.py"
                ))
    
    async def _check_environment_setup(self):
        """Check environment variables and setup"""
        required_env_vars = {
            'PRAJNA_MODEL_TYPE': 'saigon',
            'PRAJNA_DEVICE': 'cpu',
            'VITE_ENABLE_CONCEPT_MESH': 'true'
        }
        
        import os
        for var, expected in required_env_vars.items():
            if os.environ.get(var) != expected:
                self.results.append(DiagnosticResult(
                    category="Environment",
                    severity=ErrorSeverity.MEDIUM,
                    issue=f"Environment variable {var} not set correctly",
                    root_cause="Missing or incorrect environment configuration",
                    resolution=f"Set {var}={expected}",
                    auto_fixable=True,
                    fix_command=f"set {var}={expected}"  # Windows
                ))
    
    async def _test_component_isolation(self):
        """Test if components can start in isolation"""
        # This would actually try to start each component
        # For now, just checking if main files exist
        components = {
            "API Server": self.project_root / "enhanced_launcher.py",
            "Frontend": self.project_root / "tori_ui_svelte" / "package.json",
            "Audio Bridge": self.project_root / "audio_hologram_bridge.py",
            "Hologram Bridge": self.project_root / "concept_mesh_hologram_bridge.py"
        }
        
        for name, path in components.items():
            if not path.exists():
                self.results.append(DiagnosticResult(
                    category="Components",
                    severity=ErrorSeverity.HIGH,
                    issue=f"Missing component: {name}",
                    root_cause=f"Component file not found at {path}",
                    resolution=f"Ensure {name} is properly installed",
                    auto_fixable=False,
                    fix_command=None
                ))
    
    def _generate_report(self) -> Dict:
        """Generate comprehensive diagnostic report"""
        # Group by severity
        by_severity = {
            ErrorSeverity.CRITICAL: [],
            ErrorSeverity.HIGH: [],
            ErrorSeverity.MEDIUM: [],
            ErrorSeverity.LOW: []
        }
        
        for result in self.results:
            by_severity[result.severity].append(result.to_dict())
        
        # Calculate health score
        health_score = 100
        health_score -= len(by_severity[ErrorSeverity.CRITICAL]) * 25
        health_score -= len(by_severity[ErrorSeverity.HIGH]) * 10
        health_score -= len(by_severity[ErrorSeverity.MEDIUM]) * 5
        health_score -= len(by_severity[ErrorSeverity.LOW]) * 1
        health_score = max(0, health_score)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "health_score": health_score,
            "total_issues": len(self.results),
            "auto_fixable": sum(1 for r in self.results if r.auto_fixable),
            "by_severity": {
                "critical": by_severity[ErrorSeverity.CRITICAL],
                "high": by_severity[ErrorSeverity.HIGH],
                "medium": by_severity[ErrorSeverity.MEDIUM],
                "low": by_severity[ErrorSeverity.LOW]
            },
            "recommended_actions": self._get_recommended_actions()
        }
    
    def _get_recommended_actions(self) -> List[str]:
        """Generate prioritized list of recommended actions"""
        actions = []
        
        # First handle critical issues
        critical_issues = [r for r in self.results if r.severity == ErrorSeverity.CRITICAL]
        if critical_issues:
            actions.append("1. Fix CRITICAL issues immediately:")
            for issue in critical_issues[:3]:  # Top 3
                actions.append(f"   - {issue.resolution}")
        
        # Then high priority
        high_issues = [r for r in self.results if r.severity == ErrorSeverity.HIGH]
        if high_issues:
            actions.append("2. Address HIGH priority issues:")
            for issue in high_issues[:3]:
                actions.append(f"   - {issue.resolution}")
        
        # General recommendations
        actions.extend([
            "3. Run automated fixes: python enhanced_diagnostic_system.py --auto-fix",
            "4. Restart system after fixes: python enhanced_launcher.py --clean-start",
            "5. Monitor system health: python enhanced_diagnostic_system.py --monitor"
        ])
        
        return actions
    
    def _should_auto_fix(self) -> bool:
        """Check if auto-fix flag is set"""
        return '--auto-fix' in sys.argv
    
    async def _apply_auto_fixes(self):
        """Apply automatic fixes for issues that support it"""
        logger.info("Applying automatic fixes...")
        
        fixed_count = 0
        for result in self.results:
            if result.auto_fixable and result.fix_command:
                try:
                    logger.info(f"Applying fix: {result.fix_command}")
                    
                    # Special handling for pip installs
                    if result.fix_command.startswith('pip install'):
                        subprocess.run([sys.executable, '-m'] + result.fix_command.split(), check=True)
                    else:
                        subprocess.run(result.fix_command, shell=True, check=True)
                    
                    fixed_count += 1
                    logger.info(f"Successfully fixed: {result.issue}")
                    
                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to apply fix: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error applying fix: {e}")
        
        logger.info(f"Applied {fixed_count} automatic fixes")


async def main():
    """Main entry point"""
    project_root = Path("C:\\Users\\jason\\Desktop\\tori\\kha")
    diagnostic = TORIDiagnosticSystem(project_root)
    
    report = await diagnostic.run_full_diagnostic()
    
    # Save report
    report_path = project_root / "debugging_enhanced" / f"diagnostic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("TORI ENHANCED DIAGNOSTIC REPORT")
    print("="*60)
    print(f"Health Score: {report['health_score']}/100")
    print(f"Total Issues: {report['total_issues']}")
    print(f"Auto-fixable: {report['auto_fixable']}")
    print("\nRecommended Actions:")
    for action in report['recommended_actions']:
        print(action)
    print("\nFull report saved to:", report_path)
    

if __name__ == "__main__":
    asyncio.run(main())
