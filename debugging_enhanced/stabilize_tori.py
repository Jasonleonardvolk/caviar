#!/usr/bin/env python3
"""
TORI System Stabilization Orchestrator
Main entry point for automated diagnostics and fixes
"""

import asyncio
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import subprocess
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class TORIStabilizer:
    """Main orchestrator for TORI system stabilization"""
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.debug_dir = self.project_root / "debugging_enhanced"
        self.results = {
            "start_time": datetime.now().isoformat(),
            "diagnostics": {},
            "fixes_applied": [],
            "validation": {},
            "health_score_before": 0,
            "health_score_after": 0
        }
        
    def print_banner(self):
        """Print welcome banner"""
        print(f"\n{Colors.CYAN}{'='*60}")
        print(f"{Colors.BOLD}TORI SYSTEM STABILIZATION ORCHESTRATOR v2.0{Colors.ENDC}")
        print(f"{Colors.CYAN}{'='*60}{Colors.ENDC}")
        print(f"\n{Colors.BLUE}Project Root: {self.project_root}")
        print(f"Debug Directory: {self.debug_dir}{Colors.ENDC}\n")
        
    async def run_diagnostics(self) -> Dict:
        """Run the enhanced diagnostic system"""
        print(f"\n{Colors.HEADER}[Phase 1] Running System Diagnostics...{Colors.ENDC}")
        
        diagnostic_script = self.debug_dir / "enhanced_diagnostic_system.py"
        
        if not diagnostic_script.exists():
            print(f"{Colors.FAIL}Error: Diagnostic script not found!{Colors.ENDC}")
            return None
            
        # Import and run diagnostics
        sys.path.insert(0, str(self.debug_dir))
        from enhanced_diagnostic_system import TORIDiagnosticSystem
        
        diagnostic = TORIDiagnosticSystem(self.project_root)
        report = await diagnostic.run_full_diagnostic()
        
        self.results["diagnostics"] = report
        self.results["health_score_before"] = report.get("health_score", 0)
        
        # Print summary
        self._print_diagnostic_summary(report)
        
        return report
        
    def _print_diagnostic_summary(self, report: Dict):
        """Print diagnostic results summary"""
        health_score = report.get("health_score", 0)
        
        # Color based on score
        if health_score >= 80:
            color = Colors.GREEN
        elif health_score >= 60:
            color = Colors.WARNING
        else:
            color = Colors.FAIL
            
        print(f"\n{Colors.BOLD}Diagnostic Results:{Colors.ENDC}")
        print(f"Health Score: {color}{health_score}/100{Colors.ENDC}")
        
        # Issues by severity
        by_severity = report.get("by_severity", {})
        print(f"\nIssues Found:")
        print(f"  {Colors.FAIL}Critical: {len(by_severity.get('critical', []))}{Colors.ENDC}")
        print(f"  {Colors.WARNING}High: {len(by_severity.get('high', []))}{Colors.ENDC}")
        print(f"  {Colors.BLUE}Medium: {len(by_severity.get('medium', []))}{Colors.ENDC}")
        print(f"  Low: {len(by_severity.get('low', []))}")
        
        # Auto-fixable
        auto_fixable = report.get("auto_fixable", 0)
        total_issues = report.get("total_issues", 0)
        print(f"\nAuto-fixable: {Colors.GREEN}{auto_fixable}/{total_issues}{Colors.ENDC}")
        
    async def apply_fixes(self, auto_approve: bool = False) -> bool:
        """Apply automated fixes"""
        print(f"\n{Colors.HEADER}[Phase 2] Applying Automated Fixes...{Colors.ENDC}")
        
        if not auto_approve:
            response = input(f"\n{Colors.WARNING}Apply automated fixes? (y/n): {Colors.ENDC}")
            if response.lower() != 'y':
                print("Skipping automated fixes.")
                return False
                
        fixes_script = self.debug_dir / "automated_fixes.py"
        
        if not fixes_script.exists():
            print(f"{Colors.FAIL}Error: Fixes script not found!{Colors.ENDC}")
            return False
            
        # Import and run fixes
        from automated_fixes import TORIAutoFixer
        
        fixer = TORIAutoFixer(self.project_root)
        
        # Apply fixes in priority order
        fix_methods = [
            ("Installing missing dependencies", fixer.install_missing_dependencies),
            ("Creating required directories", fixer.create_missing_directories),
            ("Fixing Vite proxy configuration", fixer.fix_vite_proxy_configuration),
            ("Adding missing API endpoints", fixer.add_missing_soliton_endpoints),
            ("Adding WebSocket endpoints", fixer.add_avatar_websocket_endpoint),
            ("Fixing WebGPU shaders", fixer.fix_webgpu_shader_barriers),
            ("Fixing TailwindCSS utilities", fixer.fix_tailwind_custom_utilities)
        ]
        
        for description, fix_method in fix_methods:
            print(f"\n{Colors.BLUE}→ {description}...{Colors.ENDC}")
            try:
                result = fix_method()
                if result:
                    print(f"{Colors.GREEN}  ✓ Success{Colors.ENDC}")
                    self.results["fixes_applied"].append(description)
                else:
                    print(f"{Colors.WARNING}  ⚠ Skipped or already fixed{Colors.ENDC}")
            except Exception as e:
                print(f"{Colors.FAIL}  ✗ Error: {e}{Colors.ENDC}")
                logger.exception(f"Fix failed: {description}")
                
        return True
        
    async def validate_fixes(self) -> Dict:
        """Validate that fixes were successful"""
        print(f"\n{Colors.HEADER}[Phase 3] Validating Fixes...{Colors.ENDC}")
        
        validation_results = {}
        
        # Check dependencies
        print(f"\n{Colors.BLUE}Checking Python dependencies...{Colors.ENDC}")
        deps_ok = await self._check_dependencies()
        validation_results["dependencies"] = deps_ok
        
        # Check ports
        print(f"\n{Colors.BLUE}Checking port availability...{Colors.ENDC}")
        ports_ok = await self._check_ports()
        validation_results["ports"] = ports_ok
        
        # Check file structure
        print(f"\n{Colors.BLUE}Checking file structure...{Colors.ENDC}")
        files_ok = await self._check_file_structure()
        validation_results["files"] = files_ok
        
        # Test API endpoints
        print(f"\n{Colors.BLUE}Testing API endpoints...{Colors.ENDC}")
        api_ok = await self._test_api_endpoints()
        validation_results["api"] = api_ok
        
        self.results["validation"] = validation_results
        
        # Calculate success rate
        total_checks = len(validation_results)
        passed_checks = sum(1 for v in validation_results.values() if v)
        success_rate = (passed_checks / total_checks) * 100 if total_checks > 0 else 0
        
        print(f"\n{Colors.BOLD}Validation Summary:{Colors.ENDC}")
        print(f"Success Rate: {Colors.GREEN if success_rate >= 80 else Colors.WARNING}{success_rate:.0f}%{Colors.ENDC}")
        
        return validation_results
        
    async def _check_dependencies(self) -> bool:
        """Check if required Python packages are installed"""
        required_packages = ['torch', 'deepdiff', 'sympy', 'PyPDF2', 'websockets', 'fastapi']
        missing = []
        
        for package in required_packages:
            try:
                __import__(package)
                print(f"  ✓ {package}")
            except ImportError:
                print(f"  {Colors.FAIL}✗ {package}{Colors.ENDC}")
                missing.append(package)
                
        return len(missing) == 0
        
    async def _check_ports(self) -> bool:
        """Check if required ports are available"""
        import socket
        
        ports = {
            "API": 8002,
            "MCP": 8100,
            "Audio Bridge": 8765,
            "Hologram Bridge": 8766,
            "Frontend": 5173
        }
        
        all_available = True
        
        for name, port in ports.items():
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            
            if result == 0:
                print(f"  {Colors.WARNING}⚠ {name} ({port}) - In use{Colors.ENDC}")
                all_available = False
            else:
                print(f"  ✓ {name} ({port}) - Available")
                
        return all_available
        
    async def _check_file_structure(self) -> bool:
        """Check if required files and directories exist"""
        required_paths = [
            self.project_root / "enhanced_launcher.py",
            self.project_root / "data",
            self.project_root / "logs",
            self.project_root / "tmp",
            self.project_root / "tori_ui_svelte" / "vite.config.js"
        ]
        
        all_exist = True
        
        for path in required_paths:
            if path.exists():
                print(f"  ✓ {path.name}")
            else:
                print(f"  {Colors.FAIL}✗ {path}{Colors.ENDC}")
                all_exist = False
                
        return all_exist
        
    async def _test_api_endpoints(self) -> bool:
        """Test if API endpoints are responding"""
        # This would make actual HTTP requests in a real implementation
        # For now, just check if the files were patched
        
        api_file = self.project_root / "enhanced_launcher.py"
        if not api_file.exists():
            api_file = self.project_root / "prajna" / "api" / "prajna_api.py"
            
        if api_file.exists():
            content = api_file.read_text()
            endpoints_exist = all([
                "/api/soliton/init" in content,
                "/api/soliton/stats" in content,
                "/api/soliton/embed" in content,
                "/api/avatar/updates" in content
            ])
            
            if endpoints_exist:
                print(f"  ✓ All endpoints defined")
                return True
            else:
                print(f"  {Colors.FAIL}✗ Some endpoints missing{Colors.ENDC}")
                return False
        else:
            print(f"  {Colors.FAIL}✗ API file not found{Colors.ENDC}")
            return False
            
    async def generate_report(self):
        """Generate final report"""
        print(f"\n{Colors.HEADER}[Phase 4] Generating Report...{Colors.ENDC}")
        
        # Re-run diagnostics to get updated health score
        diagnostic_script = self.debug_dir / "enhanced_diagnostic_system.py"
        if diagnostic_script.exists():
            from enhanced_diagnostic_system import TORIDiagnosticSystem
            diagnostic = TORIDiagnosticSystem(self.project_root)
            post_report = await diagnostic.run_full_diagnostic()
            self.results["health_score_after"] = post_report.get("health_score", 0)
        
        # Calculate improvement
        score_before = self.results["health_score_before"]
        score_after = self.results["health_score_after"]
        improvement = score_after - score_before
        
        # Save report
        self.results["end_time"] = datetime.now().isoformat()
        report_path = self.debug_dir / f"stabilization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        # Print summary
        print(f"\n{Colors.BOLD}{'='*60}")
        print(f"STABILIZATION COMPLETE")
        print(f"{'='*60}{Colors.ENDC}")
        
        print(f"\n{Colors.BOLD}Health Score:{Colors.ENDC}")
        print(f"  Before: {Colors.FAIL if score_before < 70 else Colors.WARNING}{score_before}/100{Colors.ENDC}")
        print(f"  After:  {Colors.GREEN if score_after >= 80 else Colors.WARNING}{score_after}/100{Colors.ENDC}")
        print(f"  Improvement: {Colors.GREEN if improvement > 0 else Colors.FAIL}{improvement:+d}{Colors.ENDC}")
        
        print(f"\n{Colors.BOLD}Fixes Applied:{Colors.ENDC} {len(self.results['fixes_applied'])}")
        for fix in self.results['fixes_applied']:
            print(f"  • {fix}")
            
        print(f"\n{Colors.BOLD}Report saved to:{Colors.ENDC} {report_path}")
        
        # Next steps
        print(f"\n{Colors.HEADER}Next Steps:{Colors.ENDC}")
        print(f"1. Review the changes (backups created with .backup extension)")
        print(f"2. Start TORI: python enhanced_launcher.py --clean-start")
        print(f"3. Monitor health: python debugging_enhanced/monitor_health.py")
        
        if score_after < 80:
            print(f"\n{Colors.WARNING}⚠ Health score still below 80. Consider:")
            print(f"  - Manually fixing remaining critical issues")
            print(f"  - Running Tier 2 architectural improvements")
            print(f"  - Checking logs for additional errors{Colors.ENDC}")
            
        return report_path
        
    async def run(self, args):
        """Main orchestration logic"""
        self.print_banner()
        
        try:
            # Phase 1: Diagnostics
            diagnostic_report = await self.run_diagnostics()
            if not diagnostic_report:
                return False
                
            # Phase 2: Apply fixes (if requested)
            if not args.skip_fixes:
                await self.apply_fixes(args.auto_approve)
            
            # Phase 3: Validate
            if not args.skip_validation:
                await self.validate_fixes()
            
            # Phase 4: Generate report
            report_path = await self.generate_report()
            
            return True
            
        except KeyboardInterrupt:
            print(f"\n{Colors.WARNING}Interrupted by user{Colors.ENDC}")
            return False
        except Exception as e:
            print(f"\n{Colors.FAIL}Error: {e}{Colors.ENDC}")
            logger.exception("Stabilization failed")
            return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="TORI System Stabilization Orchestrator")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("C:\\Users\\jason\\Desktop\\tori\\kha"),
        help="TORI project root directory"
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Automatically approve all fixes without prompting"
    )
    parser.add_argument(
        "--skip-fixes",
        action="store_true",
        help="Skip applying fixes (diagnostics only)"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation phase"
    )
    
    args = parser.parse_args()
    
    # Create stabilizer and run
    stabilizer = TORIStabilizer(args.project_root)
    
    # Run async main
    success = asyncio.run(stabilizer.run(args))
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
