#!/usr/bin/env python3
"""
Integration helper for vault_inspector.py with enhanced_launcher.py
Provides easy-to-use functions for self-test and monitoring
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

class VaultHealthChecker:
    """Helper class for vault health checks in enhanced_launcher.py"""
    
    def __init__(self, vault_path: str = "data/memory_vault"):
        self.vault_path = vault_path
        self.tools_dir = Path(__file__).parent
        self.inspector_script = self.tools_dir / "vault_inspector.py"
    
    def run_health_check(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Run comprehensive vault health check
        Returns: (is_healthy, details)
        """
        results = {
            "summary": {},
            "consistency": {},
            "fingerprint": {},
            "healthy": True,
            "issues": []
        }
        
        # Run summary check
        try:
            summary_result = subprocess.run(
                [sys.executable, str(self.inspector_script), 
                 "--vault-path", self.vault_path, "--summary", "--json"],
                capture_output=True,
                text=True
            )
            
            if summary_result.returncode == 0:
                results["summary"] = json.loads(summary_result.stdout)
                
                # Check for issues
                if results["summary"].get("corrupt_lines", 0) > 0:
                    results["healthy"] = False
                    results["issues"].append(f"Found {results['summary']['corrupt_lines']} corrupt lines")
            else:
                results["healthy"] = False
                results["issues"].append("Summary check failed")
        except Exception as e:
            results["healthy"] = False
            results["issues"].append(f"Summary check error: {e}")
        
        # Run consistency check
        try:
            consistency_result = subprocess.run(
                [sys.executable, str(self.inspector_script), 
                 "--vault-path", self.vault_path, "--check-consistency", "--json"],
                capture_output=True,
                text=True
            )
            
            if consistency_result.returncode == 0:
                consistency_data = json.loads(consistency_result.stdout)
                results["consistency"] = consistency_data
                
                # Check for issues
                for issue_type, issue_list in consistency_data.items():
                    if issue_list:
                        results["healthy"] = False
                        results["issues"].append(f"{issue_type}: {len(issue_list)} issues")
            else:
                results["healthy"] = False
                results["issues"].append("Consistency check failed")
        except Exception as e:
            results["healthy"] = False
            results["issues"].append(f"Consistency check error: {e}")
        
        # Generate fingerprint
        try:
            fingerprint_result = subprocess.run(
                [sys.executable, str(self.inspector_script), 
                 "--vault-path", self.vault_path, "--fingerprint", "--json"],
                capture_output=True,
                text=True
            )
            
            if fingerprint_result.returncode == 0:
                results["fingerprint"] = json.loads(fingerprint_result.stdout)
        except Exception as e:
            # Fingerprint is optional, don't fail health check
            results["issues"].append(f"Fingerprint generation failed: {e}")
        
        return results["healthy"], results
    
    def quick_summary(self) -> Optional[Dict[str, Any]]:
        """Get quick summary stats"""
        try:
            result = subprocess.run(
                [sys.executable, str(self.inspector_script), 
                 "--vault-path", self.vault_path, "--summary", "--json"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                return json.loads(result.stdout)
        except Exception:
            pass
        
        return None
    
    def print_health_report(self):
        """Print formatted health report"""
        is_healthy, details = self.run_health_check()
        
        print("\n🧠 Vault Health Check")
        print("─" * 40)
        
        # Summary stats
        if details["summary"]:
            print(f"✅ Entries: {details['summary'].get('entries', 0)}")
            print(f"✅ Unique hashes: {details['summary'].get('unique_hashes', 0)}")
            print(f"✅ Sessions: {details['summary'].get('session_count', 0)}")
            print(f"✅ Size: {details['summary'].get('size_mb', 0):.2f} MB")
        
        # Health status
        if is_healthy:
            print("\n✅ Vault is healthy!")
        else:
            print("\n❌ Vault has issues:")
            for issue in details["issues"]:
                print(f"  - {issue}")
        
        # Fingerprint
        if details["fingerprint"] and details["fingerprint"].get("combined_sha256"):
            sha = details["fingerprint"]["combined_sha256"]
            print(f"\n🔑 Fingerprint: {sha[:16]}...{sha[-16:]}")
        
        return is_healthy

# Integration for enhanced_launcher.py
def add_vault_health_to_self_test():
    """
    Add this to your enhanced_launcher.py --self-test implementation:
    
    from tools.vault_health_integration import add_vault_health_to_self_test
    
    # In your self_test method:
    if not add_vault_health_to_self_test():
        print("Vault health check failed!")
        return False
    """
    checker = VaultHealthChecker()
    return checker.print_health_report()

# Example usage
if __name__ == "__main__":
    # Test the health checker
    checker = VaultHealthChecker()
    is_healthy, details = checker.run_health_check()
    
    print(f"Vault healthy: {is_healthy}")
    if not is_healthy:
        print(f"Issues found: {details['issues']}")
    
    # Print full report
    checker.print_health_report()
