#!/usr/bin/env python3
"""
Master Vault Management Script
One command to rule them all - maintain, audit, and optimize your memory vault
"""

import subprocess
import json
import logging
from pathlib import Path
from datetime import datetime
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("vault_master")


class VaultMaster:
    """Master controller for all vault operations"""
    
    def __init__(self):
        self.vault_path = Path("data/memory_vault/memories")
        self.scripts = {
            'audit': 'audit_memory_vault.py',
            'enrich': 'enrich_concepts.py',
            'classify': 'classify_general_concepts.py',
            'dedupe': 'soft_dedupe.py',
            'merge': 'dedupe_merge.py',
            'dashboard': 'vault_dashboard_safe.py',
            'quality': 'verify_extraction_quality_fixed.py'
        }
    
    def run_script(self, script_name: str, args: list = None) -> bool:
        """Run a script and return success status"""
        if args is None:
            args = []
        
        cmd = ["poetry", "run", "python", script_name] + args
        logger.info(f"🚀 Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"✅ {script_name} completed successfully")
                return True
            else:
                logger.error(f"❌ {script_name} failed: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to run {script_name}: {e}")
            return False
    
    def full_maintenance(self, apply_changes: bool = False):
        """Run complete vault maintenance workflow"""
        logger.info("\n" + "="*70)
        logger.info("🔧 FULL VAULT MAINTENANCE")
        logger.info("="*70)
        
        workflow = [
            # Step 1: Audit current state
            ("Auditing vault integrity", self.scripts['audit'], ["--fix"] if apply_changes else []),
            
            # Step 2: Enrich concepts
            ("Enriching concepts with types", self.scripts['enrich'], ["--save"] if apply_changes else ["--preview"]),
            
            # Step 3: Classify general concepts
            ("Classifying general concepts", self.scripts['classify'], ["--apply"] if apply_changes else []),
            
            # Step 4: Find duplicates
            ("Finding soft duplicates", self.scripts['dedupe'], []),
            
            # Step 5: Show dashboard
            ("Generating dashboard", self.scripts['dashboard'], [])
        ]
        
        results = {}
        for step_name, script, args in workflow:
            logger.info(f"\n📍 {step_name}...")
            results[step_name] = self.run_script(script, args)
        
        # Summary
        logger.info("\n" + "="*70)
        logger.info("📊 MAINTENANCE SUMMARY")
        logger.info("="*70)
        
        success_count = sum(1 for r in results.values() if r)
        logger.info(f"Steps completed: {success_count}/{len(results)}")
        
        for step, success in results.items():
            status = "✅" if success else "❌"
            logger.info(f"  {status} {step}")
        
        # Recommendations
        if Path("soft_duplicates.json").exists():
            with open("soft_duplicates.json", 'r') as f:
                dup_data = json.load(f)
            
            if dup_data['stats']['duplicate_pairs'] > 0:
                logger.info(f"\n💡 Found {dup_data['stats']['duplicate_pairs']} duplicate pairs")
                logger.info("   Run: python vault_master.py merge")
        
        return all(results.values())
    
    def quick_status(self):
        """Quick vault status check"""
        logger.info("\n🔍 Quick Vault Status")
        
        # Count files
        memory_files = list(self.vault_path.glob("*.json"))
        logger.info(f"  Files: {len(memory_files)}")
        
        # Check recent audits
        audit_dir = Path("data/memory_vault/audits")
        if audit_dir.exists():
            audits = list(audit_dir.glob("audit_*.json"))
            if audits:
                latest = max(audits, key=lambda x: x.stat().st_mtime)
                logger.info(f"  Latest audit: {latest.name}")
        
        # Check for reports
        reports = []
        if Path("soft_duplicates.json").exists():
            reports.append("soft_duplicates.json")
        if Path("concept_graph.json").exists():
            reports.append("concept_graph.json")
        if Path("extraction_summary.json").exists():
            reports.append("extraction_summary.json")
        
        if reports:
            logger.info(f"  Available reports: {', '.join(reports)}")
    
    def merge_duplicates(self, dry_run: bool = True):
        """Merge duplicate concepts"""
        if not Path("soft_duplicates.json").exists():
            logger.error("❌ No duplicate analysis found. Run 'dedupe' first.")
            return
        
        args = [] if dry_run else ["--confirm"]
        self.run_script(self.scripts['merge'], args)
    
    def generate_report(self, output_path: str = "vault_report.html"):
        """Generate comprehensive HTML report"""
        logger.info(f"📊 Generating comprehensive report...")
        
        # Gather all data
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'vault_path': str(self.vault_path),
            'file_count': len(list(self.vault_path.glob("*.json")))
        }
        
        # Run dashboard and capture stats
        self.run_script(self.scripts['dashboard'], [])
        
        # Create HTML report
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Memory Vault Report - {date}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        .header {{ background: #2196F3; color: white; padding: 20px; border-radius: 8px; }}
        .section {{ background: #f5f5f5; padding: 15px; margin: 15px 0; border-radius: 8px; }}
        .metric {{ display: inline-block; background: white; padding: 10px; margin: 5px; border-radius: 4px; }}
        pre {{ background: #333; color: #0f0; padding: 10px; border-radius: 4px; overflow-x: auto; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Memory Vault Report</h1>
        <p>Generated: {timestamp}</p>
    </div>
    
    <div class="section">
        <h2>Vault Overview</h2>
        <div class="metric">Files: {file_count}</div>
        <div class="metric">Path: {vault_path}</div>
    </div>
    
    <div class="section">
        <h2>Recent Activity</h2>
        <p>Check the dashboard output above for detailed statistics.</p>
    </div>
    
    <div class="section">
        <h2>Available Tools</h2>
        <ul>
            <li><code>python vault_master.py maintain</code> - Run full maintenance</li>
            <li><code>python vault_master.py audit</code> - Check vault integrity</li>
            <li><code>python vault_master.py enrich</code> - Add concept types</li>
            <li><code>python vault_master.py dedupe</code> - Find duplicates</li>
            <li><code>python vault_master.py merge</code> - Merge duplicates</li>
        </ul>
    </div>
</body>
</html>
        """
        
        html_content = html_template.format(
            date=datetime.now().strftime("%Y-%m-%d"),
            timestamp=report_data['timestamp'],
            file_count=report_data['file_count'],
            vault_path=report_data['vault_path']
        )
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"✅ Report saved to: {output_path}")


def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Master Memory Vault Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python vault_master.py status              # Quick status check
  python vault_master.py maintain            # Full maintenance (preview)
  python vault_master.py maintain --apply    # Full maintenance (apply changes)
  python vault_master.py audit --fix         # Audit and fix issues
  python vault_master.py dedupe              # Find duplicates
  python vault_master.py merge --confirm     # Merge duplicates
  python vault_master.py report              # Generate HTML report
        """
    )
    
    parser.add_argument('command', choices=[
        'status', 'maintain', 'audit', 'enrich', 'classify', 
        'dedupe', 'merge', 'dashboard', 'report'
    ], help='Command to run')
    
    parser.add_argument('--apply', action='store_true', 
                       help='Apply changes (for maintain)')
    parser.add_argument('--confirm', action='store_true', 
                       help='Confirm changes (for merge)')
    parser.add_argument('--fix', action='store_true', 
                       help='Fix issues (for audit)')
    parser.add_argument('--save', action='store_true', 
                       help='Save changes (for enrich)')
    
    args = parser.parse_args()
    
    vault_master = VaultMaster()
    
    if args.command == 'status':
        vault_master.quick_status()
    
    elif args.command == 'maintain':
        vault_master.full_maintenance(apply_changes=args.apply)
    
    elif args.command == 'audit':
        script_args = ['--fix'] if args.fix else []
        vault_master.run_script(vault_master.scripts['audit'], script_args)
    
    elif args.command == 'enrich':
        script_args = ['--save'] if args.save else ['--preview']
        vault_master.run_script(vault_master.scripts['enrich'], script_args)
    
    elif args.command == 'classify':
        script_args = ['--apply'] if args.apply else []
        vault_master.run_script(vault_master.scripts['classify'], script_args)
    
    elif args.command == 'dedupe':
        vault_master.run_script(vault_master.scripts['dedupe'], [])
    
    elif args.command == 'merge':
        vault_master.merge_duplicates(dry_run=not args.confirm)
    
    elif args.command == 'dashboard':
        vault_master.run_script(vault_master.scripts['dashboard'], [])
    
    elif args.command == 'report':
        vault_master.generate_report()


if __name__ == "__main__":
    main()
