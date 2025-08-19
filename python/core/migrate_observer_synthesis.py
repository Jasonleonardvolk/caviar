#!/usr/bin/env python3
"""
Migration script for Observer Synthesis enhancement.
Safely migrates from original to enhanced version with validation and rollback.
"""

import sys
import shutil
import json
import logging
from pathlib import Path
from datetime import datetime
import hashlib
import subprocess
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('observer_synthesis_migration.log')
    ]
)
logger = logging.getLogger(__name__)

class ObserverSynthesisMigration:
    """Handles migration to enhanced Observer Synthesis."""
    
    def __init__(self):
        self.core_dir = Path(__file__).parent
        self.original_file = self.core_dir / "observer_synthesis.py"
        self.enhanced_file = self.core_dir / "observer_synthesis_enhanced.py"
        self.backup_dir = self.core_dir / "backups"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def create_backup(self) -> Path:
        """Create timestamped backup of original file."""
        logger.info("Creating backup of original observer_synthesis.py")
        
        # Ensure backup directory exists
        self.backup_dir.mkdir(exist_ok=True)
        
        # Create backup filename with timestamp
        backup_file = self.backup_dir / f"observer_synthesis_{self.timestamp}.py"
        
        # Copy original file
        if self.original_file.exists():
            shutil.copy2(self.original_file, backup_file)
            logger.info(f"Backup created: {backup_file}")
            
            # Calculate and store checksum
            with open(self.original_file, 'rb') as f:
                checksum = hashlib.sha256(f.read()).hexdigest()
            
            checksum_file = backup_file.with_suffix('.sha256')
            checksum_file.write_text(checksum)
            logger.info(f"Checksum stored: {checksum_file}")
            
            return backup_file
        else:
            raise FileNotFoundError(f"Original file not found: {self.original_file}")
    
    def validate_enhanced_file(self) -> bool:
        """Validate the enhanced file syntax and imports."""
        logger.info("Validating enhanced observer_synthesis.py")
        
        if not self.enhanced_file.exists():
            logger.error(f"Enhanced file not found: {self.enhanced_file}")
            return False
        
        # Check Python syntax
        try:
            compile(self.enhanced_file.read_text(), str(self.enhanced_file), 'exec')
            logger.info("✓ Python syntax is valid")
        except SyntaxError as e:
            logger.error(f"✗ Syntax error in enhanced file: {e}")
            return False
        
        # Check critical imports
        try:
            result = subprocess.run(
                [sys.executable, "-c", 
                 f"import sys; sys.path.insert(0, '{self.core_dir}'); "
                 f"from observer_synthesis_enhanced import ObserverObservedSynthesis"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                logger.info("✓ Imports validated successfully")
            else:
                logger.error(f"✗ Import validation failed: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"✗ Import validation error: {e}")
            return False
        
        return True
    
    def check_dependencies(self) -> bool:
        """Check if all required dependencies are available."""
        logger.info("Checking dependencies")
        
        required_modules = [
            'numpy',
            'threading',
            'logging',
            'json',
            'hashlib',
            'pathlib',
            'dataclasses',
            'collections',
            'contextlib',
            'traceback'
        ]
        
        missing = []
        for module in required_modules:
            try:
                __import__(module)
                logger.debug(f"✓ {module} available")
            except ImportError:
                missing.append(module)
                logger.error(f"✗ Missing module: {module}")
        
        if missing:
            logger.error(f"Missing dependencies: {', '.join(missing)}")
            return False
        
        logger.info("✓ All dependencies available")
        return True
    
    def test_basic_functionality(self) -> bool:
        """Test basic functionality of enhanced version."""
        logger.info("Testing basic functionality")
        
        test_script = '''
import sys
sys.path.insert(0, "{core_dir}")
from observer_synthesis_enhanced import ObserverObservedSynthesis, get_observer_synthesis
import numpy as np

# Test instantiation
synthesis = ObserverObservedSynthesis()
print("✓ Instantiation successful")

# Test input validation
try:
    synthesis.measure(None, 'local', 0.5)
except ValueError:
    print("✓ Input validation working")

# Test measurement
eigenvalues = np.array([0.1, 0.2, 0.3])
measurement = synthesis.measure(eigenvalues, 'local', 0.5)
if measurement:
    print("✓ Basic measurement working")
    
# Test metacognitive context
context = synthesis.generate_metacognitive_context()
if 'health' in context:
    print("✓ Context generation working")
    
# Test health status
health = synthesis.get_health_status()
if health['status'] == 'healthy':
    print("✓ Health monitoring working")
'''.format(core_dir=self.core_dir)
        
        try:
            result = subprocess.run(
                [sys.executable, "-c", test_script],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                logger.info("Basic functionality tests passed")
                logger.debug(f"Test output:\n{result.stdout}")
                return True
            else:
                logger.error(f"Tests failed:\n{result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error running tests: {e}")
            return False
    
    def apply_migration(self) -> bool:
        """Apply the migration by replacing the original file."""
        logger.info("Applying migration")
        
        try:
            # Copy enhanced version to original location
            shutil.copy2(self.enhanced_file, self.original_file)
            logger.info(f"✓ Enhanced version copied to {self.original_file}")
            
            # Verify the copy
            with open(self.enhanced_file, 'rb') as f1, open(self.original_file, 'rb') as f2:
                if f1.read() == f2.read():
                    logger.info("✓ File integrity verified")
                    return True
                else:
                    logger.error("✗ File integrity check failed")
                    return False
                    
        except Exception as e:
            logger.error(f"Error applying migration: {e}")
            return False
    
    def rollback(self, backup_file: Path) -> bool:
        """Rollback to the backup version."""
        logger.warning("Rolling back migration")
        
        try:
            # Verify backup checksum
            checksum_file = backup_file.with_suffix('.sha256')
            if checksum_file.exists():
                stored_checksum = checksum_file.read_text().strip()
                with open(backup_file, 'rb') as f:
                    actual_checksum = hashlib.sha256(f.read()).hexdigest()
                
                if stored_checksum != actual_checksum:
                    logger.error("Backup checksum mismatch!")
                    return False
            
            # Restore backup
            shutil.copy2(backup_file, self.original_file)
            logger.info(f"✓ Restored from backup: {backup_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error during rollback: {e}")
            return False
    
    def create_migration_report(self, success: bool, backup_file: Path = None):
        """Create a detailed migration report."""
        report = {
            'timestamp': self.timestamp,
            'success': success,
            'backup_file': str(backup_file) if backup_file else None,
            'enhanced_features': [
                'Comprehensive input validation',
                'Thread-safe with RLock',
                'Timezone-aware datetime handling',
                'Memory-bounded collections',
                'Custom exception types',
                'Enhanced oscillation detection',
                'Performance tracking',
                'Health monitoring',
                'Atomic file operations'
            ],
            'breaking_changes': [
                'RefexBudgetExhausted exception replaces silent failures',
                'Forced measurements now have rate limits',
                'Input validation is stricter',
                'Some internal methods have changed signatures'
            ]
        }
        
        report_file = self.core_dir / f"migration_report_{self.timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Migration report saved: {report_file}")
    
    def run(self, dry_run: bool = False) -> bool:
        """Run the complete migration process."""
        logger.info("=" * 60)
        logger.info("Starting Observer Synthesis Migration")
        logger.info(f"Dry run: {dry_run}")
        logger.info("=" * 60)
        
        backup_file = None
        success = False
        
        try:
            # Step 1: Check dependencies
            if not self.check_dependencies():
                logger.error("Dependency check failed")
                return False
            
            # Step 2: Validate enhanced file
            if not self.validate_enhanced_file():
                logger.error("Enhanced file validation failed")
                return False
            
            # Step 3: Test functionality
            if not self.test_basic_functionality():
                logger.error("Functionality tests failed")
                return False
            
            if dry_run:
                logger.info("Dry run completed successfully")
                logger.info("Run without --dry-run to apply migration")
                return True
            
            # Step 4: Create backup
            backup_file = self.create_backup()
            
            # Step 5: Apply migration
            if self.apply_migration():
                logger.info("=" * 60)
                logger.info("✓ Migration completed successfully!")
                logger.info(f"✓ Backup saved at: {backup_file}")
                logger.info("=" * 60)
                success = True
            else:
                # Rollback on failure
                if backup_file and backup_file.exists():
                    if self.rollback(backup_file):
                        logger.info("✓ Successfully rolled back to original")
                    else:
                        logger.critical("✗ Rollback failed! Manual intervention required")
                success = False
        
        except Exception as e:
            logger.error(f"Unexpected error during migration: {e}")
            logger.debug(traceback.format_exc())
            
            # Attempt rollback
            if backup_file and backup_file.exists():
                self.rollback(backup_file)
            
            success = False
        
        finally:
            # Create report
            self.create_migration_report(success, backup_file)
        
        return success


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Migrate Observer Synthesis to enhanced version"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run validation without applying changes'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    migration = ObserverSynthesisMigration()
    success = migration.run(dry_run=args.dry_run)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
