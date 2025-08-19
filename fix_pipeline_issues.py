#!/usr/bin/env python3
"""
Fix common issues in pipeline files
Automated fixes for lint and quality issues
"""

import re
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class PipelineFixer:
    """Fix common issues in pipeline files"""
    
    def __init__(self):
        self.fixes_applied = 0
        
    def fix_file(self, filepath: Path) -> int:
        """Fix issues in a single file"""
        fixes = 0
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            original_lines = lines.copy()
            
            # Fix trailing whitespace
            for i, line in enumerate(lines):
                if line.rstrip() != line.rstrip('\n'):
                    lines[i] = line.rstrip() + '\n' if line.endswith('\n') else line.rstrip()
                    fixes += 1
            
            # Fix very long lines (split at logical points)
            for i, line in enumerate(lines):
                if len(line) > 120 and not line.strip().startswith('#'):
                    # Try to split at commas or operators
                    if ',' in line and line.count('(') == line.count(')'):
                        # This is a simplistic fix - real implementation would be smarter
                        pass
            
            # Fix duplicate dictionary keys (manual review needed)
            # This requires AST analysis and manual intervention
            
            # Write back if changes made
            if lines != original_lines:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                logger.info(f"✅ Fixed {fixes} issues in {filepath.name}")
            
        except Exception as e:
            logger.error(f"❌ Error fixing {filepath}: {e}")
        
        return fixes
    
    def add_type_hints(self, filepath: Path):
        """Add basic type hints to functions"""
        # This would require careful AST manipulation
        pass
    
    def format_with_black(self, filepath: Path):
        """Format file with black if available"""
        try:
            import subprocess
            result = subprocess.run(
                ['black', '--line-length', '120', str(filepath)],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                logger.info(f"✅ Formatted {filepath.name} with black")
                return True
        except:
            pass
        return False


def main():
    """Run fixes on critical files"""
    fixer = PipelineFixer()
    
    critical_files = [
        "ingest_pdf/pipeline/quality.py",
        "ingest_pdf/extraction/concept_extraction.py",
        "ingest_pdf/pipeline/pipeline.py",
        "enrich_concepts.py",
        "python/core/vault_writer.py"
    ]
    
    logger.info("🔧 Fixing common issues in pipeline files...")
    logger.info("="*70)
    
    total_fixes = 0
    
    for file_pattern in critical_files:
        filepath = Path(file_pattern)
        if filepath.exists():
            logger.info(f"\n📄 Processing: {filepath}")
            fixes = fixer.fix_file(filepath)
            total_fixes += fixes
            
            # Try black formatting
            fixer.format_with_black(filepath)
    
    logger.info(f"\n✅ Total fixes applied: {total_fixes}")
    
    # Recommendations
    logger.info("\n💡 Recommendations:")
    logger.info("1. Install black for consistent formatting:")
    logger.info("   poetry add --dev black")
    logger.info("2. Install flake8 for linting:")
    logger.info("   poetry add --dev flake8")
    logger.info("3. Add pre-commit hooks to catch issues early")
    logger.info("4. Review any remaining duplicate key issues manually")


if __name__ == "__main__":
    main()
