# Re-export ConceptEnricher from the parent directory
import sys
from pathlib import Path

# Add parent to path
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Import from existing location
from enrich_concepts import ConceptEnricher

__all__ = ['ConceptEnricher']
