# Tools module
# Import ConceptEnricher from its current location to maintain compatibility

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from existing location
from enrich_concepts import ConceptEnricher

__all__ = ['ConceptEnricher']
