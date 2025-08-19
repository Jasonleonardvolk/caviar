#!/usr/bin/env python3
"""
Concept Type and Synonym Enrichment
Adds semantic types and normalizes synonyms
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Concept type rules
CONCEPT_TYPE_RULES = {
    'hardware': [
        'gpu', 'cpu', 'processor', 'chip', 'circuit', 'hardware', 'device',
        'memory', 'storage', 'cache', 'ram', 'ssd'
    ],
    'software': [
        'algorithm', 'software', 'program', 'code', 'api', 'framework',
        'library', 'system', 'platform', 'application'
    ],
    'theory': [
        'theory', 'concept', 'principle', 'law', 'hypothesis', 'model',
        'paradigm', 'framework', 'approach', 'methodology'
    ],
    'physics': [
        'soliton', 'wave', 'quantum', 'particle', 'field', 'energy',
        'momentum', 'dynamics', 'mechanics', 'thermodynamics'
    ],
    'ai_ml': [
        'ai', 'ml', 'neural', 'network', 'learning', 'training',
        'model', 'transformer', 'attention', 'bert', 'gpt'
    ],
    'business': [
        'market', 'business', 'revenue', 'growth', 'investment',
        'strategy', 'competitive', 'customer', 'pricing'
    ],
    'metric': [
        'performance', 'efficiency', 'speed', 'latency', 'throughput',
        'accuracy', 'precision', 'recall', 'score', 'rate'
    ],
    'time': [
        '2024', '2025', '2026', '2027', '2028', '2029', '2030',
        'phase', 'timeline', 'schedule', 'deadline', 'milestone'
    ]
}

# Synonym mappings
SYNONYM_MAPPINGS = {
    # Soliton variants
    'living soliton': ['living soliton memory', 'living soliton-based memory', 'living soliton system'],
    'soliton memory': ['soliton-based memory', 'soliton memory system', 'solitonic memory'],
    
    # Memory variants
    'memory system': ['memory systems', 'memory architecture', 'memory subsystem'],
    
    # Von Neumann variants
    'von neumann': ['von neumann architecture', 'neumann architecture', 'vna'],
    
    # AI/ML variants
    'artificial intelligence': ['ai', 'a.i.', 'machine intelligence'],
    'machine learning': ['ml', 'm.l.', 'deep learning'],
    
    # Common abbreviations
    'gpu': ['graphics processing unit', 'graphics processor'],
    'cpu': ['central processing unit', 'processor']
}


class ConceptEnricher:
    """Enrich concepts with types and normalize synonyms"""
    
    def __init__(self):
        # Build reverse synonym map
        self.synonym_to_canonical = {}
        for canonical, synonyms in SYNONYM_MAPPINGS.items():
            for syn in synonyms:
                self.synonym_to_canonical[syn.lower()] = canonical
    
    def detect_concept_type(self, concept: Dict) -> Optional[str]:
        """Detect the semantic type of a concept"""
        label = concept.get('label', '').lower()
        
        # Check each type's keywords
        for concept_type, keywords in CONCEPT_TYPE_RULES.items():
            for keyword in keywords:
                if keyword in label:
                    return concept_type
        
        # Check relationships for additional context
        relationships = concept.get('metadata', {}).get('relationships', [])
        for rel in relationships:
            target = rel.get('target', '').lower()
            for concept_type, keywords in CONCEPT_TYPE_RULES.items():
                for keyword in keywords:
                    if keyword in target:
                        return concept_type
        
        return 'general'  # Default type
    
    def normalize_synonym(self, label: str) -> str:
        """Normalize a label to its canonical form"""
        label_lower = label.lower().strip()
        
        # Direct match first (exact match only)
        if label_lower in self.synonym_to_canonical:
            return self.synonym_to_canonical[label_lower]
        
        # For partial matches, use word boundaries to avoid false positives
        import re
        for syn, canonical in self.synonym_to_canonical.items():
            # Use word boundaries to match whole words only
            pattern = r'\b' + re.escape(syn) + r'\b'
            if re.search(pattern, label_lower):
                # Replace only whole word matches
                return re.sub(pattern, canonical, label_lower)
        
        return label  # Return original if no match
    
    def enrich_concept(self, concept: Dict) -> Dict:
        """Enrich a single concept with type and normalized label"""
        enriched = concept.copy()
        
        # Add concept type
        if 'concept_type' not in enriched.get('metadata', {}):
            concept_type = self.detect_concept_type(concept)
            if 'metadata' not in enriched:
                enriched['metadata'] = {}
            enriched['metadata']['concept_type'] = concept_type
        
        # Normalize synonym
        original_label = enriched.get('label', '')
        normalized_label = self.normalize_synonym(original_label)
        
        if normalized_label != original_label:
            enriched['metadata']['original_label'] = original_label
            enriched['metadata']['normalized_label'] = normalized_label
            logger.debug(f"Normalized: '{original_label}' → '{normalized_label}'")
        
        return enriched
    
    def enrich_vault(self, vault_path: Path, save: bool = True) -> Dict:
        """Enrich all concepts in the vault"""
        memory_files = list(vault_path.glob("*.json"))
        logger.info(f"📁 Enriching {len(memory_files)} memory files...")
        
        stats = {
            'total': len(memory_files),
            'enriched': 0,
            'types_added': 0,
            'synonyms_normalized': 0,
            'errors': 0
        }
        
        type_counts = {}
        
        for file in memory_files:
            try:
                # Load memory
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                content = data.get('content', {})
                if not isinstance(content, dict):
                    continue
                
                # Enrich concept
                original_content = content.copy()
                enriched_content = self.enrich_concept(content)
                
                # Track changes
                if enriched_content.get('metadata', {}).get('concept_type'):
                    concept_type = enriched_content['metadata']['concept_type']
                    type_counts[concept_type] = type_counts.get(concept_type, 0) + 1
                    if 'concept_type' not in original_content.get('metadata', {}):
                        stats['types_added'] += 1
                
                if enriched_content.get('metadata', {}).get('normalized_label'):
                    stats['synonyms_normalized'] += 1
                
                # Save if requested
                if save and enriched_content != original_content:
                    data['content'] = enriched_content
                    with open(file, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2)
                    stats['enriched'] += 1
                    
            except Exception as e:
                logger.error(f"Failed to enrich {file.name}: {e}")
                stats['errors'] += 1
        
        # Report results
        logger.info("\n" + "="*60)
        logger.info("🎯 CONCEPT ENRICHMENT COMPLETE")
        logger.info("="*60)
        logger.info(f"Total files: {stats['total']}")
        logger.info(f"Files enriched: {stats['enriched']}")
        logger.info(f"Types added: {stats['types_added']}")
        logger.info(f"Synonyms normalized: {stats['synonyms_normalized']}")
        
        if type_counts:
            logger.info("\n📊 Concept Type Distribution:")
            for ctype, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  {ctype}: {count}")
        
        return stats


def show_enrichment_preview(vault_path: Path, sample_size: int = 10):
    """Preview enrichment without saving"""
    enricher = ConceptEnricher()
    memory_files = list(vault_path.glob("*.json"))[:sample_size]
    
    logger.info(f"\n🔍 Enrichment Preview ({len(memory_files)} samples)")
    logger.info("="*60)
    
    for file in memory_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            content = data.get('content', {})
            if not isinstance(content, dict):
                continue
            
            enriched = enricher.enrich_concept(content)
            
            # Show changes
            label = content.get('label', 'N/A')
            concept_type = enriched.get('metadata', {}).get('concept_type', 'N/A')
            normalized = enriched.get('metadata', {}).get('normalized_label')
            
            logger.info(f"\n📄 {file.name}")
            logger.info(f"  Label: {label}")
            logger.info(f"  Type: {concept_type}")
            if normalized:
                logger.info(f"  Normalized: {normalized}")
                
        except Exception as e:
            logger.error(f"Failed to preview {file.name}: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enrich concepts with types and normalize synonyms")
    parser.add_argument('--preview', action='store_true', help='Preview enrichment without saving')
    parser.add_argument('--save', action='store_true', help='Save enrichment to vault files')
    
    args = parser.parse_args()
    
    vault_path = Path("data/memory_vault/memories")
    
    if not vault_path.exists():
        logger.error("❌ Memory vault not found")
    else:
        if args.preview:
            show_enrichment_preview(vault_path)
        elif args.save:
            enricher = ConceptEnricher()
            enricher.enrich_vault(vault_path, save=True)
        else:
            logger.info("Use --preview to see enrichment or --save to apply it")
