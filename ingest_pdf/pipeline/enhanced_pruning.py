"""
enhanced_pruning.py

Advanced entropy pruning with detailed logging, adaptive thresholds,
and diversity-aware reranking.
"""

import logging
import json
import time
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, Counter
import numpy as np
from pathlib import Path
import os

# Get logger
logger = logging.getLogger("tori.ingest_pdf.enhanced_pruning")

# Prune reasons
class PruneReason(Enum):
    ENTROPY_THRESHOLD = "entropy_threshold"
    SIMILARITY_THRESHOLD = "similarity_threshold"
    CATEGORY_LIMIT = "category_limit"
    LOW_QUALITY = "low_quality"
    DUPLICATE = "duplicate"
    DIVERSITY_CONSTRAINT = "diversity_constraint"
    
# Document types
class DocumentType(Enum):
    RESEARCH = "research"
    TECHNICAL = "technical"
    NEWS = "news"
    GENERAL = "general"
    
@dataclass
class PruneDecision:
    """Detailed pruning decision for a concept."""
    concept_name: str
    pruned: bool
    reason: Optional[PruneReason] = None
    category: Optional[str] = None
    score: float = 0.0
    quality_score: float = 0.0
    entropy_score: float = 0.0
    similarity_score: float = 0.0
    threshold_used: float = 0.0
    section: str = "body"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
@dataclass
class PruneStatistics:
    """Aggregated pruning statistics."""
    total_concepts: int = 0
    pruned_count: int = 0
    retained_count: int = 0
    prune_by_reason: Dict[str, int] = field(default_factory=dict)
    prune_by_category: Dict[str, int] = field(default_factory=dict)
    prune_by_section: Dict[str, int] = field(default_factory=dict)
    score_distribution: Dict[str, List[float]] = field(default_factory=dict)
    avg_entropy_pruned: float = 0.0
    avg_entropy_retained: float = 0.0
    processing_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_concepts": self.total_concepts,
            "pruned_count": self.pruned_count,
            "retained_count": self.retained_count,
            "prune_rate": self.pruned_count / max(1, self.total_concepts),
            "prune_by_reason": dict(self.prune_by_reason),
            "prune_by_category": dict(self.prune_by_category),
            "prune_by_section": dict(self.prune_by_section),
            "avg_entropy_pruned": round(self.avg_entropy_pruned, 4),
            "avg_entropy_retained": round(self.avg_entropy_retained, 4),
            "processing_time_ms": round(self.processing_time * 1000, 2)
        }

class AdaptiveEntropyConfig:
    """Adaptive entropy configuration based on context."""
    
    def __init__(self, base_config: Dict[str, Any]):
        self.base_config = base_config
        self.overrides = {}
        
    def get_threshold(self, section: str, doc_type: DocumentType, 
                     concept_density: float) -> float:
        """Get adaptive entropy threshold based on context."""
        base_threshold = self.base_config.get("entropy_threshold", 0.0005)
        
        # Section-based adjustments
        section_multipliers = {
            "title": 0.8,      # More strict for titles
            "abstract": 0.85,  # Slightly strict for abstracts
            "introduction": 0.9,
            "conclusion": 0.95,
            "methodology": 1.0,
            "body": 1.0
        }
        
        # Document type adjustments
        doc_type_multipliers = {
            DocumentType.RESEARCH: 0.9,    # Stricter for research
            DocumentType.TECHNICAL: 0.95,
            DocumentType.NEWS: 1.1,        # More lenient for news
            DocumentType.GENERAL: 1.0
        }
        
        # Density adjustment (high density = stricter pruning)
        density_factor = 1.0
        if concept_density > 50:  # concepts per 1000 words
            density_factor = 0.9
        elif concept_density < 10:
            density_factor = 1.1
            
        # Calculate final threshold
        threshold = base_threshold
        threshold *= section_multipliers.get(section, 1.0)
        threshold *= doc_type_multipliers.get(doc_type, 1.0)
        threshold *= density_factor
        
        return threshold
        
    def get_similarity_threshold(self, category: str) -> float:
        """Get similarity threshold by category."""
        base_threshold = self.base_config.get("similarity_threshold", 0.83)
        
        # Category-specific thresholds
        category_thresholds = {
            "technical_terms": 0.85,  # Stricter for technical
            "general_concepts": 0.80,
            "named_entities": 0.90,   # Very strict for names
            "abbreviations": 0.95     # Extremely strict for abbreviations
        }
        
        return category_thresholds.get(category, base_threshold)

class EnhancedPruner:
    """Enhanced pruning with detailed logging and adaptive thresholds."""
    
    def __init__(self, config: Dict[str, Any], doc_type: DocumentType = DocumentType.GENERAL):
        self.config = AdaptiveEntropyConfig(config)
        self.doc_type = doc_type
        self.decisions: List[PruneDecision] = []
        self.stats = PruneStatistics()
        self.enable_logging = config.get("enable_prune_logging", True)
        self.enable_mmr = config.get("enable_mmr_reranking", True)
        self.mmr_lambda = config.get("mmr_lambda", 0.7)  # Balance relevance vs diversity
        
    def prune_concepts(self, concepts: List[Dict[str, Any]], 
                      max_concepts: Optional[int] = None,
                      admin_mode: bool = False) -> Tuple[List[Dict[str, Any]], PruneStatistics]:
        """Prune concepts with detailed logging and adaptive thresholds."""
        start_time = time.time()
        
        if admin_mode and max_concepts is None:
            # No pruning in admin mode without limit
            self.stats.total_concepts = len(concepts)
            self.stats.retained_count = len(concepts)
            return concepts, self.stats
            
        # Calculate concept density
        total_text_length = sum(len(c.get('text', '')) for c in concepts)
        concept_density = len(concepts) / max(1, total_text_length / 1000)
        
        # Phase 1: Initial pruning with detailed decisions
        pruned_concepts = self._initial_prune(concepts, concept_density)
        
        # Phase 2: Apply max limit if needed
        if max_concepts and len(pruned_concepts) > max_concepts:
            pruned_concepts = self._apply_limit(pruned_concepts, max_concepts)
            
        # Phase 3: MMR-based reranking if enabled
        if self.enable_mmr and len(pruned_concepts) > 1:
            pruned_concepts = self._mmr_rerank(pruned_concepts)
            
        # Calculate statistics
        self._calculate_statistics(concepts, pruned_concepts)
        self.stats.processing_time = time.time() - start_time
        
        # Log detailed statistics if enabled
        if self.enable_logging:
            self._log_statistics()
            
        # Save decisions for analysis
        if os.environ.get('SAVE_PRUNE_DECISIONS', '').lower() == 'true':
            self._save_decisions()
            
        return pruned_concepts, self.stats
        
    def _initial_prune(self, concepts: List[Dict[str, Any]], 
                      concept_density: float) -> List[Dict[str, Any]]:
        """Initial pruning phase with adaptive thresholds."""
        retained = []
        
        # Group by category for category-aware pruning
        by_category = defaultdict(list)
        for concept in concepts:
            category = concept.get('metadata', {}).get('category', 'general')
            by_category[category].append(concept)
            
        # Process each category
        for category, cat_concepts in by_category.items():
            # Sort by quality score
            cat_concepts.sort(key=lambda x: x.get('quality_score', x.get('score', 0)), reverse=True)
            
            # Get category limit
            category_limit = self.config.base_config.get('category_limits', {}).get(category)
            
            for i, concept in enumerate(cat_concepts):
                decision = self._evaluate_concept(concept, i, category, category_limit, concept_density)
                self.decisions.append(decision)
                
                if not decision.pruned:
                    retained.append(concept)
                    
        return retained
        
    def _evaluate_concept(self, concept: Dict[str, Any], index: int, category: str,
                         category_limit: Optional[int], concept_density: float) -> PruneDecision:
        """Evaluate a single concept for pruning."""
        name = concept.get('name', '')
        score = concept.get('score', 0)
        quality_score = concept.get('quality_score', 0)
        metadata = concept.get('metadata', {})
        section = metadata.get('section', 'body')
        
        # Calculate entropy score (simplified - in real implementation would use embeddings)
        entropy_score = self._calculate_entropy(concept)
        
        # Get adaptive thresholds
        entropy_threshold = self.config.get_threshold(section, self.doc_type, concept_density)
        similarity_threshold = self.config.get_similarity_threshold(category)
        
        # Initialize decision
        decision = PruneDecision(
            concept_name=name,
            pruned=False,
            category=category,
            score=score,
            quality_score=quality_score,
            entropy_score=entropy_score,
            section=section,
            metadata=metadata
        )
        
        # Check various pruning conditions
        
        # 1. Entropy threshold
        if entropy_score < entropy_threshold:
            decision.pruned = True
            decision.reason = PruneReason.ENTROPY_THRESHOLD
            decision.threshold_used = entropy_threshold
            
        # 2. Low quality score
        elif quality_score < 0.3:
            decision.pruned = True
            decision.reason = PruneReason.LOW_QUALITY
            decision.threshold_used = 0.3
            
        # 3. Category limit
        elif category_limit and index >= category_limit:
            decision.pruned = True
            decision.reason = PruneReason.CATEGORY_LIMIT
            decision.threshold_used = category_limit
            
        # 4. Similarity check (simplified - would use embeddings)
        elif self._is_too_similar(concept, similarity_threshold):
            decision.pruned = True
            decision.reason = PruneReason.SIMILARITY_THRESHOLD
            decision.threshold_used = similarity_threshold
            decision.similarity_score = 0.9  # Placeholder
            
        return decision
        
    def _calculate_entropy(self, concept: Dict[str, Any]) -> float:
        """Calculate entropy score for a concept."""
        # Simplified entropy calculation
        # In real implementation, would use embeddings and information theory
        score = concept.get('score', 0.5)
        quality = concept.get('quality_score', 0.5)
        frequency = concept.get('metadata', {}).get('frequency', 1)
        
        # Simple entropy proxy
        entropy = score * quality * (1 / (1 + np.log(frequency + 1)))
        return entropy
        
    def _is_too_similar(self, concept: Dict[str, Any], threshold: float) -> bool:
        """Check if concept is too similar to already retained concepts."""
        # Simplified similarity check
        # In real implementation, would compare embeddings
        return False  # Placeholder
        
    def _apply_limit(self, concepts: List[Dict[str, Any]], max_concepts: int) -> List[Dict[str, Any]]:
        """Apply maximum concept limit with diversity consideration."""
        if len(concepts) <= max_concepts:
            return concepts
            
        # Mark additional concepts as pruned
        for concept in concepts[max_concepts:]:
            name = concept.get('name', '')
            decision = next((d for d in self.decisions if d.concept_name == name), None)
            if decision:
                decision.pruned = True
                decision.reason = PruneReason.DIVERSITY_CONSTRAINT
                
        return concepts[:max_concepts]
        
    def _mmr_rerank(self, concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank concepts using Maximal Marginal Relevance."""
        if len(concepts) <= 1:
            return concepts
            
        # Initialize with highest scoring concept
        selected = [concepts[0]]
        candidates = concepts[1:]
        
        while candidates and len(selected) < len(concepts):
            # Calculate MMR scores
            mmr_scores = []
            for candidate in candidates:
                relevance = candidate.get('quality_score', candidate.get('score', 0))
                
                # Calculate max similarity to selected concepts
                max_similarity = 0
                for selected_concept in selected:
                    # Simplified similarity (would use embeddings)
                    similarity = self._calculate_similarity(candidate, selected_concept)
                    max_similarity = max(max_similarity, similarity)
                    
                # MMR score
                mmr = self.mmr_lambda * relevance - (1 - self.mmr_lambda) * max_similarity
                mmr_scores.append((candidate, mmr))
                
            # Select concept with highest MMR score
            mmr_scores.sort(key=lambda x: x[1], reverse=True)
            best_candidate = mmr_scores[0][0]
            
            selected.append(best_candidate)
            candidates.remove(best_candidate)
            
        return selected
        
    def _calculate_similarity(self, concept1: Dict[str, Any], concept2: Dict[str, Any]) -> float:
        """Calculate similarity between two concepts."""
        # Simplified similarity calculation
        # In real implementation, would use embeddings
        name1 = concept1.get('name', '').lower()
        name2 = concept2.get('name', '').lower()
        
        # Simple overlap ratio
        if name1 == name2:
            return 1.0
        elif name1 in name2 or name2 in name1:
            return 0.8
        else:
            return 0.0
            
    def _calculate_statistics(self, original: List[Dict[str, Any]], 
                            retained: List[Dict[str, Any]]):
        """Calculate detailed pruning statistics."""
        self.stats.total_concepts = len(original)
        self.stats.retained_count = len(retained)
        self.stats.pruned_count = len(original) - len(retained)
        
        # Count by reason
        reason_counts = Counter()
        category_counts = Counter()
        section_counts = Counter()
        
        pruned_entropies = []
        retained_entropies = []
        
        for decision in self.decisions:
            if decision.pruned:
                reason_counts[decision.reason.value if decision.reason else "unknown"] += 1
                category_counts[decision.category or "unknown"] += 1
                section_counts[decision.section] += 1
                pruned_entropies.append(decision.entropy_score)
            else:
                retained_entropies.append(decision.entropy_score)
                
        self.stats.prune_by_reason = dict(reason_counts)
        self.stats.prune_by_category = dict(category_counts)
        self.stats.prune_by_section = dict(section_counts)
        
        # Calculate average entropies
        if pruned_entropies:
            self.stats.avg_entropy_pruned = np.mean(pruned_entropies)
        if retained_entropies:
            self.stats.avg_entropy_retained = np.mean(retained_entropies)
            
    def _log_statistics(self):
        """Log detailed pruning statistics."""
        stats_dict = self.stats.to_dict()
        logger.info(f"Pruning statistics: {json.dumps(stats_dict, indent=2)}")
        
        # Log top pruned concepts by reason
        if self.decisions:
            logger.debug("Sample pruning decisions:")
            for decision in self.decisions[:10]:
                if decision.pruned:
                    logger.debug(f"  Pruned '{decision.concept_name}': {decision.reason.value}, "
                               f"score={decision.score:.3f}, threshold={decision.threshold_used:.3f}")
                               
    def _save_decisions(self):
        """Save pruning decisions for analysis."""
        output_dir = Path(os.environ.get('PRUNE_LOG_DIR', './prune_logs'))
        output_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"prune_decisions_{timestamp}.json"
        
        decisions_data = [
            {
                "name": d.concept_name,
                "pruned": d.pruned,
                "reason": d.reason.value if d.reason else None,
                "category": d.category,
                "score": round(d.score, 4),
                "quality_score": round(d.quality_score, 4),
                "entropy_score": round(d.entropy_score, 4),
                "threshold": round(d.threshold_used, 4),
                "section": d.section
            }
            for d in self.decisions
        ]
        
        with open(output_file, 'w') as f:
            json.dump({
                "timestamp": timestamp,
                "doc_type": self.doc_type.value,
                "statistics": self.stats.to_dict(),
                "decisions": decisions_data
            }, f, indent=2)
            
        logger.info(f"Saved pruning decisions to {output_file}")


# Factory function
def create_enhanced_pruner(config: Dict[str, Any], 
                          doc_type: Optional[str] = None) -> EnhancedPruner:
    """Create an enhanced pruner with appropriate configuration."""
    
    # Determine document type
    if doc_type:
        try:
            doc_type_enum = DocumentType(doc_type)
        except ValueError:
            doc_type_enum = DocumentType.GENERAL
    else:
        doc_type_enum = DocumentType.GENERAL
        
    return EnhancedPruner(config, doc_type_enum)


# Visualization helper
def create_prune_visualization(decisions: List[PruneDecision], output_path: str):
    """Create visualization of pruning decisions (requires matplotlib)."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Prepare data
        reasons = [d.reason.value for d in decisions if d.pruned and d.reason]
        categories = [d.category for d in decisions if d.pruned and d.category]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Prune reasons
        reason_counts = Counter(reasons)
        axes[0, 0].bar(reason_counts.keys(), reason_counts.values())
        axes[0, 0].set_title("Pruning Reasons")
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Categories pruned
        cat_counts = Counter(categories)
        axes[0, 1].bar(cat_counts.keys(), cat_counts.values())
        axes[0, 1].set_title("Pruned by Category")
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Score distributions
        pruned_scores = [d.score for d in decisions if d.pruned]
        retained_scores = [d.score for d in decisions if not d.pruned]
        
        axes[1, 0].hist([pruned_scores, retained_scores], label=['Pruned', 'Retained'], bins=20)
        axes[1, 0].set_title("Score Distributions")
        axes[1, 0].legend()
        
        # 4. Entropy comparison
        pruned_entropy = [d.entropy_score for d in decisions if d.pruned]
        retained_entropy = [d.entropy_score for d in decisions if not d.pruned]
        
        axes[1, 1].boxplot([pruned_entropy, retained_entropy], labels=['Pruned', 'Retained'])
        axes[1, 1].set_title("Entropy Score Comparison")
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Created pruning visualization at {output_path}")
        
    except ImportError:
        logger.warning("Matplotlib not available for visualization")
