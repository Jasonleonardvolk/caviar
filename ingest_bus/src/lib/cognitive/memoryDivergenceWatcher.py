"""
Memory Divergence Watcher for TORI

This module automatically detects when TORI's interpretation of a concept drifts
from its original source meaning through misattribution, hallucination, or
speculative overgrowth. It provides continuous monitoring and alerting for
concept integrity across the memory systems.

Key Features:
- Periodic scanning of ConceptMesh and ψMesh entries
- Embedding drift detection using vector similarity
- Metadata consistency validation
- User and Ghost feedback correlation
- Automated drift alerts and revalidation triggers
- Comprehensive drift analytics and reporting
"""

import asyncio
import logging
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import uuid

# ML imports for embedding comparison
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Configure logging
logger = logging.getLogger("tori.memory_divergence")

class DriftType(str, Enum):
    """Types of memory drift detected."""
    EMBEDDING_DRIFT = "embedding_drift"
    DEFINITION_CHANGE = "definition_change"
    ATTRIBUTION_ERROR = "attribution_error"
    USAGE_SHIFT = "usage_shift"
    METADATA_INCONSISTENCY = "metadata_inconsistency"
    CONTRADICTORY_EVIDENCE = "contradictory_evidence"
    SPECULATIVE_OVERGROWTH = "speculative_overgrowth"
    SOURCE_DEGRADATION = "source_degradation"

class DriftSeverity(str, Enum):
    """Severity levels for drift detection."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ConceptSnapshot:
    """Snapshot of a concept at a specific point in time."""
    concept_id: str
    timestamp: datetime
    definition: str
    embeddings: np.ndarray
    metadata: Dict[str, Any]
    source_references: List[str]
    usage_examples: List[str]
    ghost_reflections: List[Dict[str, Any]]
    confidence_score: float
    version_hash: str

@dataclass
class DriftAlert:
    """Alert generated when concept drift is detected."""
    alert_id: str
    concept_id: str
    drift_type: DriftType
    severity: DriftSeverity
    drift_score: float
    
    # Comparison data
    original_snapshot: ConceptSnapshot
    current_snapshot: ConceptSnapshot
    drift_details: Dict[str, Any]
    
    # Evidence and context
    evidence: List[str]
    affected_sources: List[str]
    detection_method: str
    
    # Timestamps and tracking
    detected_at: datetime
    first_drift_detected: Optional[datetime] = None
    drift_progression: List[Tuple[datetime, float]] = None
    
    def __post_init__(self):
        if self.drift_progression is None:
            self.drift_progression = []

@dataclass
class DriftAnalysis:
    """Comprehensive analysis of drift patterns."""
    concept_id: str
    analysis_period: Tuple[datetime, datetime]
    total_drift_events: int
    average_drift_score: float
    drift_velocity: float  # Rate of drift over time
    drift_types_detected: List[DriftType]
    most_severe_drift: DriftAlert
    stability_score: float  # 1.0 = perfectly stable, 0.0 = highly unstable
    recommendations: List[str]

class MemoryDivergenceWatcher:
    """
    Monitors concept memory integrity and detects when interpretations
    drift from original source meanings.
    """
    
    def __init__(self):
        """Initialize the memory divergence watcher."""
        self.concept_snapshots = {}  # concept_id -> List[ConceptSnapshot]
        self.drift_alerts = {}  # alert_id -> DriftAlert
        self.concept_baselines = {}  # concept_id -> ConceptSnapshot
        self.active_monitoring = {}  # concept_id -> monitoring_config
        
        # Drift detection thresholds
        self.drift_thresholds = {
            DriftType.EMBEDDING_DRIFT: 0.15,  # Cosine distance threshold
            DriftType.DEFINITION_CHANGE: 0.3,
            DriftType.ATTRIBUTION_ERROR: 0.5,
            DriftType.USAGE_SHIFT: 0.2,
            DriftType.METADATA_INCONSISTENCY: 0.4,
            DriftType.CONTRADICTORY_EVIDENCE: 0.6,
            DriftType.SPECULATIVE_OVERGROWTH: 0.25,
            DriftType.SOURCE_DEGRADATION: 0.7
        }
        
        # Monitoring configuration
        self.scan_interval = 3600  # 1 hour in seconds
        self.max_snapshots_per_concept = 100
        self.embedding_model = None
        self.monitoring_active = False
        
        # Event callbacks
        self.drift_callbacks = []
        self.alert_callbacks = []
        
        logger.info("🔍 Memory Divergence Watcher initialized")
        
        # Initialize embedding model
        asyncio.create_task(self._initialize_embedding_model())
    
    async def _initialize_embedding_model(self):
        """Initialize the embedding model for drift detection."""
        try:
            logger.info("🧠 Loading embedding model for drift detection...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {str(e)}")
            raise
    
    def register_concept_baseline(
        self,
        concept_id: str,
        definition: str,
        metadata: Dict[str, Any],
        source_references: List[str]
    ):
        """
        Register a baseline snapshot for a concept.
        
        Args:
            concept_id: Unique identifier for the concept
            definition: Original definition of the concept
            metadata: Concept metadata
            source_references: Original source references
        """
        try:
            # Generate embeddings for the definition
            embeddings = self.embedding_model.encode([definition])[0] if self.embedding_model else np.array([])
            
            # Create baseline snapshot
            baseline = ConceptSnapshot(
                concept_id=concept_id,
                timestamp=datetime.now(timezone.utc),
                definition=definition,
                embeddings=embeddings,
                metadata=metadata,
                source_references=source_references,
                usage_examples=[],
                ghost_reflections=[],
                confidence_score=1.0,
                version_hash=self._calculate_concept_hash(definition, metadata)
            )
            
            self.concept_baselines[concept_id] = baseline
            
            # Initialize snapshot history
            if concept_id not in self.concept_snapshots:
                self.concept_snapshots[concept_id] = []
            self.concept_snapshots[concept_id].append(baseline)
            
            # Start monitoring
            self.active_monitoring[concept_id] = {
                "enabled": True,
                "last_scan": datetime.now(timezone.utc),
                "scan_count": 0,
                "drift_count": 0
            }
            
            logger.info(f"📊 Registered baseline for concept: {concept_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to register concept baseline: {str(e)}")
            raise
    
    def take_concept_snapshot(
        self,
        concept_id: str,
        current_definition: str,
        current_metadata: Dict[str, Any],
        current_sources: List[str],
        usage_examples: List[str] = None,
        ghost_reflections: List[Dict[str, Any]] = None,
        confidence_score: float = 1.0
    ) -> ConceptSnapshot:
        """
        Take a snapshot of a concept's current state.
        
        Args:
            concept_id: Concept identifier
            current_definition: Current definition
            current_metadata: Current metadata
            current_sources: Current source references
            usage_examples: Examples of concept usage
            ghost_reflections: Recent Ghost reflections
            confidence_score: Current confidence in the concept
            
        Returns:
            ConceptSnapshot of current state
        """
        try:
            # Generate embeddings
            embeddings = self.embedding_model.encode([current_definition])[0] if self.embedding_model else np.array([])
            
            # Create snapshot
            snapshot = ConceptSnapshot(
                concept_id=concept_id,
                timestamp=datetime.now(timezone.utc),
                definition=current_definition,
                embeddings=embeddings,
                metadata=current_metadata,
                source_references=current_sources,
                usage_examples=usage_examples or [],
                ghost_reflections=ghost_reflections or [],
                confidence_score=confidence_score,
                version_hash=self._calculate_concept_hash(current_definition, current_metadata)
            )
            
            # Store snapshot
            if concept_id not in self.concept_snapshots:
                self.concept_snapshots[concept_id] = []
            
            self.concept_snapshots[concept_id].append(snapshot)
            
            # Limit snapshot history
            if len(self.concept_snapshots[concept_id]) > self.max_snapshots_per_concept:
                self.concept_snapshots[concept_id] = self.concept_snapshots[concept_id][-self.max_snapshots_per_concept:]
            
            logger.debug(f"📸 Snapshot taken for concept: {concept_id}")
            return snapshot
            
        except Exception as e:
            logger.error(f"❌ Failed to take concept snapshot: {str(e)}")
            raise
    
    async def detect_drift(self, concept_id: str) -> List[DriftAlert]:
        """
        Detect drift for a specific concept.
        
        Args:
            concept_id: Concept to analyze for drift
            
        Returns:
            List of drift alerts detected
        """
        try:
            if concept_id not in self.concept_baselines:
                logger.warning(f"⚠️  No baseline found for concept: {concept_id}")
                return []
            
            if concept_id not in self.concept_snapshots or len(self.concept_snapshots[concept_id]) < 2:
                logger.debug(f"📊 Insufficient snapshots for drift detection: {concept_id}")
                return []
            
            baseline = self.concept_baselines[concept_id]
            current = self.concept_snapshots[concept_id][-1]
            
            drift_alerts = []
            
            # 1. Embedding drift detection
            if len(baseline.embeddings) > 0 and len(current.embeddings) > 0:
                drift_alert = await self._detect_embedding_drift(baseline, current)
                if drift_alert:
                    drift_alerts.append(drift_alert)
            
            # 2. Definition change detection
            drift_alert = await self._detect_definition_change(baseline, current)
            if drift_alert:
                drift_alerts.append(drift_alert)
            
            # 3. Attribution error detection
            drift_alert = await self._detect_attribution_errors(baseline, current)
            if drift_alert:
                drift_alerts.append(drift_alert)
            
            # 4. Usage shift detection
            drift_alert = await self._detect_usage_shift(baseline, current)
            if drift_alert:
                drift_alerts.append(drift_alert)
            
            # 5. Metadata inconsistency detection
            drift_alert = await self._detect_metadata_inconsistency(baseline, current)
            if drift_alert:
                drift_alerts.append(drift_alert)
            
            # 6. Contradictory evidence detection
            drift_alert = await self._detect_contradictory_evidence(baseline, current)
            if drift_alert:
                drift_alerts.append(drift_alert)
            
            # 7. Speculative overgrowth detection
            drift_alert = await self._detect_speculative_overgrowth(baseline, current)
            if drift_alert:
                drift_alerts.append(drift_alert)
            
            # Store alerts
            for alert in drift_alerts:
                self.drift_alerts[alert.alert_id] = alert
                
                # Update monitoring stats
                if concept_id in self.active_monitoring:
                    self.active_monitoring[concept_id]["drift_count"] += 1
                
                # Trigger callbacks
                for callback in self.drift_callbacks:
                    try:
                        await callback(alert)
                    except Exception as e:
                        logger.error(f"❌ Drift callback failed: {str(e)}")
            
            if drift_alerts:
                logger.warning(f"🚨 Detected {len(drift_alerts)} drift alert(s) for concept: {concept_id}")
            
            return drift_alerts
            
        except Exception as e:
            logger.error(f"❌ Drift detection failed for concept {concept_id}: {str(e)}")
            return []
    
    async def _detect_embedding_drift(
        self,
        baseline: ConceptSnapshot,
        current: ConceptSnapshot
    ) -> Optional[DriftAlert]:
        """Detect drift using embedding similarity."""
        try:
            if len(baseline.embeddings) == 0 or len(current.embeddings) == 0:
                return None
            
            # Calculate cosine similarity
            similarity = cosine_similarity(
                baseline.embeddings.reshape(1, -1),
                current.embeddings.reshape(1, -1)
            )[0][0]
            
            drift_distance = 1 - similarity
            threshold = self.drift_thresholds[DriftType.EMBEDDING_DRIFT]
            
            if drift_distance > threshold:
                severity = self._calculate_severity(drift_distance, threshold)
                
                drift_alert = DriftAlert(
                    alert_id=str(uuid.uuid4()),
                    concept_id=current.concept_id,
                    drift_type=DriftType.EMBEDDING_DRIFT,
                    severity=severity,
                    drift_score=drift_distance,
                    original_snapshot=baseline,
                    current_snapshot=current,
                    drift_details={
                        "cosine_similarity": similarity,
                        "drift_distance": drift_distance,
                        "threshold": threshold
                    },
                    evidence=[
                        f"Embedding similarity dropped to {similarity:.3f}",
                        f"Drift distance {drift_distance:.3f} exceeds threshold {threshold}"
                    ],
                    affected_sources=current.source_references,
                    detection_method="embedding_cosine_similarity",
                    detected_at=datetime.now(timezone.utc)
                )
                
                return drift_alert
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Embedding drift detection failed: {str(e)}")
            return None
    
    async def _detect_definition_change(
        self,
        baseline: ConceptSnapshot,
        current: ConceptSnapshot
    ) -> Optional[DriftAlert]:
        """Detect significant changes in concept definition."""
        try:
            # Simple text similarity (could be enhanced with NLP)
            baseline_words = set(baseline.definition.lower().split())
            current_words = set(current.definition.lower().split())
            
            # Jaccard similarity
            intersection = len(baseline_words.intersection(current_words))
            union = len(baseline_words.union(current_words))
            jaccard_similarity = intersection / union if union > 0 else 0
            
            definition_drift = 1 - jaccard_similarity
            threshold = self.drift_thresholds[DriftType.DEFINITION_CHANGE]
            
            if definition_drift > threshold:
                severity = self._calculate_severity(definition_drift, threshold)
                
                # Analyze specific changes
                added_words = current_words - baseline_words
                removed_words = baseline_words - current_words
                
                drift_alert = DriftAlert(
                    alert_id=str(uuid.uuid4()),
                    concept_id=current.concept_id,
                    drift_type=DriftType.DEFINITION_CHANGE,
                    severity=severity,
                    drift_score=definition_drift,
                    original_snapshot=baseline,
                    current_snapshot=current,
                    drift_details={
                        "jaccard_similarity": jaccard_similarity,
                        "words_added": list(added_words)[:10],  # Limit for readability
                        "words_removed": list(removed_words)[:10],
                        "definition_length_change": len(current.definition) - len(baseline.definition)
                    },
                    evidence=[
                        f"Definition similarity dropped to {jaccard_similarity:.3f}",
                        f"Added {len(added_words)} new terms, removed {len(removed_words)} terms"
                    ],
                    affected_sources=current.source_references,
                    detection_method="definition_text_analysis",
                    detected_at=datetime.now(timezone.utc)
                )
                
                return drift_alert
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Definition change detection failed: {str(e)}")
            return None
    
    async def _detect_attribution_errors(
        self,
        baseline: ConceptSnapshot,
        current: ConceptSnapshot
    ) -> Optional[DriftAlert]:
        """Detect attribution errors in source references."""
        try:
            baseline_sources = set(baseline.source_references)
            current_sources = set(current.source_references)
            
            # Check for source inconsistencies
            removed_sources = baseline_sources - current_sources
            added_sources = current_sources - baseline_sources
            
            # Calculate attribution drift
            total_sources = len(baseline_sources.union(current_sources))
            changed_sources = len(removed_sources) + len(added_sources)
            attribution_drift = changed_sources / total_sources if total_sources > 0 else 0
            
            threshold = self.drift_thresholds[DriftType.ATTRIBUTION_ERROR]
            
            # Also check for suspicious patterns
            suspicious_patterns = []
            
            # Check if core sources were removed
            if len(removed_sources) > len(baseline_sources) / 2:
                suspicious_patterns.append("Major source removal")
            
            # Check for attribution to questionable sources
            for source in added_sources:
                if any(pattern in source.lower() for pattern in ['unknown', 'generated', 'inferred']):
                    suspicious_patterns.append(f"Questionable source: {source}")
            
            if attribution_drift > threshold or suspicious_patterns:
                severity = self._calculate_severity(attribution_drift, threshold)
                if suspicious_patterns:
                    severity = max(severity, DriftSeverity.MEDIUM)
                
                drift_alert = DriftAlert(
                    alert_id=str(uuid.uuid4()),
                    concept_id=current.concept_id,
                    drift_type=DriftType.ATTRIBUTION_ERROR,
                    severity=severity,
                    drift_score=attribution_drift,
                    original_snapshot=baseline,
                    current_snapshot=current,
                    drift_details={
                        "removed_sources": list(removed_sources),
                        "added_sources": list(added_sources),
                        "suspicious_patterns": suspicious_patterns,
                        "source_retention_rate": len(baseline_sources.intersection(current_sources)) / len(baseline_sources)
                    },
                    evidence=[
                        f"Attribution drift score: {attribution_drift:.3f}",
                        f"Removed {len(removed_sources)} sources, added {len(added_sources)} sources"
                    ] + suspicious_patterns,
                    affected_sources=list(removed_sources.union(added_sources)),
                    detection_method="source_attribution_analysis",
                    detected_at=datetime.now(timezone.utc)
                )
                
                return drift_alert
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Attribution error detection failed: {str(e)}")
            return None
    
    async def _detect_usage_shift(
        self,
        baseline: ConceptSnapshot,
        current: ConceptSnapshot
    ) -> Optional[DriftAlert]:
        """Detect shifts in how the concept is being used."""
        try:
            if not current.usage_examples:
                return None
            
            # Analyze usage patterns (simplified implementation)
            baseline_usage_text = " ".join(baseline.usage_examples)
            current_usage_text = " ".join(current.usage_examples)
            
            if not baseline_usage_text:
                return None
            
            # Generate embeddings for usage patterns
            baseline_usage_embedding = self.embedding_model.encode([baseline_usage_text])[0]
            current_usage_embedding = self.embedding_model.encode([current_usage_text])[0]
            
            # Calculate usage similarity
            usage_similarity = cosine_similarity(
                baseline_usage_embedding.reshape(1, -1),
                current_usage_embedding.reshape(1, -1)
            )[0][0]
            
            usage_drift = 1 - usage_similarity
            threshold = self.drift_thresholds[DriftType.USAGE_SHIFT]
            
            if usage_drift > threshold:
                severity = self._calculate_severity(usage_drift, threshold)
                
                drift_alert = DriftAlert(
                    alert_id=str(uuid.uuid4()),
                    concept_id=current.concept_id,
                    drift_type=DriftType.USAGE_SHIFT,
                    severity=severity,
                    drift_score=usage_drift,
                    original_snapshot=baseline,
                    current_snapshot=current,
                    drift_details={
                        "usage_similarity": usage_similarity,
                        "baseline_examples_count": len(baseline.usage_examples),
                        "current_examples_count": len(current.usage_examples)
                    },
                    evidence=[
                        f"Usage pattern similarity: {usage_similarity:.3f}",
                        f"Usage drift score: {usage_drift:.3f}"
                    ],
                    affected_sources=current.source_references,
                    detection_method="usage_pattern_analysis",
                    detected_at=datetime.now(timezone.utc)
                )
                
                return drift_alert
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Usage shift detection failed: {str(e)}")
            return None
    
    async def _detect_metadata_inconsistency(
        self,
        baseline: ConceptSnapshot,
        current: ConceptSnapshot
    ) -> Optional[DriftAlert]:
        """Detect inconsistencies in concept metadata."""
        try:
            inconsistencies = []
            
            # Check for key metadata changes
            baseline_meta = baseline.metadata
            current_meta = current.metadata
            
            # Type consistency
            baseline_type = baseline_meta.get('type', 'unknown')
            current_type = current_meta.get('type', 'unknown')
            
            if baseline_type != current_type and baseline_type != 'unknown':
                inconsistencies.append(f"Type changed from '{baseline_type}' to '{current_type}'")
            
            # Confidence score drops
            confidence_drop = baseline.confidence_score - current.confidence_score
            if confidence_drop > 0.3:
                inconsistencies.append(f"Confidence dropped by {confidence_drop:.2f}")
            
            # Version hash changes without source changes
            if (baseline.version_hash != current.version_hash and 
                set(baseline.source_references) == set(current.source_references)):
                inconsistencies.append("Content changed without source attribution")
            
            # Calculate inconsistency score
            inconsistency_score = len(inconsistencies) / 5.0  # Normalize to 0-1
            threshold = self.drift_thresholds[DriftType.METADATA_INCONSISTENCY]
            
            if inconsistency_score > threshold:
                severity = self._calculate_severity(inconsistency_score, threshold)
                
                drift_alert = DriftAlert(
                    alert_id=str(uuid.uuid4()),
                    concept_id=current.concept_id,
                    drift_type=DriftType.METADATA_INCONSISTENCY,
                    severity=severity,
                    drift_score=inconsistency_score,
                    original_snapshot=baseline,
                    current_snapshot=current,
                    drift_details={
                        "inconsistencies": inconsistencies,
                        "confidence_change": confidence_drop,
                        "type_change": f"{baseline_type} -> {current_type}"
                    },
                    evidence=inconsistencies,
                    affected_sources=current.source_references,
                    detection_method="metadata_consistency_analysis",
                    detected_at=datetime.now(timezone.utc)
                )
                
                return drift_alert
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Metadata inconsistency detection failed: {str(e)}")
            return None
    
    async def _detect_contradictory_evidence(
        self,
        baseline: ConceptSnapshot,
        current: ConceptSnapshot
    ) -> Optional[DriftAlert]:
        """Detect contradictory evidence in Ghost reflections or usage."""
        try:
            contradictions = []
            
            # Analyze Ghost reflections for contradictions
            for reflection in current.ghost_reflections:
                reflection_text = reflection.get('message', '').lower()
                
                # Look for contradiction keywords
                contradiction_keywords = [
                    'however', 'but', 'contradicts', 'disagrees', 'opposite',
                    'incorrect', 'wrong', 'mistaken', 'retract', 'correction'
                ]
                
                if any(keyword in reflection_text for keyword in contradiction_keywords):
                    contradictions.append(f"Ghost reflection indicates contradiction: {reflection_text[:100]}...")
            
            # Check for conflicting definitions
            if ('not' in current.definition.lower() and 'not' not in baseline.definition.lower()):
                contradictions.append("Definition now contains negation")
            
            contradiction_score = min(len(contradictions) / 3.0, 1.0)  # Normalize
            threshold = self.drift_thresholds[DriftType.CONTRADICTORY_EVIDENCE]
            
            if contradiction_score > threshold:
                severity = self._calculate_severity(contradiction_score, threshold)
                
                drift_alert = DriftAlert(
                    alert_id=str(uuid.uuid4()),
                    concept_id=current.concept_id,
                    drift_type=DriftType.CONTRADICTORY_EVIDENCE,
                    severity=severity,
                    drift_score=contradiction_score,
                    original_snapshot=baseline,
                    current_snapshot=current,
                    drift_details={
                        "contradictions_found": contradictions,
                        "ghost_reflection_count": len(current.ghost_reflections)
                    },
                    evidence=contradictions,
                    affected_sources=current.source_references,
                    detection_method="contradiction_analysis",
                    detected_at=datetime.now(timezone.utc)
                )
                
                return drift_alert
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Contradictory evidence detection failed: {str(e)}")
            return None
    
    async def _detect_speculative_overgrowth(
        self,
        baseline: ConceptSnapshot,
        current: ConceptSnapshot
    ) -> Optional[DriftAlert]:
        """Detect speculative overgrowth where concept has grown beyond source material."""
        try:
            # Calculate definition growth
            growth_ratio = len(current.definition) / len(baseline.definition) if baseline.definition else 1
            
            # Check for speculative language
            speculative_keywords = [
                'might be', 'could be', 'possibly', 'perhaps', 'may indicate',
                'suggests', 'implies', 'potentially', 'likely', 'probably'
            ]
            
            current_text = current.definition.lower()
            speculative_count = sum(1 for keyword in speculative_keywords if keyword in current_text)
            
            # Calculate overgrowth score
            overgrowth_indicators = []
            
            if growth_ratio > 2.0:
                overgrowth_indicators.append(f"Definition grew by {growth_ratio:.1f}x")
            
            if speculative_count > 2:
                overgrowth_indicators.append(f"Contains {speculative_count} speculative terms")
            
            if len(current.usage_examples) > len(baseline.usage_examples) * 3:
                overgrowth_indicators.append("Usage examples multiplied excessively")
            
            overgrowth_score = min(len(overgrowth_indicators) / 3.0, 1.0)
            threshold = self.drift_thresholds[DriftType.SPECULATIVE_OVERGROWTH]
            
            if overgrowth_score > threshold:
                severity = self._calculate_severity(overgrowth_score, threshold)
                
                drift_alert = DriftAlert(
                    alert_id=str(uuid.uuid4()),
                    concept_id=current.concept_id,
                    drift_type=DriftType.SPECULATIVE_OVERGROWTH,
                    severity=severity,
                    drift_score=overgrowth_score,
                    original_snapshot=baseline,
                    current_snapshot=current,
                    drift_details={
                        "growth_ratio": growth_ratio,
                        "speculative_terms": speculative_count,
                        "overgrowth_indicators": overgrowth_indicators
                    },
                    evidence=overgrowth_indicators,
                    affected_sources=current.source_references,
                    detection_method="speculative_overgrowth_analysis",
                    detected_at=datetime.now(timezone.utc)
                )
                
                return drift_alert
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Speculative overgrowth detection failed: {str(e)}")
            return None
    
    def _calculate_severity(self, drift_score: float, threshold: float) -> DriftSeverity:
        """Calculate drift severity based on score and threshold."""
        if drift_score > threshold * 3:
            return DriftSeverity.CRITICAL
        elif drift_score > threshold * 2:
            return DriftSeverity.HIGH
        elif drift_score > threshold * 1.5:
            return DriftSeverity.MEDIUM
        else:
            return DriftSeverity.LOW
    
    def _calculate_concept_hash(self, definition: str, metadata: Dict[str, Any]) -> str:
        """Calculate hash for concept version tracking."""
        content = json.dumps({
            "definition": definition,
            "metadata": metadata
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    async def start_monitoring(self):
        """Start continuous monitoring of all registered concepts."""
        if self.monitoring_active:
            logger.warning("⚠️  Monitoring already active")
            return
        
        self.monitoring_active = True
        logger.info("🔍 Starting memory divergence monitoring...")
        
        while self.monitoring_active:
            try:
                await self._run_monitoring_cycle()
                await asyncio.sleep(self.scan_interval)
            except Exception as e:
                logger.error(f"❌ Monitoring cycle failed: {str(e)}")
                await asyncio.sleep(60)  # Short wait before retry
    
    async def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.monitoring_active = False
        logger.info("🛑 Memory divergence monitoring stopped")
    
    async def _run_monitoring_cycle(self):
        """Run a single monitoring cycle for all concepts."""
        try:
            logger.debug("🔄 Running memory divergence monitoring cycle...")
            
            monitored_concepts = list(self.active_monitoring.keys())
            cycle_start = datetime.now(timezone.utc)
            
            for concept_id in monitored_concepts:
                if not self.monitoring_active:  # Check if still active
                    break
                
                try:
                    # Update monitoring stats
                    self.active_monitoring[concept_id]["last_scan"] = cycle_start
                    self.active_monitoring[concept_id]["scan_count"] += 1
                    
                    # Take new snapshot (would normally get from memory systems)
                    # For now, simulate with slight variations
                    await self._simulate_concept_update(concept_id)
                    
                    # Detect drift
                    drift_alerts = await self.detect_drift(concept_id)
                    
                    # Emit drift events if found
                    if drift_alerts:
                        await self._emit_drift_alerts(drift_alerts)
                    
                except Exception as e:
                    logger.error(f"❌ Monitoring failed for concept {concept_id}: {str(e)}")
            
            cycle_duration = (datetime.now(timezone.utc) - cycle_start).total_seconds()
            logger.debug(f"✅ Monitoring cycle completed in {cycle_duration:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Monitoring cycle failed: {str(e)}")
    
    async def _simulate_concept_update(self, concept_id: str):
        """Simulate concept updates for testing (would be replaced with real memory queries)."""
        if concept_id not in self.concept_baselines:
            return
        
        baseline = self.concept_baselines[concept_id]
        
        # Simulate slight changes to test drift detection
        import random
        
        modified_definition = baseline.definition
        modified_metadata = baseline.metadata.copy()
        
        # Random chance of introducing drift
        if random.random() < 0.1:  # 10% chance of drift
            # Add speculative language
            modified_definition += " This might also suggest additional implications."
            modified_metadata["confidence"] = modified_metadata.get("confidence", 1.0) * 0.9
        
        # Take snapshot with modifications
        self.take_concept_snapshot(
            concept_id=concept_id,
            current_definition=modified_definition,
            current_metadata=modified_metadata,
            current_sources=baseline.source_references,
            confidence_score=modified_metadata.get("confidence", 1.0)
        )
    
    async def _emit_drift_alerts(self, drift_alerts: List[DriftAlert]):
        """Emit drift alerts to registered callbacks."""
        for alert in drift_alerts:
            # Log drift alert
            logger.warning(
                f"🚨 DRIFT ALERT: {alert.drift_type} detected for concept {alert.concept_id} "
                f"(severity: {alert.severity}, score: {alert.drift_score:.3f})"
            )
            
            # Store in drift log for LoopRecord
            drift_log_entry = {
                "event": "memory_drift_detected",
                "alert_id": alert.alert_id,
                "concept_id": alert.concept_id,
                "drift_type": alert.drift_type,
                "severity": alert.severity,
                "drift_score": alert.drift_score,
                "detection_method": alert.detection_method,
                "timestamp": alert.detected_at.isoformat(),
                "evidence": alert.evidence,
                "ψTrajectory": f"drift_detected_{alert.concept_id}"
            }
            
            # Trigger alert callbacks
            for callback in self.alert_callbacks:
                try:
                    await callback(drift_log_entry)
                except Exception as e:
                    logger.error(f"❌ Alert callback failed: {str(e)}")
    
    def analyze_concept_drift(
        self,
        concept_id: str,
        analysis_period: Optional[Tuple[datetime, datetime]] = None
    ) -> Optional[DriftAnalysis]:
        """
        Analyze drift patterns for a concept over time.
        
        Args:
            concept_id: Concept to analyze
            analysis_period: Time period to analyze (start, end)
            
        Returns:
            DriftAnalysis with comprehensive drift information
        """
        try:
            if concept_id not in self.concept_snapshots:
                return None
            
            # Set default analysis period to last 30 days
            if analysis_period is None:
                end_time = datetime.now(timezone.utc)
                start_time = end_time - timedelta(days=30)
                analysis_period = (start_time, end_time)
            
            # Get alerts in the analysis period
            concept_alerts = [
                alert for alert in self.drift_alerts.values()
                if (alert.concept_id == concept_id and 
                    analysis_period[0] <= alert.detected_at <= analysis_period[1])
            ]
            
            if not concept_alerts:
                # No drift detected - concept is stable
                return DriftAnalysis(
                    concept_id=concept_id,
                    analysis_period=analysis_period,
                    total_drift_events=0,
                    average_drift_score=0.0,
                    drift_velocity=0.0,
                    drift_types_detected=[],
                    most_severe_drift=None,
                    stability_score=1.0,
                    recommendations=["Concept remains stable - no action needed"]
                )
            
            # Calculate drift metrics
            total_events = len(concept_alerts)
            average_score = sum(alert.drift_score for alert in concept_alerts) / total_events
            
            # Calculate drift velocity (drift events per day)
            period_days = (analysis_period[1] - analysis_period[0]).days
            drift_velocity = total_events / max(period_days, 1)
            
            # Find most severe drift
            most_severe = max(concept_alerts, key=lambda a: a.drift_score)
            
            # Get unique drift types
            drift_types = list(set(alert.drift_type for alert in concept_alerts))
            
            # Calculate stability score (inverse of drift intensity)
            max_possible_score = 3.0  # Assuming max drift score is around 3.0
            stability_score = max(0.0, 1.0 - (average_score / max_possible_score))
            
            # Generate recommendations
            recommendations = self._generate_drift_recommendations(concept_alerts, stability_score)
            
            return DriftAnalysis(
                concept_id=concept_id,
                analysis_period=analysis_period,
                total_drift_events=total_events,
                average_drift_score=average_score,
                drift_velocity=drift_velocity,
                drift_types_detected=drift_types,
                most_severe_drift=most_severe,
                stability_score=stability_score,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"❌ Drift analysis failed for concept {concept_id}: {str(e)}")
            return None
    
    def _generate_drift_recommendations(
        self,
        alerts: List[DriftAlert],
        stability_score: float
    ) -> List[str]:
        """Generate recommendations based on drift analysis."""
        recommendations = []
        
        if stability_score < 0.3:
            recommendations.append("CRITICAL: Concept shows severe instability - immediate review required")
        elif stability_score < 0.6:
            recommendations.append("HIGH: Concept requires attention and possible re-verification")
        elif stability_score < 0.8:
            recommendations.append("MEDIUM: Monitor concept for continued drift")
        
        # Type-specific recommendations
        drift_types = set(alert.drift_type for alert in alerts)
        
        if DriftType.EMBEDDING_DRIFT in drift_types:
            recommendations.append("Review concept definition for semantic consistency")
        
        if DriftType.ATTRIBUTION_ERROR in drift_types:
            recommendations.append("Verify and correct source attributions")
        
        if DriftType.SPECULATIVE_OVERGROWTH in drift_types:
            recommendations.append("Trim speculative content not supported by sources")
        
        if DriftType.CONTRADICTORY_EVIDENCE in drift_types:
            recommendations.append("Resolve contradictory information with additional validation")
        
        return recommendations
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        total_alerts = len(self.drift_alerts)
        
        # Count by severity
        severity_counts = {}
        for severity in DriftSeverity:
            severity_counts[severity.value] = len([
                alert for alert in self.drift_alerts.values()
                if alert.severity == severity
            ])
        
        # Count by type
        type_counts = {}
        for drift_type in DriftType:
            type_counts[drift_type.value] = len([
                alert for alert in self.drift_alerts.values()
                if alert.drift_type == drift_type
            ])
        
        return {
            "total_concepts_monitored": len(self.active_monitoring),
            "total_snapshots": sum(len(snapshots) for snapshots in self.concept_snapshots.values()),
            "total_drift_alerts": total_alerts,
            "monitoring_active": self.monitoring_active,
            "scan_interval_hours": self.scan_interval / 3600,
            "alerts_by_severity": severity_counts,
            "alerts_by_type": type_counts,
            "concepts_with_drift": len(set(alert.concept_id for alert in self.drift_alerts.values())),
            "most_recent_alert": max(
                (alert.detected_at for alert in self.drift_alerts.values()),
                default=None
            )
        }
    
    def register_drift_callback(self, callback):
        """Register callback for drift detection events."""
        self.drift_callbacks.append(callback)
    
    def register_alert_callback(self, callback):
        """Register callback for alert events."""
        self.alert_callbacks.append(callback)

# Global instance
memory_divergence_watcher = MemoryDivergenceWatcher()

# Example usage
async def example_drift_detection():
    """Example of how to use the memory divergence watcher."""
    watcher = memory_divergence_watcher
    
    # Register a concept baseline
    watcher.register_concept_baseline(
        concept_id="ai_consciousness",
        definition="The theoretical ability of artificial intelligence systems to have subjective experiences and self-awareness.",
        metadata={"type": "concept", "domain": "AI", "confidence": 1.0},
        source_references=["video_123", "document_456"]
    )
    
    # Take a snapshot after some time
    watcher.take_concept_snapshot(
        concept_id="ai_consciousness",
        current_definition="AI consciousness refers to machines having feelings and thoughts like humans, possibly involving quantum effects.",
        current_metadata={"type": "concept", "domain": "AI", "confidence": 0.7},
        current_sources=["video_123", "speculative_analysis_789"],
        usage_examples=["The AI showed signs of consciousness", "Consciousness emerged from the neural network"]
    )
    
    # Detect drift
    alerts = await watcher.detect_drift("ai_consciousness")
    
    for alert in alerts:
        print(f"Drift detected: {alert.drift_type} (severity: {alert.severity})")
        print(f"Evidence: {alert.evidence}")
