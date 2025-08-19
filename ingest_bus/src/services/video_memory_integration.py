"""
Video Memory Integration Service for TORI

This module handles the integration of processed video content with TORI's
cognitive memory systems including ConceptMesh, BraidMemory, LoopRecord,
ψMesh, and ScholarSphere. It ensures that video-derived knowledge becomes
a first-class citizen in TORI's memory architecture.

Key Integration Points:
- ConceptMesh: Add video concepts and their relationships
- BraidMemory: Link video content with existing memories
- LoopRecord: Log video ingestion events with time anchors
- ψMesh: Update semantic network with video knowledge
- ScholarSphere: Archive video content for long-term storage
- ψTrajectory: Track conceptual journey through video content
"""

import asyncio
import logging
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
import uuid

# Data structures
from dataclasses import dataclass, asdict
from enum import Enum

# Import video processing results
from .video_ingestion_service import VideoIngestionResult, ExtractedConcept, TranscriptSegment

# Configure logging
logger = logging.getLogger("tori.video_memory")

class MemoryIntegrationType(str, Enum):
    """Types of memory integration."""
    CONCEPT_MESH = "concept_mesh"
    BRAID_MEMORY = "braid_memory"
    LOOP_RECORD = "loop_record"
    PSI_MESH = "psi_mesh"
    SCHOLAR_SPHERE = "scholar_sphere"
    PSI_TRAJECTORY = "psi_trajectory"

@dataclass
class VideoMemoryNode:
    """Memory node representing video content in TORI's memory."""
    node_id: str
    video_id: str
    node_type: str  # segment, concept, speaker, etc.
    content: Dict[str, Any]
    timestamp_range: Tuple[float, float]
    source_hash: str
    confidence: float
    
    # Memory relationships
    concept_links: List[str] = None
    memory_braids: List[str] = None
    psi_anchors: List[str] = None
    
    def __post_init__(self):
        if self.concept_links is None:
            self.concept_links = []
        if self.memory_braids is None:
            self.memory_braids = []
        if self.psi_anchors is None:
            self.psi_anchors = []

@dataclass
class ConceptRelationship:
    """Relationship between concepts in video content."""
    source_concept: str
    target_concept: str
    relationship_type: str  # co_occurrence, temporal_sequence, causal, etc.
    confidence: float
    evidence: List[str]  # segment IDs where relationship is evident
    timestamp_range: Tuple[float, float]

@dataclass
class MemoryIntegrationResult:
    """Result of memory integration process."""
    video_id: str
    integration_types: List[MemoryIntegrationType]
    nodes_created: int
    relationships_created: int
    concepts_linked: int
    memories_braided: int
    psi_anchors_added: int
    scholar_sphere_archived: bool
    trajectory_updated: bool
    integration_time: float
    integrity_verified: bool

class VideoMemoryIntegrationService:
    """
    Service for integrating video content into TORI's memory systems.
    
    This service takes processed video results and integrates them into
    TORI's comprehensive memory architecture, ensuring that video knowledge
    becomes searchable, cross-referenceable, and contextually available.
    """
    
    def __init__(self):
        """Initialize the memory integration service."""
        self.integration_cache = {}
        self.concept_graph = {}  # Simple in-memory concept graph
        self.memory_store = {}   # Simple in-memory storage
        self.trajectory_log = [] # ψTrajectory tracking
        
    async def integrate_video_memory(
        self,
        result: VideoIngestionResult,
        integration_options: Optional[Dict[str, Any]] = None
    ) -> MemoryIntegrationResult:
        """
        Integrate video processing results into TORI's memory systems.
        
        Args:
            result: Processed video ingestion result
            integration_options: Optional integration configuration
            
        Returns:
            Memory integration result with statistics
        """
        try:
            start_time = datetime.now(timezone.utc)
            logger.info(f"Starting memory integration for video: {result.video_id}")
            
            # Default integration options
            if integration_options is None:
                integration_options = {}
            
            default_options = {
                "enable_concept_mesh": True,
                "enable_braid_memory": True,
                "enable_loop_record": True,
                "enable_psi_mesh": True,
                "enable_scholar_sphere": True,
                "enable_trajectory_tracking": True,
                "create_relationships": True,
                "verify_integrity": True
            }
            
            options = {**default_options, **integration_options}
            
            # Track integration results
            integration_result = MemoryIntegrationResult(
                video_id=result.video_id,
                integration_types=[],
                nodes_created=0,
                relationships_created=0,
                concepts_linked=0,
                memories_braided=0,
                psi_anchors_added=0,
                scholar_sphere_archived=False,
                trajectory_updated=False,
                integration_time=0.0,
                integrity_verified=False
            )
            
            # Step 1: ConceptMesh Integration
            if options["enable_concept_mesh"]:
                await self._integrate_concept_mesh(result, integration_result)
                integration_result.integration_types.append(MemoryIntegrationType.CONCEPT_MESH)
            
            # Step 2: BraidMemory Integration
            if options["enable_braid_memory"]:
                await self._integrate_braid_memory(result, integration_result)
                integration_result.integration_types.append(MemoryIntegrationType.BRAID_MEMORY)
            
            # Step 3: LoopRecord Integration
            if options["enable_loop_record"]:
                await self._integrate_loop_record(result, integration_result)
                integration_result.integration_types.append(MemoryIntegrationType.LOOP_RECORD)
            
            # Step 4: ψMesh Integration
            if options["enable_psi_mesh"]:
                await self._integrate_psi_mesh(result, integration_result)
                integration_result.integration_types.append(MemoryIntegrationType.PSI_MESH)
            
            # Step 5: ScholarSphere Archival
            if options["enable_scholar_sphere"]:
                await self._integrate_scholar_sphere(result, integration_result)
                integration_result.integration_types.append(MemoryIntegrationType.SCHOLAR_SPHERE)
            
            # Step 6: ψTrajectory Tracking
            if options["enable_trajectory_tracking"]:
                await self._update_psi_trajectory(result, integration_result)
                integration_result.integration_types.append(MemoryIntegrationType.PSI_TRAJECTORY)
            
            # Step 7: Create Concept Relationships
            if options["create_relationships"]:
                await self._create_concept_relationships(result, integration_result)
            
            # Step 8: Verify Integration Integrity
            if options["verify_integrity"]:
                integration_result.integrity_verified = await self._verify_integration_integrity(
                    result, integration_result
                )
            
            # Calculate integration time
            integration_result.integration_time = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds()
            
            # Cache integration result
            self.integration_cache[result.video_id] = integration_result
            
            logger.info(
                f"✅ Memory integration completed for {result.video_id}: "
                f"{integration_result.nodes_created} nodes, "
                f"{integration_result.concepts_linked} concepts, "
                f"{integration_result.integration_time:.2f}s"
            )
            
            return integration_result
            
        except Exception as e:
            logger.error(f"❌ Memory integration failed for {result.video_id}: {str(e)}")
            raise
    
    async def _integrate_concept_mesh(
        self,
        result: VideoIngestionResult,
        integration_result: MemoryIntegrationResult
    ):
        """Integrate concepts into ConceptMesh."""
        try:
            logger.info(f"🕸️ Integrating {len(result.concepts)} concepts into ConceptMesh")
            
            for concept in result.concepts:
                # Create concept node
                concept_id = f"video_concept_{result.video_id}_{concept.term.replace(' ', '_')}"
                
                concept_node = VideoMemoryNode(
                    node_id=concept_id,
                    video_id=result.video_id,
                    node_type="concept",
                    content={
                        "term": concept.term,
                        "type": concept.concept_type,
                        "confidence": concept.confidence,
                        "context": concept.context,
                        "source_segments": concept.source_segments,
                        "timestamp_ranges": concept.timestamp_ranges
                    },
                    timestamp_range=(
                        min([tr[0] for tr in concept.timestamp_ranges]),
                        max([tr[1] for tr in concept.timestamp_ranges])
                    ),
                    source_hash=result.source_hash,
                    confidence=concept.confidence
                )
                
                # Add to concept graph
                if concept.term not in self.concept_graph:
                    self.concept_graph[concept.term] = []
                
                self.concept_graph[concept.term].append(concept_node)
                
                # Store in memory
                self.memory_store[concept_id] = concept_node
                
                integration_result.concepts_linked += 1
                integration_result.nodes_created += 1
            
            logger.info(f"✅ ConceptMesh integration completed: {integration_result.concepts_linked} concepts")
            
        except Exception as e:
            logger.error(f"❌ ConceptMesh integration failed: {str(e)}")
            raise
    
    async def _integrate_braid_memory(
        self,
        result: VideoIngestionResult,
        integration_result: MemoryIntegrationResult
    ):
        """Integrate content into BraidMemory system."""
        try:
            logger.info(f"🧠 Integrating {len(result.segments)} segments into BraidMemory")
            
            # Create memory braids for each segment
            for segment in result.segments:
                braid_id = f"video_braid_{result.video_id}_{segment['id']}"
                
                # Create segment memory node
                segment_node = VideoMemoryNode(
                    node_id=braid_id,
                    video_id=result.video_id,
                    node_type="segment",
                    content={
                        "segment_id": segment["id"],
                        "text": segment["text"],
                        "summary": segment["summary"],
                        "topic": segment["topic"],
                        "speakers": segment["speakers"],
                        "visual_context": len(segment.get("visual_context", [])),
                        "start_time": segment["start_time"],
                        "end_time": segment["end_time"]
                    },
                    timestamp_range=(segment["start_time"], segment["end_time"]),
                    source_hash=result.source_hash,
                    confidence=0.9  # High confidence for direct transcript content
                )
                
                # Link to related concepts
                segment_concepts = []
                for concept in result.concepts:
                    if segment["id"] in concept.source_segments:
                        concept_id = f"video_concept_{result.video_id}_{concept.term.replace(' ', '_')}"
                        segment_node.concept_links.append(concept_id)
                        segment_concepts.append(concept.term)
                
                # Create memory braid entry
                braid_entry = {
                    "braid_id": braid_id,
                    "content_type": "video_segment",
                    "content": segment_node.content,
                    "linked_concepts": segment_concepts,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "source_video": result.video_id,
                    "integrity_hash": hashlib.sha256(
                        json.dumps(segment_node.content, sort_keys=True).encode()
                    ).hexdigest()[:16]
                }
                
                # Store braid
                segment_node.memory_braids.append(braid_id)
                self.memory_store[braid_id] = segment_node
                self.memory_store[f"{braid_id}_braid"] = braid_entry
                
                integration_result.memories_braided += 1
                integration_result.nodes_created += 1
            
            logger.info(f"✅ BraidMemory integration completed: {integration_result.memories_braided} braids")
            
        except Exception as e:
            logger.error(f"❌ BraidMemory integration failed: {str(e)}")
            raise
    
    async def _integrate_loop_record(
        self,
        result: VideoIngestionResult,
        integration_result: MemoryIntegrationResult
    ):
        """Integrate video event into LoopRecord."""
        try:
            logger.info(f"⏰ Creating LoopRecord entry for video: {result.video_id}")
            
            # Create LoopRecord entry
            loop_entry = {
                "record_id": f"video_loop_{result.video_id}",
                "event_type": "video_ingestion",
                "video_id": result.video_id,
                "timestamp": result.created_at.isoformat(),
                "duration": result.duration,
                "processing_time": result.processing_time,
                "source_file": result.source_file,
                "source_hash": result.source_hash,
                "integrity_score": result.integrity_score,
                "content_summary": {
                    "segments_count": len(result.segments),
                    "concepts_count": len(result.concepts),
                    "questions_count": len(result.questions),
                    "speakers_count": len(result.speakers),
                    "trust_flags_count": len(result.trust_flags)
                },
                "psi_time_anchor": result.created_at.timestamp(),
                "trajectory_marker": f"video_ingestion_{result.video_id}"
            }
            
            # Add to trajectory log
            self.trajectory_log.append(loop_entry)
            
            # Store in memory
            loop_record_id = f"loop_record_{result.video_id}"
            self.memory_store[loop_record_id] = loop_entry
            
            logger.info(f"✅ LoopRecord entry created: {loop_entry['record_id']}")
            
        except Exception as e:
            logger.error(f"❌ LoopRecord integration failed: {str(e)}")
            raise
    
    async def _integrate_psi_mesh(
        self,
        result: VideoIngestionResult,
        integration_result: MemoryIntegrationResult
    ):
        """Integrate video content into ψMesh semantic network."""
        try:
            logger.info(f"🌀 Integrating video into ψMesh: {result.video_id}")
            
            # Create ψMesh anchors for key content
            psi_anchors = []
            
            # Anchor for overall video
            video_anchor = {
                "anchor_id": f"psi_video_{result.video_id}",
                "anchor_type": "video_content",
                "content_summary": {
                    "title": f"Video Content: {result.source_file}",
                    "duration": result.duration,
                    "key_concepts": [c.term for c in result.concepts[:10]],
                    "question_count": len(result.questions),
                    "integrity_score": result.integrity_score
                },
                "semantic_weight": min(result.integrity_score * len(result.concepts) / 10, 1.0),
                "temporal_anchor": result.created_at.timestamp(),
                "concept_density": len(result.concepts) / max(result.duration / 60, 1),  # concepts per minute
                "linked_nodes": []
            }
            
            psi_anchors.append(video_anchor)
            
            # Create anchors for significant segments
            for segment in result.segments:
                if segment.get("topic") and len(segment.get("text", "")) > 100:
                    segment_anchor = {
                        "anchor_id": f"psi_segment_{result.video_id}_{segment['id']}",
                        "anchor_type": "video_segment",
                        "content_summary": {
                            "topic": segment["topic"],
                            "summary": segment["summary"],
                            "duration": segment["end_time"] - segment["start_time"],
                            "speaker_count": len(segment.get("speakers", []))
                        },
                        "semantic_weight": 0.7,
                        "temporal_anchor": segment["start_time"],
                        "parent_anchor": video_anchor["anchor_id"],
                        "linked_nodes": []
                    }
                    psi_anchors.append(segment_anchor)
            
            # Store ψMesh anchors
            for anchor in psi_anchors:
                anchor_id = anchor["anchor_id"]
                self.memory_store[anchor_id] = anchor
                
                # Link concepts to anchors
                for concept in result.concepts:
                    concept_id = f"video_concept_{result.video_id}_{concept.term.replace(' ', '_')}"
                    if concept_id in self.memory_store:
                        self.memory_store[concept_id].psi_anchors.append(anchor_id)
                        anchor["linked_nodes"].append(concept_id)
                
                integration_result.psi_anchors_added += 1
            
            logger.info(f"✅ ψMesh integration completed: {len(psi_anchors)} anchors")
            
        except Exception as e:
            logger.error(f"❌ ψMesh integration failed: {str(e)}")
            raise
    
    async def _integrate_scholar_sphere(
        self,
        result: VideoIngestionResult,
        integration_result: MemoryIntegrationResult
    ):
        """Archive video content in ScholarSphere."""
        try:
            logger.info(f"📚 Archiving video in ScholarSphere: {result.video_id}")
            
            # Create ScholarSphere archive entry
            archive_entry = {
                "archive_id": f"scholar_video_{result.video_id}",
                "content_type": "video_archive",
                "source_file": result.source_file,
                "source_hash": result.source_hash,
                "archived_at": datetime.now(timezone.utc).isoformat(),
                "metadata": {
                    "duration": result.duration,
                    "file_size": result.file_size,
                    "processing_time": result.processing_time,
                    "integrity_score": result.integrity_score,
                    "concept_count": len(result.concepts),
                    "segment_count": len(result.segments),
                    "speaker_count": len(result.speakers)
                },
                "content_digest": {
                    "key_concepts": [c.term for c in result.concepts[:20]],
                    "main_topics": [s.get("topic", "Unknown") for s in result.segments[:10]],
                    "questions_raised": result.questions[:10],
                    "ghost_reflections": [r.get("message", "") for r in result.ghost_reflections]
                },
                "access_controls": {
                    "public": False,
                    "searchable": True,
                    "concept_extractable": True
                },
                "version": "1.0",
                "curator": "TORI_VideoIngestion_v1.0"
            }
            
            # Store in ScholarSphere
            archive_id = f"scholar_sphere_{result.video_id}"
            self.memory_store[archive_id] = archive_entry
            
            integration_result.scholar_sphere_archived = True
            
            logger.info(f"✅ ScholarSphere archival completed: {archive_entry['archive_id']}")
            
        except Exception as e:
            logger.error(f"❌ ScholarSphere integration failed: {str(e)}")
            raise
    
    async def _update_psi_trajectory(
        self,
        result: VideoIngestionResult,
        integration_result: MemoryIntegrationResult
    ):
        """Update ψTrajectory with video content progression."""
        try:
            logger.info(f"🛤️ Updating ψTrajectory for video: {result.video_id}")
            
            # Create trajectory markers for concept evolution
            trajectory_markers = []
            
            # Overall video trajectory marker
            video_marker = {
                "marker_id": f"trajectory_video_{result.video_id}",
                "marker_type": "video_ingestion",
                "timestamp": result.created_at.timestamp(),
                "concept_snapshot": [c.term for c in result.concepts],
                "trajectory_metadata": {
                    "video_duration": result.duration,
                    "concept_density": len(result.concepts) / max(result.duration / 60, 1),
                    "knowledge_domains": list(set([c.concept_type for c in result.concepts])),
                    "integrity_score": result.integrity_score
                },
                "progression_indicators": {
                    "new_concepts_introduced": len(result.concepts),
                    "questions_raised": len(result.questions),
                    "knowledge_gaps_identified": len([q for q in result.questions if "?" in q]),
                    "action_items_generated": len(result.action_items)
                }
            }
            
            trajectory_markers.append(video_marker)
            
            # Segment-level trajectory markers for significant content
            for i, segment in enumerate(result.segments):
                if len(segment.get("text", "")) > 200:  # Significant segments only
                    segment_marker = {
                        "marker_id": f"trajectory_segment_{result.video_id}_{i}",
                        "marker_type": "video_segment",
                        "timestamp": result.created_at.timestamp() + segment["start_time"],
                        "concept_snapshot": [
                            c.term for c in result.concepts 
                            if segment["id"] in c.source_segments
                        ],
                        "segment_metadata": {
                            "topic": segment.get("topic", "Unknown"),
                            "duration": segment["end_time"] - segment["start_time"],
                            "speaker_count": len(segment.get("speakers", []))
                        },
                        "parent_marker": video_marker["marker_id"]
                    }
                    trajectory_markers.append(segment_marker)
            
            # Store trajectory markers
            for marker in trajectory_markers:
                marker_id = marker["marker_id"]
                self.memory_store[marker_id] = marker
                self.trajectory_log.append(marker)
            
            integration_result.trajectory_updated = True
            
            logger.info(f"✅ ψTrajectory updated: {len(trajectory_markers)} markers")
            
        except Exception as e:
            logger.error(f"❌ ψTrajectory update failed: {str(e)}")
            raise
    
    async def _create_concept_relationships(
        self,
        result: VideoIngestionResult,
        integration_result: MemoryIntegrationResult
    ):
        """Create relationships between concepts found in video."""
        try:
            logger.info(f"🔗 Creating concept relationships for {len(result.concepts)} concepts")
            
            relationships = []
            
            # Find co-occurring concepts
            for i, concept_a in enumerate(result.concepts):
                for j, concept_b in enumerate(result.concepts[i+1:], i+1):
                    # Check for co-occurrence in segments
                    shared_segments = set(concept_a.source_segments) & set(concept_b.source_segments)
                    
                    if shared_segments:
                        relationship = ConceptRelationship(
                            source_concept=concept_a.term,
                            target_concept=concept_b.term,
                            relationship_type="co_occurrence",
                            confidence=len(shared_segments) / max(len(concept_a.source_segments), len(concept_b.source_segments)),
                            evidence=list(shared_segments),
                            timestamp_range=(
                                min([tr[0] for tr in concept_a.timestamp_ranges + concept_b.timestamp_ranges]),
                                max([tr[1] for tr in concept_a.timestamp_ranges + concept_b.timestamp_ranges])
                            )
                        )
                        relationships.append(relationship)
            
            # Find temporal sequences
            sorted_concepts = sorted(result.concepts, key=lambda c: min([tr[0] for tr in c.timestamp_ranges]))
            
            for i in range(len(sorted_concepts) - 1):
                current_concept = sorted_concepts[i]
                next_concept = sorted_concepts[i + 1]
                
                current_end = max([tr[1] for tr in current_concept.timestamp_ranges])
                next_start = min([tr[0] for tr in next_concept.timestamp_ranges])
                
                # If concepts are mentioned within 30 seconds of each other
                if next_start - current_end < 30:
                    relationship = ConceptRelationship(
                        source_concept=current_concept.term,
                        target_concept=next_concept.term,
                        relationship_type="temporal_sequence",
                        confidence=0.8,
                        evidence=[],
                        timestamp_range=(current_end, next_start)
                    )
                    relationships.append(relationship)
            
            # Store relationships
            for rel in relationships:
                rel_id = f"relationship_{result.video_id}_{hashlib.sha256((rel.source_concept + rel.target_concept).encode()).hexdigest()[:8]}"
                self.memory_store[rel_id] = asdict(rel)
                integration_result.relationships_created += 1
            
            logger.info(f"✅ Created {len(relationships)} concept relationships")
            
        except Exception as e:
            logger.error(f"❌ Concept relationship creation failed: {str(e)}")
            raise
    
    async def _verify_integration_integrity(
        self,
        result: VideoIngestionResult,
        integration_result: MemoryIntegrationResult
    ) -> bool:
        """Verify integrity of memory integration."""
        try:
            logger.info(f"🔍 Verifying integration integrity for: {result.video_id}")
            
            integrity_checks = []
            
            # Check 1: All concepts have memory nodes
            concepts_in_memory = 0
            for concept in result.concepts:
                concept_id = f"video_concept_{result.video_id}_{concept.term.replace(' ', '_')}"
                if concept_id in self.memory_store:
                    concepts_in_memory += 1
            
            concept_integrity = concepts_in_memory / len(result.concepts) if result.concepts else 1.0
            integrity_checks.append(("concepts", concept_integrity))
            
            # Check 2: All segments have braid entries
            segments_in_memory = 0
            for segment in result.segments:
                braid_id = f"video_braid_{result.video_id}_{segment['id']}"
                if braid_id in self.memory_store:
                    segments_in_memory += 1
            
            segment_integrity = segments_in_memory / len(result.segments) if result.segments else 1.0
            integrity_checks.append(("segments", segment_integrity))
            
            # Check 3: LoopRecord entry exists
            loop_record_id = f"loop_record_{result.video_id}"
            loop_integrity = 1.0 if loop_record_id in self.memory_store else 0.0
            integrity_checks.append(("loop_record", loop_integrity))
            
            # Check 4: ψMesh anchors exist
            video_anchor_id = f"psi_video_{result.video_id}"
            psi_integrity = 1.0 if video_anchor_id in self.memory_store else 0.0
            integrity_checks.append(("psi_mesh", psi_integrity))
            
            # Check 5: ScholarSphere archive exists
            archive_id = f"scholar_sphere_{result.video_id}"
            scholar_integrity = 1.0 if archive_id in self.memory_store else 0.0
            integrity_checks.append(("scholar_sphere", scholar_integrity))
            
            # Calculate overall integrity score
            overall_integrity = sum([score for _, score in integrity_checks]) / len(integrity_checks)
            
            logger.info(f"✅ Integration integrity verified: {overall_integrity:.2%}")
            for check_name, score in integrity_checks:
                logger.info(f"  - {check_name}: {score:.2%}")
            
            return overall_integrity > 0.9  # 90% threshold for success
            
        except Exception as e:
            logger.error(f"❌ Integration integrity verification failed: {str(e)}")
            return False
    
    def search_video_memory(
        self,
        query: str,
        memory_types: Optional[List[str]] = None,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search integrated video memory.
        
        Args:
            query: Search query
            memory_types: Types of memory to search
            max_results: Maximum results to return
            
        Returns:
            Search results from integrated memory
        """
        try:
            if memory_types is None:
                memory_types = ["concept", "segment", "video"]
            
            results = []
            query_lower = query.lower()
            
            for node_id, node_data in self.memory_store.items():
                if isinstance(node_data, VideoMemoryNode):
                    if node_data.node_type in memory_types:
                        # Simple text matching
                        content_text = json.dumps(node_data.content).lower()
                        if query_lower in content_text:
                            results.append({
                                "node_id": node_id,
                                "node_type": node_data.node_type,
                                "video_id": node_data.video_id,
                                "content": node_data.content,
                                "confidence": node_data.confidence,
                                "timestamp_range": node_data.timestamp_range
                            })
                elif isinstance(node_data, dict):
                    # Search in dict-based nodes
                    content_text = json.dumps(node_data).lower()
                    if query_lower in content_text:
                        results.append({
                            "node_id": node_id,
                            "node_type": "archive",
                            "content": node_data
                        })
            
            # Sort by relevance (simple frequency-based)
            results.sort(key=lambda x: x.get("confidence", 0.5), reverse=True)
            
            return results[:max_results]
            
        except Exception as e:
            logger.error(f"Video memory search failed: {str(e)}")
            return []
    
    def get_video_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about integrated video memory."""
        try:
            stats = {
                "total_nodes": len(self.memory_store),
                "concepts_in_graph": len(self.concept_graph),
                "trajectory_markers": len(self.trajectory_log),
                "videos_integrated": len(self.integration_cache),
                "memory_breakdown": {
                    "concept_nodes": 0,
                    "segment_nodes": 0,
                    "archive_nodes": 0,
                    "relationship_nodes": 0,
                    "psi_anchors": 0,
                    "loop_records": 0
                }
            }
            
            # Count node types
            for node_data in self.memory_store.values():
                if isinstance(node_data, VideoMemoryNode):
                    node_type = node_data.node_type
                    if node_type in stats["memory_breakdown"]:
                        stats["memory_breakdown"][f"{node_type}_nodes"] += 1
                elif isinstance(node_data, dict):
                    if "archive_id" in node_data:
                        stats["memory_breakdown"]["archive_nodes"] += 1
                    elif "relationship" in str(node_data):
                        stats["memory_breakdown"]["relationship_nodes"] += 1
                    elif "anchor_id" in node_data:
                        stats["memory_breakdown"]["psi_anchors"] += 1
                    elif "record_id" in node_data:
                        stats["memory_breakdown"]["loop_records"] += 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting memory stats: {str(e)}")
            return {"error": str(e)}

# Global service instance
video_memory_service = VideoMemoryIntegrationService()
