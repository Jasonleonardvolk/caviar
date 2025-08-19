"""
pipeline/storage.py

Storage operations for concepts including Soliton Memory integration
and cognitive interface connections.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional

# Local imports
from .utils import safe_get, get_logger
from .quality import calculate_concept_quality
from .pruning import cluster_similar_concepts

# Setup logger
logger = get_logger(__name__)

# Import storage dependencies with fallback
try:
    from ..cognitive_interface import add_concept_diff
    from ..memory_sculptor import memory_sculptor
    COGNITIVE_INTERFACE_AVAILABLE = True
except ImportError:
    try:
        from cognitive_interface import add_concept_diff
        from memory_sculptor import memory_sculptor
        COGNITIVE_INTERFACE_AVAILABLE = True
    except ImportError:
        # Try importing from pipeline package (new structure)
        try:
            from pipeline import add_concept_diff, memory_sculptor
            COGNITIVE_INTERFACE_AVAILABLE = True
        except ImportError:
            logger.warning("⚠️ Some storage modules not available")
            memory_sculptor = None
            add_concept_diff = None
            COGNITIVE_INTERFACE_AVAILABLE = False

# Import phase bridge from core
try:
    from python.core.psi_phase_bridge import psi_phase_bridge
    PHASE_BRIDGE_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ Phase bridge not available")
    psi_phase_bridge = None
    PHASE_BRIDGE_AVAILABLE = False


async def store_concepts_in_soliton(concepts: List[Dict], doc_metadata: Dict) -> bool:
    """
    Store concepts in Soliton Memory with relationship mapping and entity phase locking.
    
    Args:
        concepts: List of concepts to store
        doc_metadata: Document metadata for context
        
    Returns:
        True if successful, False otherwise
    """
    if not memory_sculptor:
        logger.info("Memory sculptor not available")
        return False
        
    try:
        # Phase locking is now handled through the phase bridge
        phase_locking_available = PHASE_BRIDGE_AVAILABLE
        
        # Get user ID
        user_id = doc_metadata.get('tenant_id', 'default')
        
        # Group concepts by similarity for relationship detection
        concept_clusters = cluster_similar_concepts(concepts)
        
        # Track entity-linked memories for phase bonding
        entity_linked_memories = []
        
        for cluster in concept_clusters:
            # Store primary concept
            primary = cluster[0]
            
            # Check if primary concept has Wikidata ID
            primary_kb_id = None
            if 'metadata' in primary and 'wikidata_id' in primary['metadata']:
                primary_kb_id = primary['metadata']['wikidata_id']
            
            memory_ids = await memory_sculptor.sculpt_and_store(
                user_id=user_id,
                raw_concept=primary,
                metadata={
                    **doc_metadata,
                    'cluster_size': len(cluster),
                    'is_primary': True,
                    'quality_score': calculate_concept_quality(primary, doc_metadata)
                }
            )
            
            # Phase locking through core bridge
            if primary_kb_id and memory_ids and phase_locking_available:
                # Extract numeric ID and calculate phase
                import re
                import numpy as np
                numeric_match = re.match(r'Q(\d+)', str(primary_kb_id))
                if numeric_match:
                    numeric_id = int(numeric_match.group(1))
                    phi = 1.618033988749895  # Golden ratio
                    phase_value = (2 * np.pi * numeric_id / phi) % (2 * np.pi)
                    
                    # Calculate curvature-based amplitude (higher score = lower curvature = higher amplitude)
                    quality_score = calculate_concept_quality(primary, doc_metadata)
                    amplitude_value = 0.3 + 0.7 * quality_score  # Map [0,1] quality to [0.3,1.0] amplitude
                    
                    # Inject phase for primary concept
                    for memory_id in memory_ids:
                        try:
                            # Inject phase into bridge
                            psi_phase_bridge.inject_phase_modulation(memory_id, {
                                'phase_value': phase_value,
                                'amplitude_value': amplitude_value,
                                'curvature_value': 1.0 / (amplitude_value + 0.1),  # Inverse relation
                                'source': 'ingestion_pipeline',
                                'kb_id': primary_kb_id
                            })
                            
                            logger.info(f"✅ Injected phase φ={phase_value:.3f} for {primary['name']} -> {primary_kb_id}")
                            entity_linked_memories.append({
                                'memory_id': memory_id,
                                'kb_id': primary_kb_id,
                                'concept_name': primary['name'],
                                'phase': phase_value,
                                'amplitude': amplitude_value
                            })
                        except Exception as e:
                            logger.error(f"Failed to inject phase for {primary['name']}: {e}")
            
            # Store related concepts with links
            for related in cluster[1:]:
                # Check for Wikidata ID in related concept
                related_kb_id = None
                if 'metadata' in related and 'wikidata_id' in related['metadata']:
                    related_kb_id = related['metadata']['wikidata_id']
                
                related_memory_ids = await memory_sculptor.sculpt_and_store(
                    user_id=user_id,
                    raw_concept=related,
                    metadata={
                        **doc_metadata,
                        'primary_concept': primary['name'],
                        'relationship': 'semantic_cluster',
                        'quality_score': calculate_concept_quality(related, doc_metadata)
                    }
                )
                
                # Phase locking for related concepts
                if related_kb_id and related_memory_ids and phase_locking_available:
                    import re
                    import numpy as np
                    numeric_match = re.match(r'Q(\d+)', str(related_kb_id))
                    if numeric_match:
                        numeric_id = int(numeric_match.group(1))
                        phi = 1.618033988749895
                        phase_value = (2 * np.pi * numeric_id / phi) % (2 * np.pi)
                        
                        # Lower amplitude for related concepts
                        quality_score = calculate_concept_quality(related, doc_metadata)
                        amplitude_value = 0.2 + 0.5 * quality_score  # Map to [0.2,0.7]
                        
                        for memory_id in related_memory_ids:
                            try:
                                psi_phase_bridge.inject_phase_modulation(memory_id, {
                                    'phase_value': phase_value,
                                    'amplitude_value': amplitude_value,
                                    'curvature_value': 1.0 / (amplitude_value + 0.1),
                                    'source': 'ingestion_pipeline',
                                    'kb_id': related_kb_id,
                                    'primary_concept': primary['name']
                                })
                                
                                logger.info(f"✅ Injected phase φ={phase_value:.3f} for {related['name']} -> {related_kb_id}")
                                entity_linked_memories.append({
                                    'memory_id': memory_id,
                                    'kb_id': related_kb_id,
                                    'concept_name': related['name'],
                                    'phase': phase_value,
                                    'amplitude': amplitude_value
                                })
                            except Exception as e:
                                logger.error(f"Failed to inject phase for {related['name']}: {e}")
                
                # Propagate phase through concept cluster
                if primary_kb_id and related_kb_id and memory_ids and phase_locking_available:
                    # Check phase coherence between concepts
                    primary_phase = entity_linked_memories[-2]['phase'] if len(entity_linked_memories) >= 2 else 0
                    related_phase = entity_linked_memories[-1]['phase'] if entity_linked_memories else 0
                    phase_diff = abs(primary_phase - related_phase)
                    
                    # Flag potential vortex if phase difference is large
                    if phase_diff > np.pi:
                        logger.warning(f"⚠️ Large phase difference detected: {primary_kb_id} <-> {related_kb_id} (Δφ={phase_diff:.2f})")
                        # This will be used for vortex detection
                        entity_linked_memories[-1]['potential_vortex'] = True
        
        # Log summary and detect vortices
        if entity_linked_memories:
            logger.info(f"🔗 Injected phase for {len(entity_linked_memories)} entity-linked concepts")
            
            # Detect vortex-prone paths
            vortex_candidates = [m for m in entity_linked_memories if m.get('potential_vortex', False)]
            if vortex_candidates:
                logger.warning(f"🌀 Detected {len(vortex_candidates)} potential phase vortices")
                
                # Propagate phase changes through mesh to check for actual vortices
                if psi_phase_bridge:
                    try:
                        # Start from highest amplitude concept
                        highest_amp = max(entity_linked_memories, key=lambda x: x.get('amplitude', 0))
                        await psi_phase_bridge.propagate_phase_through_mesh(
                            initial_concept_id=highest_amp['memory_id'],
                            phase_value=highest_amp['phase'],
                            amplitude_value=highest_amp['amplitude'],
                            propagation_depth=2,
                            decay_factor=0.9
                        )
                        
                        # Check for vortices
                        vortices = await psi_phase_bridge.detect_phase_vortices_in_mesh(min_loop_size=3)
                        if vortices:
                            logger.warning(f"🌀 Confirmed {len(vortices)} phase vortices in concept network")
                            # Store vortex info for Ghost UI
                            doc_metadata['phase_vortices'] = [{
                                'loop': v['loop'][:3] + ['...'] if len(v['loop']) > 3 else v['loop'],
                                'winding_number': v['winding_number']
                            } for v in vortices]
                    except Exception as e:
                        logger.error(f"Failed to detect vortices: {e}")
            
        logger.info(f"✅ Stored {len(concepts)} concepts in {len(concept_clusters)} clusters with entity phase locking")
        return True
        
    except Exception as e:
        logger.error(f"Failed to store concepts in Soliton: {e}")
        return False


def inject_concept_diff(concepts: List[Dict], doc_metadata: Dict, doc_name: str) -> bool:
    """
    Inject concepts into the cognitive interface.
    
    Args:
        concepts: List of concepts to inject
        doc_metadata: Document metadata
        doc_name: Document name/title
        
    Returns:
        True if successful, False otherwise
    """
    if not COGNITIVE_INTERFACE_AVAILABLE or not add_concept_diff:
        logger.warning("Cognitive interface not available - skipping concept diff injection")
        return False
        
    try:
        # Prepare concept diff data
        concept_diff_data = {
            "type": "document",
            "title": doc_name,
            "concepts": concepts,
            "summary": f"{len(concepts)} concepts extracted.",
            "metadata": doc_metadata,
            "timestamp": doc_metadata.get('timestamp', ''),
            "source": "ingest_pipeline"
        }
        
        # Generate a concept ID from document name
        concept_id = f"doc_{doc_name.replace(' ', '_').replace('.', '_').replace('/', '_')}"
        
        # Ensure we have valid parameters
        if not concept_id or not concept_diff_data:
            logger.error("Invalid parameters for add_concept_diff: concept_id or diff_data is empty")
            return False
        
        # Call the function with proper parameters
        logger.info(f"Injecting concept diff for document: {doc_name} (ID: {concept_id})")
        result = add_concept_diff(concept_id, concept_diff_data)
        
        if result:
            logger.info(f"✅ Successfully injected concept diff for {doc_name} with {len(concepts)} concepts")
            return True
        else:
            logger.warning(f"⚠️ Concept diff injection returned None/False for {doc_name}")
            return False
        
    except TypeError as e:
        logger.error(f"Concept diff injection failed - TypeError: {e}")
        logger.error(f"Function signature issue - check add_concept_diff parameters")
        return False
    except Exception as e:
        logger.error(f"Concept diff injection failed - Unexpected error: {e}")
        logger.error(f"Concept ID: {concept_id if 'concept_id' in locals() else 'Not set'}")
        logger.error(f"Diff data keys: {list(concept_diff_data.keys()) if 'concept_diff_data' in locals() else 'Not set'}")
        return False


def prepare_storage_metadata(doc_metadata: Dict, 
                           title_text: str, 
                           abstract_text: str) -> Dict[str, Any]:
    """
    Prepare metadata for concept storage.
    
    Args:
        doc_metadata: Basic document metadata
        title_text: Extracted title
        abstract_text: Extracted abstract
        
    Returns:
        Enhanced metadata dictionary
    """
    enhanced_metadata = doc_metadata.copy()
    
    # Add extracted text metadata
    if title_text:
        enhanced_metadata['extracted_title'] = title_text
    if abstract_text:
        enhanced_metadata['extracted_abstract'] = abstract_text[:500]  # Limit length
    
    # Add processing metadata
    enhanced_metadata['processing_pipeline'] = 'tori_enhanced_v2'
    enhanced_metadata['storage_backend'] = 'soliton_memory'
    
    return enhanced_metadata
