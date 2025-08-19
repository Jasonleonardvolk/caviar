"""
Core pipeline module for concept processing with alias normalization.
This module handles loading and merging concept data with canonical name mapping.
"""

import json
import functools
import yaml
import threading
import time
import numpy as np
import hdbscan
from pathlib import Path
from typing import List, Dict, Any, Set
import asyncio


# ---------- NEW: live-reload ingestion thresholds ----------
_CFG_PATH = Path(__file__).parent / "config" / "ingestion_config.yaml"

def _load_cfg():
    try:
        with open(_CFG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"[Warning] Config file not found at {_CFG_PATH}, using defaults")
        return {}
    except Exception as e:
        print(f"[Error] Failed to load config: {e}, using defaults")
        return {}

_CFG = _load_cfg()

def cfg(key, default=None):
    return _CFG.get(key, default)

# watch for manual edits (every 30 s)
def _watch_cfg():
    while True:
        time.sleep(30)
        try:
            globals()["_CFG"] = _load_cfg()
        except Exception:
            pass

threading.Thread(target=_watch_cfg, daemon=True).start()

ENABLE_CONTEXT_EXTRACTION = cfg("enable_context_extraction", True)
ENABLE_FREQUENCY_TRACKING = cfg("enable_frequency_tracking", True)
ENABLE_SMART_FILTERING = cfg("enable_smart_filtering", True)
ENABLE_CONCEPT_CLUSTERING = cfg("enable_concept_clustering", True)


def load_universal_concept_file_storage() -> List[Dict[str, Any]]:
    """
    Load and merge concepts from various sources with alias normalization.
    
    Returns:
        List of merged concepts with canonical names applied.
    """
    # Simulate loading concepts from multiple PDF sources
    pdf_sources = [
        ("paper1.pdf", [
            {"name": "GPU", "description": "Graphics processing unit for parallel computation"},
            {"name": "CPU", "description": "Central processing unit - main processor"},
            {"name": "machine learning", "description": "AI algorithms that learn from data"},
        ]),
        ("paper2.pdf", [
            {"name": "graphics processor", "description": "Processor specialized for graphics rendering"},
            {"name": "GPU", "description": "High-performance graphics processor"},
            {"name": "neural network", "description": "Deep learning architecture"},
        ]),
        ("paper3.pdf", [
            {"name": "processor", "description": "General purpose computing unit"},
            {"name": "CPU", "description": "Main computer processor"},
            {"name": "GPU", "description": "Accelerated graphics processor"},
        ])
    ]
    
    # Extract concepts with provenance tracking
    all_concepts = []
    for pdf_path, concepts in pdf_sources:
        boosted_concepts = extract_and_boost_concepts(concepts, pdf_path)
        all_concepts.extend(boosted_concepts)
    
    # Deduplicate while preserving provenance
    final_concepts = extract_and_boost_concepts(all_concepts, "merged_sources")
    
    # ---------- NEW: merge with alias map ----------
    try:
        with open(Path(__file__).parent / "data" / "concept_aliases.json", "r", encoding="utf-8") as f:
            ALIAS_MAP = json.load(f)            # {"gpu":"graphics processor", ...}
    except FileNotFoundError:
        ALIAS_MAP = {}

    def canonical(name: str) -> str:
        """Convert a concept name to its canonical form using the alias map."""
        return ALIAS_MAP.get(name.lower(), name.lower())

    merged_concepts = []
    seen: Set[str] = set()
    
    for c in final_concepts:
        canon = canonical(c["name"])
        if canon not in seen:
            seen.add(canon)
            c["canonical"] = canon
            merged_concepts.append(c)
        else:
            # Merge provenance for concepts with same canonical form
            idx = next(i for i, v in enumerate(merged_concepts) if v["canonical"] == canon)
            # Merge sources
            existing_sources = set(merged_concepts[idx].get("sources", []))
            new_sources = set(c.get("sources", []))
            merged_concepts[idx]["sources"] = list(existing_sources | new_sources)
            # Update frequency
            merged_concepts[idx]["frequency"] = merged_concepts[idx].get("frequency", 1) + c.get("frequency", 1)
            # Merge contexts
            if "contexts" in c:
                merged_concepts[idx].setdefault("contexts", []).extend(c["contexts"])
    
    return merged_concepts


def process_concepts(concepts: List[Dict[str, Any]], doc_context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Process a list of concepts, applying transformations and filters.
    
    Args:
        concepts: List of concept dictionaries
        doc_context: Optional document context for quality scoring
        
    Returns:
        Processed list of concepts
    """
    # Default document context if none provided
    if doc_context is None:
        doc_context = {
            "title": "Advanced Graphics Processing and GPU Architecture",
            "abstract": "This paper explores modern GPU architectures and their applications in parallel processing, machine learning, and graphics rendering. We discuss the evolution of graphics processors and their impact on computational efficiency."
        }
    
    processed = concepts[:]
    
    # Apply context extraction if enabled
    if ENABLE_CONTEXT_EXTRACTION:
        print("[Config] Context extraction is ENABLED")
        # Add context extraction logic here
        for concept in processed:
            concept["context_extracted"] = True
    else:
        print("[Config] Context extraction is DISABLED")
    
    # Apply frequency tracking if enabled
    if ENABLE_FREQUENCY_TRACKING:
        print("[Config] Frequency tracking is ENABLED")
        # Add frequency tracking logic here
        frequency_threshold = cfg("frequency_threshold", 1)
        for concept in processed:
            concept["frequency"] = concept.get("frequency", 1)
            concept["above_threshold"] = concept["frequency"] >= frequency_threshold
    else:
        print("[Config] Frequency tracking is DISABLED")
    
    # Apply smart filtering if enabled
    if ENABLE_SMART_FILTERING:
        print("[Config] Smart filtering is ENABLED")
        # Filter based on configurable criteria
        min_relevance_score = cfg("min_relevance_score", 0.5)
        min_quality_score = cfg("min_quality_score", 0.3)
        max_concept_age_days = cfg("max_concept_age_days", 365)
        
        # Calculate quality scores using embedding centrality
        for concept in processed:
            # Calculate quality score with semantic centrality
            quality_score = calculate_concept_quality(concept, doc_context)
            concept["quality_score"] = round(quality_score, 3)
            
            # Keep existing relevance score or use quality as relevance
            if "relevance_score" not in concept:
                concept["relevance_score"] = quality_score
        
        # Filter concepts based on quality and relevance
        processed = [c for c in processed 
                    if c.get("relevance_score", 0) >= min_relevance_score 
                    and c.get("quality_score", 0) >= min_quality_score]
    else:
        print("[Config] Smart filtering is DISABLED")
    
    # Apply concept clustering if enabled
    if ENABLE_CONCEPT_CLUSTERING and len(processed) > 1:
        print("[Config] Concept clustering is ENABLED (using HDBSCAN)")
        clusters = cluster_similar_concepts(processed)
        
        # Mark concepts with their cluster information
        cluster_id = 0
        for cluster in clusters:
            if len(cluster) > 1:
                # Multi-concept cluster
                for concept in cluster:
                    concept["cluster_id"] = cluster_id
                    concept["cluster_size"] = len(cluster)
                    # Find cluster members
                    concept["cluster_members"] = [c["name"] for c in cluster if c["name"] != concept["name"]]
                cluster_id += 1
            else:
                # Single concept (noise or singleton)
                cluster[0]["cluster_id"] = -1
                cluster[0]["cluster_size"] = 1
    else:
        if not ENABLE_CONCEPT_CLUSTERING:
            print("[Config] Concept clustering is DISABLED")
    
    return processed


def calculate_concept_quality(concept: Dict, doc_context: Dict) -> float:
    """
    Calculate quality score for a concept based on various factors including semantic centrality.
    
    Args:
        concept: Dictionary containing concept information
        doc_context: Dictionary containing document context (title, abstract, etc.)
        
    Returns:
        Quality score between 0 and 1
    """
    # Get concept name
    concept_name = concept.get("name", "")
    
    # Base quality score
    quality = 0.5
    
    # Calculate theme relevance (placeholder - implement actual logic)
    theme_relevance = 0.6  # Example value
    
    # ---------- NEW: semantic centrality ----------
    try:
        from sentence_transformers import SentenceTransformer, util
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        doc_vec = embed_model.encode(doc_context.get("title", "") + " " +
                                     doc_context.get("abstract", ""))
        concept_vec = embed_model.encode(concept_name)
        sem_central = float(util.cos_sim(doc_vec, concept_vec))
    except:
        sem_central = 0.5   # fallback
    
    # Section weights for academic papers
    section_weight = concept.get("section_weight", 1.0)
    
    # Position weight (concepts appearing earlier might be more important)
    position_weight = concept.get("position_weight", 1.0)
    
    # Frequency weight
    frequency_weight = min(concept.get("frequency", 1) / 10.0, 1.0)
    
    # Combine all factors
    quality *= section_weight
    quality *= position_weight
    quality *= frequency_weight
    quality *= (0.8 + theme_relevance * 0.3 + sem_central * 0.3)
    
    # Ensure quality is between 0 and 1
    quality = max(0.0, min(1.0, quality))
    
    return quality


def safe_get(obj: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Safely get a value from a dictionary.
    
    Args:
        obj: Dictionary to get value from
        key: Key to retrieve
        default: Default value if key not found
        
    Returns:
        Value from dictionary or default
    """
    return obj.get(key, default) if isinstance(obj, dict) else default


def extract_and_boost_concepts(pure_concepts: List[Dict[str, Any]], pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract and boost concepts from PDF documents with provenance tracking.
    
    Args:
        pure_concepts: List of raw concept dictionaries
        pdf_path: Path to the PDF file being processed
        
    Returns:
        List of unique concepts with source provenance
    """
    seen = set()
    unique_pure = []
    
    for c in pure_concepts:
        name_lower = safe_get(c, 'name', '').lower().strip()
        if name_lower:
            if name_lower in seen:
                # append provenance to existing concept
                idx = next(i for i, v in enumerate(unique_pure) if v["name"].lower() == name_lower)
                unique_pure[idx].setdefault("sources", []).append(pdf_path)
                
                # Update frequency count
                unique_pure[idx]["frequency"] = unique_pure[idx].get("frequency", 1) + 1
                
                # Merge any additional metadata
                if "contexts" not in unique_pure[idx]:
                    unique_pure[idx]["contexts"] = []
                if "description" in c:
                    unique_pure[idx]["contexts"].append({
                        "source": pdf_path,
                        "description": c["description"]
                    })
            else:
                seen.add(name_lower)
                c["sources"] = [pdf_path]
                c["frequency"] = c.get("frequency", 1)
                if "description" in c:
                    c["contexts"] = [{
                        "source": pdf_path,
                        "description": c["description"]
                    }]
                unique_pure.append(c)
    
    # Log provenance tracking
    multi_source_concepts = [c for c in unique_pure if len(c.get("sources", [])) > 1]
    if multi_source_concepts:
        print(f"\n[Provenance] Found {len(multi_source_concepts)} concepts from multiple sources:")
        for concept in multi_source_concepts[:5]:  # Show first 5
            print(f"  - '{concept['name']}' appears in {len(concept['sources'])} documents")
            print(f"    Sources: {', '.join(concept['sources'][:3])}{'...' if len(concept['sources']) > 3 else ''}")
    
    return unique_pure


def cluster_similar_concepts(concepts: List[Dict], similarity_threshold: float = 0.8) -> List[List[Dict]]:
    """
    Cluster similar concepts using HDBSCAN density-based clustering.
    
    Args:
        concepts: List of concept dictionaries to cluster
        similarity_threshold: Not used in HDBSCAN, kept for API compatibility
        
    Returns:
        List of concept clusters (each cluster is a list of concepts)
    """
    if not concepts:
        return []
    
    try:
        from sentence_transformers import SentenceTransformer
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Encode concept names into embeddings
        embeddings = embed_model.encode([c["name"] for c in concepts])
        
        # Apply HDBSCAN clustering
        clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric="euclidean")
        labels = clusterer.fit_predict(embeddings)
        
        # Group concepts by cluster label
        clusters = {}
        for lbl, concept in zip(labels, concepts):
            clusters.setdefault(lbl, []).append(concept)
        
        # Convert to list of clusters
        # label -1 == noise → treat each as its own cluster
        cluster_list = []
        for label, cluster_concepts in clusters.items():
            if label == -1:
                # Noise points: each becomes its own cluster
                for concept in cluster_concepts:
                    cluster_list.append([concept])
            else:
                # Regular cluster
                cluster_list.append(cluster_concepts)
        
        # Log clustering results
        clustered_count = sum(1 for label in labels if label != -1)
        noise_count = sum(1 for label in labels if label == -1)
        print(f"\n[HDBSCAN] Clustered {len(concepts)} concepts:")
        print(f"  - {len(cluster_list)} total clusters")
        print(f"  - {clustered_count} concepts in clusters")
        print(f"  - {noise_count} noise points (unclustered)")
        
        # Show sample clusters
        multi_concept_clusters = [c for c in cluster_list if len(c) > 1]
        if multi_concept_clusters:
            print(f"\n[HDBSCAN] Found {len(multi_concept_clusters)} multi-concept clusters:")
            for i, cluster in enumerate(multi_concept_clusters[:3]):
                concept_names = [c['name'] for c in cluster]
                print(f"  Cluster {i+1}: {', '.join(concept_names)}")
        
        return cluster_list
        
    except ImportError:
        print("[Warning] sentence-transformers not available, returning unclustered concepts")
        # Fallback: each concept is its own cluster
        return [[c] for c in concepts]
    except Exception as e:
        print(f"[Error] Clustering failed: {e}, returning unclustered concepts")
        return [[c] for c in concepts]


async def store_concepts_async(concepts: List[Dict[str, Any]], user_id: str = "system", metadata: Dict[str, Any] = None) -> None:
    """
    Asynchronously store concepts using the memory sculptor.
    Processes concepts in batches for efficiency.
    
    Args:
        concepts: List of concepts to store
        user_id: User ID for attribution
        metadata: Additional metadata to attach
    """
    if not concepts:
        return
    
    try:
        from .memory_sculptor import batch_sculpt_and_store
        
        # Default metadata
        if metadata is None:
            metadata = {
                "source": "pipeline",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Process in batches of 100 for efficiency
        batch_size = 100
        for i in range(0, len(concepts), batch_size):
            batch = concepts[i:i + batch_size]
            
            # Run in executor to avoid blocking
            await asyncio.get_event_loop().run_in_executor(
                None,
                batch_sculpt_and_store,
                user_id,
                batch,
                metadata
            )
            
            print(f"[Memory Sculptor] Stored batch {i//batch_size + 1} ({len(batch)} concepts)")
        
        print(f"[Memory Sculptor] Completed storing {len(concepts)} concepts with embeddings")
        
    except ImportError:
        print("[Warning] Memory sculptor not available, concepts not persisted with embeddings")
    except Exception as e:
        print(f"[Error] Failed to store concepts: {e}")


def save_concepts(concepts: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Save processed concepts to a file.
    
    Args:
        concepts: List of concept dictionaries to save
        output_path: Path where to save the concepts
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(concepts, f, indent=2, ensure_ascii=False)


def main():
    """Main entry point for the pipeline."""
    print("\n=== Pipeline Starting ===")
    print(f"Config path: {_CFG_PATH}")
    print(f"Current configuration:")
    print(f"  - Context extraction: {ENABLE_CONTEXT_EXTRACTION}")
    print(f"  - Frequency tracking: {ENABLE_FREQUENCY_TRACKING}")
    print(f"  - Smart filtering: {ENABLE_SMART_FILTERING}")
    print(f"  - Concept clustering: {ENABLE_CONCEPT_CLUSTERING} (HDBSCAN)")
    print(f"  - Frequency threshold: {cfg('frequency_threshold', 1)}")
    print(f"  - Min relevance score: {cfg('min_relevance_score', 0.5)}")
    print(f"  - Min quality score: {cfg('min_quality_score', 0.3)}")
    print(f"  - Embedding model: all-MiniLM-L6-v2 (for semantic centrality)")
    print("\nNote: Config is hot-reloadable - edit the YAML file and changes will be picked up within 30 seconds\n")
    
    # Load and merge concepts with alias normalization
    merged_concepts = load_universal_concept_file_storage()
    print(f"\nLoaded {len(merged_concepts)} raw concepts")
    
    # Process the concepts
    processed_concepts = process_concepts(merged_concepts)
    
    # Print results for verification
    print(f"\n=== Results ===")
    print(f"Processed {len(processed_concepts)} concepts after filtering")
    for concept in processed_concepts:
        extras = []
        if "context_extracted" in concept:
            extras.append("context_extracted")
        if "frequency" in concept:
            extras.append(f"freq={concept['frequency']}")
        if "relevance_score" in concept:
            extras.append(f"relevance={concept['relevance_score']:.3f}")
        if "quality_score" in concept:
            extras.append(f"quality={concept['quality_score']}")
        if "sources" in concept and len(concept["sources"]) > 0:
            sources_str = ", ".join(concept["sources"][:3])
            if len(concept["sources"]) > 3:
                sources_str += f" +{len(concept['sources']) - 3} more"
            extras.append(f"sources=[{sources_str}]")
        if "cluster_id" in concept and concept["cluster_id"] >= 0:
            extras.append(f"cluster={concept['cluster_id']}")
            if concept.get("cluster_size", 1) > 1:
                members = concept.get("cluster_members", [])
                if members:
                    members_str = ", ".join(members[:2])
                    if len(members) > 2:
                        members_str += f" +{len(members) - 2}"
                    extras.append(f"with=[{members_str}]")
        extras_str = f" [{', '.join(extras)}]" if extras else ""
        print(f"  - {concept['name']} -> {concept['canonical']}{extras_str}")
    
    # Show provenance summary
    concepts_with_multiple_sources = [c for c in processed_concepts if len(c.get("sources", [])) > 1]
    if concepts_with_multiple_sources:
        print(f"\n=== Provenance Summary ===")
        print(f"Found {len(concepts_with_multiple_sources)} concepts appearing in multiple documents:")
        for concept in concepts_with_multiple_sources:
            print(f"  - '{concept['name']}' found in {len(concept['sources'])} documents")
            if "contexts" in concept and len(concept["contexts"]) > 0:
                print(f"    Sample context: \"{concept['contexts'][0]['description'][:60]}...\"")
    
    # Show clustering summary
    clustered_concepts = [c for c in processed_concepts if c.get("cluster_id", -1) >= 0]
    if clustered_concepts:
        unique_clusters = len(set(c["cluster_id"] for c in clustered_concepts))
        print(f"\n=== Clustering Summary (HDBSCAN) ===")
        print(f"Grouped {len(clustered_concepts)} concepts into {unique_clusters} clusters")
        
        # Show sample clusters
        clusters_dict = {}
        for c in clustered_concepts:
            clusters_dict.setdefault(c["cluster_id"], []).append(c["name"])
        
        for cluster_id, members in list(clusters_dict.items())[:3]:
            if len(members) > 1:
                print(f"  Cluster {cluster_id}: {', '.join(members)}")
    
    # Optionally save the results
    output_path = Path("processed_concepts.json")
    save_concepts(processed_concepts, output_path)
    print(f"\nSaved results to {output_path}")
    
    # Print embedding centrality notice
    if ENABLE_SMART_FILTERING and processed_concepts:
        print("\n[Note] Quality scores include semantic centrality calculated using embeddings.")
        print("Higher quality scores indicate stronger semantic relationship with the document context.")
    
    # Print provenance tracking notice
    print("\n[Note] Provenance tracking is enabled - concepts track their source documents.")
    print("Duplicate concepts are merged while preserving all source references.")


if __name__ == "__main__":
    # Run async version for testing
    asyncio.run(main_async())


async def main_async():
    """Async main entry point for the pipeline."""
    print("\n=== Pipeline Starting (Async) ===")
    print(f"Config path: {_CFG_PATH}")
    print(f"Current configuration:")
    print(f"  - Context extraction: {ENABLE_CONTEXT_EXTRACTION}")
    print(f"  - Frequency tracking: {ENABLE_FREQUENCY_TRACKING}")
    print(f"  - Smart filtering: {ENABLE_SMART_FILTERING}")
    print(f"  - Concept clustering: {ENABLE_CONCEPT_CLUSTERING} (HDBSCAN)")
    print(f"  - Frequency threshold: {cfg('frequency_threshold', 1)}")
    print(f"  - Min relevance score: {cfg('min_relevance_score', 0.5)}")
    print(f"  - Min quality score: {cfg('min_quality_score', 0.3)}")
    print(f"  - Embedding model: all-MiniLM-L6-v2 (for semantic centrality)")
    print("\nNote: Config is hot-reloadable - edit the YAML file and changes will be picked up within 30 seconds\n")
    
    # Load and merge concepts with alias normalization
    merged_concepts = load_universal_concept_file_storage()
    print(f"\nLoaded {len(merged_concepts)} raw concepts")
    
    # Process the concepts
    processed_concepts = process_concepts(merged_concepts)
    
    # Print results for verification
    print(f"\n=== Results ===")
    print(f"Processed {len(processed_concepts)} concepts after filtering")
    for concept in processed_concepts:
        extras = []
        if "context_extracted" in concept:
            extras.append("context_extracted")
        if "frequency" in concept:
            extras.append(f"freq={concept['frequency']}")
        if "relevance_score" in concept:
            extras.append(f"relevance={concept['relevance_score']:.3f}")
        if "quality_score" in concept:
            extras.append(f"quality={concept['quality_score']}")
        if "sources" in concept and len(concept["sources"]) > 0:
            sources_str = ", ".join(concept["sources"][:3])
            if len(concept["sources"]) > 3:
                sources_str += f" +{len(concept['sources']) - 3} more"
            extras.append(f"sources=[{sources_str}]")
        if "cluster_id" in concept and concept["cluster_id"] >= 0:
            extras.append(f"cluster={concept['cluster_id']}")
            if concept.get("cluster_size", 1) > 1:
                members = concept.get("cluster_members", [])
                if members:
                    members_str = ", ".join(members[:2])
                    if len(members) > 2:
                        members_str += f" +{len(members) - 2}"
                    extras.append(f"with=[{members_str}]")
        extras_str = f" [{', '.join(extras)}]" if extras else ""
        print(f"  - {concept['name']} -> {concept['canonical']}{extras_str}")
    
    # Show provenance summary
    concepts_with_multiple_sources = [c for c in processed_concepts if len(c.get("sources", [])) > 1]
    if concepts_with_multiple_sources:
        print(f"\n=== Provenance Summary ===")
        print(f"Found {len(concepts_with_multiple_sources)} concepts appearing in multiple documents:")
        for concept in concepts_with_multiple_sources:
            print(f"  - '{concept['name']}' found in {len(concept['sources'])} documents")
            if "contexts" in concept and len(concept["contexts"]) > 0:
                print(f"    Sample context: \"{concept['contexts'][0]['description'][:60]}...\"")
    
    # Show clustering summary
    clustered_concepts = [c for c in processed_concepts if c.get("cluster_id", -1) >= 0]
    if clustered_concepts:
        unique_clusters = len(set(c["cluster_id"] for c in clustered_concepts))
        print(f"\n=== Clustering Summary (HDBSCAN) ===")
        print(f"Grouped {len(clustered_concepts)} concepts into {unique_clusters} clusters")
        
        # Show sample clusters
        clusters_dict = {}
        for c in clustered_concepts:
            clusters_dict.setdefault(c["cluster_id"], []).append(c["name"])
        
        for cluster_id, members in list(clusters_dict.items())[:3]:
            if len(members) > 1:
                print(f"  Cluster {cluster_id}: {', '.join(members)}")
    
    # Store concepts with embeddings
    print("\n=== Storing Concepts with Embeddings ===")
    await store_concepts_async(processed_concepts)
    
    # Save to JSON as backup
    output_path = Path("processed_concepts.json")
    save_concepts(processed_concepts, output_path)
    print(f"\nSaved JSON backup to {output_path}")
    
    # Print embedding centrality notice
    if ENABLE_SMART_FILTERING and processed:
        print("\n[Note] Quality scores include semantic centrality calculated using embeddings.")
        print("Higher quality scores indicate stronger semantic relationship with the document context.")
    
    # Print provenance tracking notice
    print("\n[Note] Provenance tracking is enabled - concepts track their source documents.")
    print("Duplicate concepts are merged while preserving all source references.")
    print("\n[Note] Concepts are now stored with vector embeddings for similarity search.")
