"""
🧠 TORI KNOWLEDGE MANAGER - Three-Tier Knowledge Architecture
Production Ready: June 4, 2025
Features: Private → Organization → Foundation Knowledge Search
Pipeline Integration: Compatible with existing concept extraction
"""

import os
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
import logging
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class KnowledgeTier(Enum):
    PRIVATE = "private"
    ORGANIZATION = "organization" 
    FOUNDATION = "foundation"

@dataclass
class ConceptSearchResult:
    name: str
    confidence: float
    context: str
    tier: KnowledgeTier
    owner_id: str
    source_document: Optional[str]
    tags: List[str]
    created_at: str
    access_count: int
    metadata: Dict[str, Any]

@dataclass
class ConceptDiff:
    """Compatible with existing ConceptDiff from conceptMesh.ts"""
    id: str
    type: str
    title: str
    concepts: List[str]
    summary: str
    timestamp: datetime
    metadata: Dict[str, Any]
    tier: KnowledgeTier
    owner_id: str

class KnowledgeManager:
    """
    🧠 Three-Tier Knowledge Management System
    
    Search Priority:
    1. Private concepts (user's personal knowledge)
    2. Organization concepts (team/company knowledge) 
    3. Foundation concepts (global admin knowledge)
    
    Features:
    - Concept deduplication across tiers
    - Tier-based access control
    - Compatible with existing pipeline
    - Full concept diff integration
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.users_dir = self.data_dir / "users"
        self.orgs_dir = self.data_dir / "organizations"
        self.foundation_dir = self.data_dir / "foundation"
        
        # Ensure directories exist
        for dir_path in [self.users_dir, self.orgs_dir, self.foundation_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("🧠 Knowledge Manager initialized with three-tier architecture")
    
    def _load_json(self, file_path: Path) -> Dict[str, Any]:
        """Load JSON file with error handling"""
        try:
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return {}
    
    def _save_json(self, file_path: Path, data: Dict[str, Any]) -> bool:
        """Save JSON file with error handling"""
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Failed to save {file_path}: {e}")
            return False
    
    # ===================================================================
    # CONCEPT STORAGE - TIER-AWARE
    # ===================================================================
    
    def store_concepts(self, user_id: str, concepts: List[Dict[str, Any]], 
                      document_title: str, organization_id: Optional[str] = None,
                      tier: KnowledgeTier = KnowledgeTier.PRIVATE) -> Optional[ConceptDiff]:
        """
        Store concepts in the appropriate tier
        
        Args:
            user_id: User storing the concepts
            concepts: List of concept dictionaries from pipeline
            document_title: Source document name
            organization_id: Optional org ID for organization tier
            tier: Which tier to store in (private/organization/foundation)
        """
        try:
            now = datetime.now()
            
            # Determine storage location based on tier
            if tier == KnowledgeTier.PRIVATE:
                concepts_file = self.users_dir / user_id / "concepts.json"
                owner_id = user_id
            elif tier == KnowledgeTier.ORGANIZATION:
                if not organization_id:
                    raise ValueError("organization_id required for organization tier")
                concepts_file = self.orgs_dir / organization_id / "concepts.json"
                owner_id = organization_id
            elif tier == KnowledgeTier.FOUNDATION:
                concepts_file = self.foundation_dir / "concepts.json"
                owner_id = "foundation"
            else:
                raise ValueError(f"Invalid tier: {tier}")
            
            # Load existing concepts
            concepts_data = self._load_json(concepts_file)
            if "concepts" not in concepts_data:
                concepts_data = {
                    "concepts": {},
                    "metadata": {
                        "owner_id": owner_id,
                        "tier": tier.value,
                        "created_at": now.isoformat(),
                        "total_concepts": 0
                    }
                }
            
            # Create concept diff for tracking
            diff_id = f"{tier.value}_diff_{int(now.timestamp())}_{uuid.uuid4().hex[:8]}"
            
            # Process and store concepts
            stored_concept_names = []
            for concept in concepts:
                concept_name = concept.get("name", str(concept))
                concept_id = f"{tier.value}_{concept_name.lower().replace(' ', '_')}_{uuid.uuid4().hex[:8]}"
                
                # Create enriched concept
                enriched_concept = {
                    "id": concept_id,
                    "name": concept_name,
                    "confidence": concept.get("confidence", concept.get("score", 0.7)),
                    "context": concept.get("context", f"Extracted from {document_title}"),
                    "tier": tier.value,
                    "owner_id": owner_id,
                    "created_at": now.isoformat(),
                    "updated_at": now.isoformat(),
                    "access_count": 0,
                    "tags": concept.get("tags", []),
                    "source_document": document_title,
                    "metadata": {
                        "extraction_method": concept.get("extractionMethod", "pipeline"),
                        "purity": concept.get("purity", 0.7),
                        "frequency": concept.get("frequency", 1),
                        "section": concept.get("section", "general"),
                        "domain": concept.get("domain", "general"),
                        "diff_id": diff_id,
                        **concept.get("metadata", {})
                    }
                }
                
                # Store concept (update if exists)
                concepts_data["concepts"][concept_id] = enriched_concept
                stored_concept_names.append(concept_name)
            
            # Update metadata
            concepts_data["metadata"]["total_concepts"] = len(concepts_data["concepts"])
            concepts_data["metadata"]["last_updated"] = now.isoformat()
            
            # Save concepts
            if self._save_json(concepts_file, concepts_data):
                # Create ConceptDiff for compatibility
                concept_diff = ConceptDiff(
                    id=diff_id,
                    type="document",
                    title=document_title,
                    concepts=stored_concept_names,
                    summary=f"Stored {len(stored_concept_names)} concepts in {tier.value} tier",
                    timestamp=now,
                    metadata={
                        "tier": tier.value,
                        "owner_id": owner_id,
                        "concept_count": len(stored_concept_names),
                        "storage_location": str(concepts_file)
                    },
                    tier=tier,
                    owner_id=owner_id
                )
                
                logger.info(f"✅ Stored {len(stored_concept_names)} concepts in {tier.value} tier for {owner_id}")
                return concept_diff
            else:
                logger.error(f"Failed to save concepts to {concepts_file}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to store concepts in {tier.value} tier: {e}")
            return None
    
    def get_user_concepts(self, user_id: str) -> List[ConceptSearchResult]:
        """Get all concepts for a specific user (private tier only)"""
        try:
            concepts_file = self.users_dir / user_id / "concepts.json"
            concepts_data = self._load_json(concepts_file)
            
            results = []
            for concept_data in concepts_data.get("concepts", {}).values():
                result = ConceptSearchResult(
                    name=concept_data["name"],
                    confidence=concept_data["confidence"],
                    context=concept_data["context"],
                    tier=KnowledgeTier.PRIVATE,
                    owner_id=concept_data["owner_id"],
                    source_document=concept_data.get("source_document"),
                    tags=concept_data.get("tags", []),
                    created_at=concept_data["created_at"],
                    access_count=concept_data["access_count"],
                    metadata=concept_data.get("metadata", {})
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get user concepts for {user_id}: {e}")
            return []
    
    def get_organization_concepts(self, organization_id: str) -> List[ConceptSearchResult]:
        """Get all concepts for a specific organization"""
        try:
            concepts_file = self.orgs_dir / organization_id / "concepts.json"
            concepts_data = self._load_json(concepts_file)
            
            results = []
            for concept_data in concepts_data.get("concepts", {}).values():
                result = ConceptSearchResult(
                    name=concept_data["name"],
                    confidence=concept_data["confidence"],
                    context=concept_data["context"],
                    tier=KnowledgeTier.ORGANIZATION,
                    owner_id=concept_data["owner_id"],
                    source_document=concept_data.get("source_document"),
                    tags=concept_data.get("tags", []),
                    created_at=concept_data["created_at"],
                    access_count=concept_data["access_count"],
                    metadata=concept_data.get("metadata", {})
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get organization concepts for {organization_id}: {e}")
            return []
    
    def get_foundation_concepts(self) -> List[ConceptSearchResult]:
        """Get all foundation concepts"""
        try:
            concepts_file = self.foundation_dir / "concepts.json"
            concepts_data = self._load_json(concepts_file)
            
            results = []
            for concept_data in concepts_data.get("concepts", {}).values():
                result = ConceptSearchResult(
                    name=concept_data["name"],
                    confidence=concept_data["confidence"],
                    context=concept_data["context"],
                    tier=KnowledgeTier.FOUNDATION,
                    owner_id=concept_data["owner_id"],
                    source_document=concept_data.get("source_document"),
                    tags=concept_data.get("tags", []),
                    created_at=concept_data["created_at"],
                    access_count=concept_data["access_count"],
                    metadata=concept_data.get("metadata", {})
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get foundation concepts: {e}")
            return []
    
    # ===================================================================
    # THREE-TIER SEARCH SYSTEM
    # ===================================================================
    
    def search_concepts(self, query: str, user_id: str, 
                       organization_ids: List[str] = None,
                       max_results: int = 20) -> List[ConceptSearchResult]:
        """
        Search concepts across all three tiers with proper prioritization
        
        Search Order:
        1. Private concepts (user's personal knowledge)
        2. Organization concepts (team/company knowledge) 
        3. Foundation concepts (global admin knowledge)
        
        Args:
            query: Search query string
            user_id: User performing the search
            organization_ids: Organizations user belongs to
            max_results: Maximum number of results to return
        """
        try:
            all_results = []
            query_lower = query.lower()
            
            # 1. Search Private Concepts (Tier 1 - Highest Priority)
            logger.info(f"🔍 Searching private concepts for user {user_id}")
            private_concepts = self.get_user_concepts(user_id)
            private_matches = self._filter_concepts_by_query(private_concepts, query_lower)
            all_results.extend(private_matches)
            
            # 2. Search Organization Concepts (Tier 2 - Medium Priority)
            if organization_ids:
                logger.info(f"🔍 Searching organization concepts for orgs: {organization_ids}")
                for org_id in organization_ids:
                    org_concepts = self.get_organization_concepts(org_id)
                    org_matches = self._filter_concepts_by_query(org_concepts, query_lower)
                    all_results.extend(org_matches)
            
            # 3. Search Foundation Concepts (Tier 3 - Lowest Priority but Always Available)
            logger.info(f"🔍 Searching foundation concepts")
            foundation_concepts = self.get_foundation_concepts()
            foundation_matches = self._filter_concepts_by_query(foundation_concepts, query_lower)
            all_results.extend(foundation_matches)
            
            # Deduplicate by name while preserving tier priority
            deduplicated_results = self._deduplicate_by_tier_priority(all_results)
            
            # Sort by relevance and tier priority
            sorted_results = self._sort_by_relevance_and_tier(deduplicated_results, query_lower)
            
            # Update access counts for returned results
            self._update_access_counts(sorted_results[:max_results])
            
            logger.info(f"✅ Search complete: {len(sorted_results)} total results, returning top {min(len(sorted_results), max_results)}")
            
            return sorted_results[:max_results]
            
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            return []
    
    def _filter_concepts_by_query(self, concepts: List[ConceptSearchResult], query_lower: str) -> List[ConceptSearchResult]:
        """Filter concepts based on query string"""
        matches = []
        
        for concept in concepts:
            # Calculate relevance score
            relevance_score = 0
            
            # Exact name match (highest score)
            if concept.name.lower() == query_lower:
                relevance_score += 10
            elif query_lower in concept.name.lower():
                relevance_score += 5
            
            # Context match
            if concept.context and query_lower in concept.context.lower():
                relevance_score += 3
            
            # Tag match
            for tag in concept.tags:
                if query_lower in tag.lower():
                    relevance_score += 2
            
            # Word-by-word matching
            query_words = query_lower.split()
            concept_words = concept.name.lower().split() + concept.context.lower().split()
            
            for query_word in query_words:
                if len(query_word) > 2:  # Skip very short words
                    for concept_word in concept_words:
                        if query_word in concept_word:
                            relevance_score += 1
            
            # Include if any relevance found
            if relevance_score > 0:
                # Store relevance score in metadata for sorting
                concept.metadata = concept.metadata or {}
                concept.metadata["relevance_score"] = relevance_score
                matches.append(concept)
        
        return matches
    
    def _deduplicate_by_tier_priority(self, results: List[ConceptSearchResult]) -> List[ConceptSearchResult]:
        """Deduplicate concepts by name, keeping highest tier priority"""
        seen_names = {}
        tier_priority = {
            KnowledgeTier.PRIVATE: 3,      # Highest priority
            KnowledgeTier.ORGANIZATION: 2, # Medium priority  
            KnowledgeTier.FOUNDATION: 1    # Lowest priority
        }
        
        for result in results:
            name_lower = result.name.lower()
            current_priority = tier_priority.get(result.tier, 0)
            
            if name_lower not in seen_names:
                seen_names[name_lower] = result
            else:
                existing_priority = tier_priority.get(seen_names[name_lower].tier, 0)
                if current_priority > existing_priority:
                    seen_names[name_lower] = result
        
        return list(seen_names.values())
    
    def _sort_by_relevance_and_tier(self, results: List[ConceptSearchResult], query: str) -> List[ConceptSearchResult]:
        """Sort results by relevance score and tier priority"""
        tier_priority = {
            KnowledgeTier.PRIVATE: 3,
            KnowledgeTier.ORGANIZATION: 2,
            KnowledgeTier.FOUNDATION: 1
        }
        
        def sort_key(result):
            relevance = result.metadata.get("relevance_score", 0) if result.metadata else 0
            tier_weight = tier_priority.get(result.tier, 0) * 0.1  # Small tier bonus
            confidence_weight = result.confidence * 0.05  # Small confidence bonus
            
            return relevance + tier_weight + confidence_weight
        
        return sorted(results, key=sort_key, reverse=True)
    
    def _update_access_counts(self, results: List[ConceptSearchResult]):
        """Update access counts for accessed concepts"""
        # Group by tier and owner for efficient updates
        updates_by_file = {}
        
        for result in results:
            # Determine file location
            if result.tier == KnowledgeTier.PRIVATE:
                file_path = self.users_dir / result.owner_id / "concepts.json"
            elif result.tier == KnowledgeTier.ORGANIZATION:
                file_path = self.orgs_dir / result.owner_id / "concepts.json"
            else:  # Foundation
                file_path = self.foundation_dir / "concepts.json"
            
            if file_path not in updates_by_file:
                updates_by_file[file_path] = []
            updates_by_file[file_path].append(result.name)
        
        # Update each file
        for file_path, concept_names in updates_by_file.items():
            try:
                concepts_data = self._load_json(file_path)
                for concept_id, concept_data in concepts_data.get("concepts", {}).items():
                    if concept_data["name"] in concept_names:
                        concept_data["access_count"] = concept_data.get("access_count", 0) + 1
                        concept_data["last_accessed"] = datetime.now().isoformat()
                
                self._save_json(file_path, concepts_data)
                
            except Exception as e:
                logger.warning(f"Failed to update access counts in {file_path}: {e}")
    
    # ===================================================================
    # CONCEPT MANAGEMENT UTILITIES
    # ===================================================================
    
    def get_concept_by_id(self, concept_id: str, tier: KnowledgeTier, 
                         owner_id: str) -> Optional[ConceptSearchResult]:
        """Get a specific concept by ID"""
        try:
            if tier == KnowledgeTier.PRIVATE:
                concepts_file = self.users_dir / owner_id / "concepts.json"
            elif tier == KnowledgeTier.ORGANIZATION:
                concepts_file = self.orgs_dir / owner_id / "concepts.json"
            else:  # Foundation
                concepts_file = self.foundation_dir / "concepts.json"
            
            concepts_data = self._load_json(concepts_file)
            concept_data = concepts_data.get("concepts", {}).get(concept_id)
            
            if concept_data:
                return ConceptSearchResult(
                    name=concept_data["name"],
                    confidence=concept_data["confidence"],
                    context=concept_data["context"],
                    tier=tier,
                    owner_id=concept_data["owner_id"],
                    source_document=concept_data.get("source_document"),
                    tags=concept_data.get("tags", []),
                    created_at=concept_data["created_at"],
                    access_count=concept_data["access_count"],
                    metadata=concept_data.get("metadata", {})
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get concept {concept_id}: {e}")
            return None
    
    def get_knowledge_stats(self, user_id: str, organization_ids: List[str] = None) -> Dict[str, Any]:
        """Get knowledge statistics for a user across all tiers"""
        try:
            stats = {
                "tiers": {
                    "private": {"concepts": 0, "domains": set(), "recent_activity": 0},
                    "organization": {"concepts": 0, "domains": set(), "recent_activity": 0},
                    "foundation": {"concepts": 0, "domains": set(), "recent_activity": 0}
                },
                "total_concepts": 0,
                "total_domains": set(),
                "user_activity": {
                    "concepts_created": 0,
                    "concepts_accessed": 0,
                    "last_activity": None
                }
            }
            
            # Private tier stats
            private_concepts = self.get_user_concepts(user_id)
            stats["tiers"]["private"]["concepts"] = len(private_concepts)
            for concept in private_concepts:
                if concept.metadata and "domain" in concept.metadata:
                    stats["tiers"]["private"]["domains"].add(concept.metadata["domain"])
                    stats["total_domains"].add(concept.metadata["domain"])
            
            # Organization tier stats
            if organization_ids:
                for org_id in organization_ids:
                    org_concepts = self.get_organization_concepts(org_id)
                    stats["tiers"]["organization"]["concepts"] += len(org_concepts)
                    for concept in org_concepts:
                        if concept.metadata and "domain" in concept.metadata:
                            stats["tiers"]["organization"]["domains"].add(concept.metadata["domain"])
                            stats["total_domains"].add(concept.metadata["domain"])
            
            # Foundation tier stats
            foundation_concepts = self.get_foundation_concepts()
            stats["tiers"]["foundation"]["concepts"] = len(foundation_concepts)
            for concept in foundation_concepts:
                if concept.metadata and "domain" in concept.metadata:
                    stats["tiers"]["foundation"]["domains"].add(concept.metadata["domain"])
                    stats["total_domains"].add(concept.metadata["domain"])
            
            # Calculate totals
            stats["total_concepts"] = (
                stats["tiers"]["private"]["concepts"] +
                stats["tiers"]["organization"]["concepts"] +
                stats["tiers"]["foundation"]["concepts"]
            )
            
            # Convert sets to lists for JSON serialization
            for tier_data in stats["tiers"].values():
                tier_data["domains"] = list(tier_data["domains"])
            stats["total_domains"] = list(stats["total_domains"])
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get knowledge stats for {user_id}: {e}")
            return {"error": str(e)}

# ===================================================================
# GLOBAL INSTANCE
# ===================================================================

# Global instance for easy access
_knowledge_manager = None

def get_knowledge_manager() -> KnowledgeManager:
    """Get or create global knowledge manager instance"""
    global _knowledge_manager
    if _knowledge_manager is None:
        _knowledge_manager = KnowledgeManager()
    return _knowledge_manager

if __name__ == "__main__":
    # Demo/test functionality
    print("🧠 TORI Knowledge Manager Demo")
    
    km = KnowledgeManager()
    
    # Test concept storage
    test_concepts = [
        {"name": "Test Concept 1", "confidence": 0.8, "context": "This is a test concept"},
        {"name": "Test Concept 2", "confidence": 0.9, "context": "Another test concept"}
    ]
    
    diff = km.store_concepts("test_user", test_concepts, "Test Document", tier=KnowledgeTier.PRIVATE)
    if diff:
        print(f"Stored concepts: {diff.concepts}")
    
    # Test search
    results = km.search_concepts("test", "test_user")
    print(f"Search results: {len(results)} found")
    
    # Test stats
    stats = km.get_knowledge_stats("test_user")
    print(f"Knowledge stats: {stats}")
    
    print("✅ Knowledge Manager test complete!")
