"""
ingest_pdf/soliton_multi_tenant_manager.py

Manages Soliton Memory for multi-tenant PDF ingestion, ensuring each tenant's
data is stored in the Soliton Memory system rather than in a separate path.

ENHANCED VERSION 2.0 - Now with batch operations, advanced search, analytics, and cross-document linking!
"""

import os
import time
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import defaultdict
import json
import random
from functools import wraps

# Import Soliton client
try:
    from core.soliton_client import SolitonClient
    SOLITON_IMPORT_SOURCE = "core"
except ImportError:
    try:
        from mcp_metacognitive.core.soliton_memory import SolitonMemoryClient as SolitonClient
        SOLITON_IMPORT_SOURCE = "mcp_metacognitive"
    except ImportError:
        try:
            # Last resort - import from relative path
            from ..core.soliton_client import SolitonClient
            SOLITON_IMPORT_SOURCE = "relative"
        except ImportError:
            logging.error("❌ CRITICAL: Could not import SolitonClient from any known location")
            raise RuntimeError("SolitonClient not available - cannot initialize multi-tenant manager")

# Setup logger
logger = logging.getLogger("soliton_multi_tenant")

# Retry configuration
MAX_RETRIES = 5
BASE_DELAY = 1.0  # seconds
MAX_DELAY = 30.0  # seconds
JITTER_FACTOR = 0.1  # 10% jitter

def async_retry_with_backoff(max_retries: int = MAX_RETRIES):
    """Decorator for async functions with exponential backoff retry"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt == max_retries - 1:
                        logger.error(f"❌ {func.__name__} failed after {max_retries} attempts: {str(e)}")
                        raise
                    
                    # Calculate delay with exponential backoff
                    delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
                    # Add jitter to prevent thundering herd
                    jitter = delay * JITTER_FACTOR * (2 * random.random() - 1)
                    actual_delay = delay + jitter
                    
                    logger.warning(f"⚠️ {func.__name__} failed (attempt {attempt + 1}/{max_retries}), retrying in {actual_delay:.2f}s: {str(e)}")
                    await asyncio.sleep(actual_delay)
            
            # Should never reach here, but just in case
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator

class SolitonMultiTenantManager:
    """
    Manages Soliton Memory for multiple tenants (users/organizations),
    ensuring proper isolation and efficient storage of ingested PDF content.
    
    Enhanced with batch operations, analytics, and advanced search capabilities.
    """
    
    def __init__(self, api_url: Optional[str] = None, max_concurrent_ops: int = 10):
        """
        Initialize the multi-tenant manager
        
        Args:
            api_url: Optional URL to the Soliton API. If not provided,
                    will use the default from environment variables.
            max_concurrent_ops: Maximum concurrent operations for batch processing
        """
        self.api_url = api_url or os.environ.get("SOLITON_API_URL", "http://localhost:8002/api/soliton")
        self.client = SolitonClient(api_url=self.api_url)
        self.tenant_cache = {}  # Cache of tenant IDs that have been initialized
        self.max_concurrent_ops = max_concurrent_ops
        self.relationship_cache = {}  # Cache for concept relationships
        
        logger.info(f"🌊 SolitonMultiTenantManager initialized (imported from {SOLITON_IMPORT_SOURCE})")
        logger.info(f"🌊 Soliton API URL: {self.api_url}")
        logger.info(f"⚡ Max concurrent operations: {self.max_concurrent_ops}")
    
    @async_retry_with_backoff()
    async def initialize_tenant(self, tenant_id: str) -> bool:
        """
        Initialize a tenant's memory space in Soliton
        
        Args:
            tenant_id: ID of the tenant (user or organization)
            
        Returns:
            True if successful, False otherwise
        """
        if tenant_id in self.tenant_cache:
            # Already initialized
            return True
        
        try:
            # Initialize user memory in Soliton
            result = await self.client.initialize_user(tenant_id)
            
            if result:
                self.tenant_cache[tenant_id] = {
                    "initialized_at": datetime.now().isoformat(),
                    "status": "initialized"
                }
                logger.info(f"✅ Initialized Soliton Memory for tenant: {tenant_id}")
                return True
            else:
                logger.warning(f"⚠️ Failed to initialize Soliton Memory for tenant: {tenant_id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error initializing Soliton Memory for tenant {tenant_id}: {str(e)}")
            # Do NOT fall back to legacy storage - raise the error
            raise
    
    @async_retry_with_backoff()
    async def store_concept(self, 
                           tenant_id: str, 
                           concept_id: str, 
                           content: str,
                           metadata: Optional[Dict[str, Any]] = None,
                           tags: Optional[List[str]] = None,
                           strength: float = 0.7) -> bool:
        """
        Store a concept in a tenant's Soliton Memory space
        
        Args:
            tenant_id: ID of the tenant (user or organization)
            concept_id: Unique identifier for the concept
            content: Text content of the concept
            metadata: Optional metadata about the concept
            tags: Optional tags for categorization
            strength: Memory strength (0-1)
            
        Returns:
            True if successful, False otherwise
        """
        # Ensure tenant is initialized
        if tenant_id not in self.tenant_cache:
            await self.initialize_tenant(tenant_id)
        
        # Default tags and metadata
        tags = tags or ["pdf", "ingested"]
        metadata = metadata or {}
        
        # Ensure metadata has timestamp
        if "timestamp" not in metadata:
            metadata["timestamp"] = datetime.now().isoformat()
        
        # Add source information
        metadata["source"] = "pdf_ingestion"
        metadata["storage_method"] = "soliton_memory"
        
        # Create a unique memory ID if not specified in concept_id
        memory_id = concept_id
        if ":" not in memory_id:
            memory_id = f"pdf_concept:{concept_id}_{int(time.time())}"
        
        try:
            # Store in Soliton Memory
            result = await self.client.store_memory(
                user_id=tenant_id,
                memory_id=memory_id,
                content=content,
                strength=strength,
                tags=tags,
                metadata=metadata
            )
            
            if result:
                logger.info(f"✅ Stored concept {concept_id} in Soliton Memory for tenant {tenant_id}")
                return True
            else:
                logger.warning(f"⚠️ Failed to store concept {concept_id} for tenant {tenant_id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error storing concept {concept_id} for tenant {tenant_id}: {str(e)}")
            # Do NOT fall back to legacy storage - raise the error
            raise
    
    async def store_concepts_batch(self, 
                                  tenant_id: str, 
                                  concepts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Store multiple concepts efficiently using batch operations
        
        Args:
            tenant_id: ID of the tenant
            concepts: List of concept dictionaries with required fields:
                     - id: concept ID
                     - content: text content
                     - metadata (optional)
                     - tags (optional)
                     - strength (optional)
        
        Returns:
            Dictionary with results including success/failure counts
        """
        results = {
            'total': len(concepts),
            'successful': 0,
            'failed': 0,
            'errors': [],
            'concept_results': {}
        }
        
        # Initialize tenant once
        if tenant_id not in self.tenant_cache:
            await self.initialize_tenant(tenant_id)
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent_ops)
        
        async def store_with_semaphore(concept):
            async with semaphore:
                try:
                    success = await self.store_concept(
                        tenant_id=tenant_id,
                        concept_id=concept['id'],
                        content=concept['content'],
                        metadata=concept.get('metadata'),
                        tags=concept.get('tags'),
                        strength=concept.get('strength', 0.7)
                    )
                    return concept['id'], success, None
                except Exception as e:
                    return concept['id'], False, str(e)
        
        # Execute in parallel with limited concurrency
        tasks = [store_with_semaphore(concept) for concept in concepts]
        batch_results = await asyncio.gather(*tasks)
        
        # Process results
        for concept_id, success, error in batch_results:
            results['concept_results'][concept_id] = success
            if success:
                results['successful'] += 1
            else:
                results['failed'] += 1
                if error:
                    results['errors'].append({
                        'concept_id': concept_id,
                        'error': error
                    })
        
        logger.info(f"📦 Batch stored {results['successful']}/{results['total']} concepts for tenant {tenant_id}")
        return results
    
    async def search_concepts_advanced(self,
                                     tenant_id: str,
                                     query: str,
                                     filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Advanced search with multiple filters
        
        Args:
            tenant_id: ID of the tenant
            query: Search query
            filters: Optional filters including:
                    - date_from: ISO timestamp
                    - date_to: ISO timestamp
                    - min_score: minimum concept score
                    - sections: list of sections to filter by
                    - tags: additional tags to filter by
                    - limit: max results
                    - document_id: filter by specific document
        
        Returns:
            List of filtered concept memories
        """
        filters = filters or {}
        
        # Date range filter
        date_from = filters.get('date_from')
        date_to = filters.get('date_to')
        
        # Score threshold
        min_score = filters.get('min_score', 0.5)
        
        # Section filter (for academic papers)
        sections = filters.get('sections', [])
        
        # Document filter
        document_id = filters.get('document_id')
        
        # Tags filter
        search_tags = filters.get('tags', ['pdf'])
        
        # Get base results
        base_results = await self.find_related_concepts(
            tenant_id, query, 
            limit=filters.get('limit', 50),  # Get more initially for filtering
            min_strength=min_score,
            tags=search_tags
        )
        
        # Apply additional filters
        filtered = []
        for result in base_results:
            metadata = result.get('metadata', {})
            
            # Date filter
            if date_from and metadata.get('timestamp', '') < date_from:
                continue
            if date_to and metadata.get('timestamp', '') > date_to:
                continue
            
            # Section filter
            if sections and metadata.get('section') not in sections:
                continue
            
            # Document filter
            if document_id and metadata.get('document_id') != document_id:
                continue
            
            # Score filter (quality score or concept score)
            score = metadata.get('quality_score', metadata.get('concept_score', 0))
            if score < min_score:
                continue
            
            filtered.append(result)
        
        # Apply final limit
        final_limit = filters.get('limit', 10)
        filtered = filtered[:final_limit]
        
        logger.info(f"🔍 Advanced search found {len(filtered)} results for tenant {tenant_id}")
        return filtered
    
    async def get_tenant_analytics(self, tenant_id: str, days: int = 30) -> Dict[str, Any]:
        """
        Get detailed analytics for tenant's memory usage
        
        Args:
            tenant_id: ID of the tenant
            days: Number of days to analyze (default 30)
        
        Returns:
            Comprehensive analytics dictionary
        """
        stats = await self.get_tenant_stats(tenant_id)
        
        try:
            # Get all memories for time-based analysis
            all_memories = await self.client.get_all_memories(tenant_id)
            
            # Time-based grouping
            cutoff = datetime.now() - timedelta(days=days)
            daily_counts = defaultdict(int)
            topic_distribution = defaultdict(int)
            section_distribution = defaultdict(int)
            document_distribution = defaultdict(int)
            strength_distribution = []
            quality_scores = []
            
            # Analyze each memory
            for memory in all_memories:
                metadata = memory.get('metadata', {})
                
                # Time analysis
                timestamp = metadata.get('timestamp', '')
                if timestamp:
                    try:
                        date = datetime.fromisoformat(timestamp).date()
                        if date >= cutoff.date():
                            daily_counts[str(date)] += 1
                    except:
                        pass
                
                # Topic analysis
                tags = memory.get('tags', [])
                for tag in tags:
                    if tag.startswith('topic_'):
                        topic_distribution[tag] += 1
                
                # Section analysis
                section = metadata.get('section', 'unknown')
                section_distribution[section] += 1
                
                # Document analysis
                doc_id = metadata.get('document_id', 'unknown')
                document_distribution[doc_id] += 1
                
                # Strength analysis
                strength_distribution.append(memory.get('strength', 0))
                
                # Quality score analysis
                quality = metadata.get('quality_score', metadata.get('concept_score', 0))
                if quality > 0:
                    quality_scores.append(quality)
            
            # Calculate aggregates
            analytics = {
                **stats,
                'analytics': {
                    'period_days': days,
                    'daily_activity': dict(daily_counts),
                    'topic_distribution': dict(sorted(
                        topic_distribution.items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )[:20]),  # Top 20 topics
                    'section_distribution': dict(section_distribution),
                    'document_count': len(document_distribution),
                    'avg_strength': sum(strength_distribution) / len(strength_distribution) if strength_distribution else 0,
                    'avg_quality_score': sum(quality_scores) / len(quality_scores) if quality_scores else 0,
                    'total_memories_period': sum(daily_counts.values()),
                    'growth_rate': self._calculate_growth_rate(daily_counts),
                    'most_active_day': max(daily_counts.items(), key=lambda x: x[1])[0] if daily_counts else None
                }
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"❌ Error getting analytics for tenant {tenant_id}: {str(e)}")
            return {
                **stats,
                'analytics': {
                    'error': str(e)
                }
            }
    
    def _calculate_growth_rate(self, daily_counts: Dict[str, int]) -> float:
        """Calculate daily growth rate"""
        if len(daily_counts) < 2:
            return 0.0
        
        dates = sorted(daily_counts.keys())
        first_week = sum(daily_counts.get(d, 0) for d in dates[:7])
        last_week = sum(daily_counts.get(d, 0) for d in dates[-7:])
        
        if first_week == 0:
            return 100.0 if last_week > 0 else 0.0
        
        return ((last_week - first_week) / first_week) * 100
    
    async def link_concepts_across_documents(self, 
                                           tenant_id: str,
                                           similarity_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Find and link related concepts across different documents
        
        Args:
            tenant_id: ID of the tenant
            similarity_threshold: Minimum similarity score for linking
        
        Returns:
            Dictionary with linking results
        """
        results = {
            'documents_analyzed': 0,
            'relationships_found': 0,
            'cross_document_links': [],
            'processing_time': 0
        }
        
        start_time = time.time()
        
        try:
            # Get all documents for tenant
            all_docs = await self.get_documents_for_tenant(tenant_id)
            results['documents_analyzed'] = len(all_docs)
            
            # Process each document
            for doc in all_docs:
                doc_id = doc['id']
                
                # Get concepts from this document
                doc_concepts = await self.search_concepts_advanced(
                    tenant_id,
                    query="",  # Empty query to get all
                    filters={
                        'document_id': doc_id,
                        'limit': 100
                    }
                )
                
                # Find similar concepts in other documents
                for concept in doc_concepts:
                    if not concept.get('content'):
                        continue
                    
                    # Search for similar concepts
                    similar = await self.find_related_concepts(
                        tenant_id,
                        concept['content'],
                        limit=10,
                        min_strength=similarity_threshold,
                        tags=['pdf']
                    )
                    
                    # Create cross-document links
                    for similar_concept in similar:
                        similar_metadata = similar_concept.get('metadata', {})
                        similar_doc_id = similar_metadata.get('document_id')
                        
                        # Only link if from different document
                        if similar_doc_id and similar_doc_id != doc_id:
                            link = {
                                'source_concept': concept.get('id'),
                                'source_document': doc_id,
                                'target_concept': similar_concept.get('id'),
                                'target_document': similar_doc_id,
                                'similarity_score': similar_concept.get('strength', 0),
                                'relationship_type': 'cross_document_similarity'
                            }
                            
                            # Store the relationship
                            await self.store_concept_relationship(
                                tenant_id,
                                link['source_concept'],
                                link['target_concept'],
                                link['relationship_type'],
                                {'similarity_score': link['similarity_score']}
                            )
                            
                            results['cross_document_links'].append(link)
                            results['relationships_found'] += 1
            
            results['processing_time'] = time.time() - start_time
            
            logger.info(f"🔗 Found {results['relationships_found']} cross-document relationships for tenant {tenant_id}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error linking concepts for tenant {tenant_id}: {str(e)}")
            results['error'] = str(e)
            return results
    
    async def store_concept_relationship(self,
                                       tenant_id: str,
                                       source_concept_id: str,
                                       target_concept_id: str,
                                       relationship_type: str,
                                       metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store a relationship between two concepts
        
        Args:
            tenant_id: ID of the tenant
            source_concept_id: ID of the source concept
            target_concept_id: ID of the target concept
            relationship_type: Type of relationship
            metadata: Optional relationship metadata
        
        Returns:
            True if successful
        """
        # Create a hashed ID to avoid length issues (Soliton limit: 64 chars)
        from hashlib import sha1
        raw_id = f"{source_concept_id}:{target_concept_id}:{relationship_type}"
        edge_key = sha1(raw_id.encode()).hexdigest()[:20]  # 20-char digest
        relationship_id = f"rel:{edge_key}"
        content = f"Relationship: {source_concept_id} -> {relationship_type} -> {target_concept_id}"
        
        rel_metadata = {
            'source_concept': source_concept_id,
            'target_concept': target_concept_id,
            'relationship_type': relationship_type,
            'created_at': datetime.now().isoformat()
        }
        
        if metadata:
            rel_metadata.update(metadata)
        
        # Cache the relationship
        if tenant_id not in self.relationship_cache:
            self.relationship_cache[tenant_id] = []
        
        self.relationship_cache[tenant_id].append({
            'source': source_concept_id,
            'target': target_concept_id,
            'type': relationship_type
        })
        
        return await self.store_concept(
            tenant_id=tenant_id,
            concept_id=relationship_id,
            content=content,
            metadata=rel_metadata,
            tags=['relationship', relationship_type],
            strength=0.8
        )
    
    async def get_concept_relationships(self,
                                      tenant_id: str,
                                      concept_id: str) -> List[Dict[str, Any]]:
        """
        Get all relationships for a specific concept
        
        Args:
            tenant_id: ID of the tenant
            concept_id: ID of the concept
        
        Returns:
            List of relationships
        """
        # Search for relationships involving this concept
        rel_memories = await self.find_related_concepts(
            tenant_id,
            f"rel:{concept_id}",
            limit=50,
            tags=['relationship']
        )
        
        relationships = []
        for memory in rel_memories:
            metadata = memory.get('metadata', {})
            if metadata.get('source_concept') == concept_id or metadata.get('target_concept') == concept_id:
                relationships.append({
                    'id': memory.get('id'),
                    'source': metadata.get('source_concept'),
                    'target': metadata.get('target_concept'),
                    'type': metadata.get('relationship_type'),
                    'metadata': metadata
                })
        
        return relationships
    
    @async_retry_with_backoff(max_retries=3)  # Less critical, fewer retries
    async def get_tenant_stats(self, tenant_id: str) -> Dict[str, Any]:
        """
        Get statistics about a tenant's memory space
        
        Args:
            tenant_id: ID of the tenant
            
        Returns:
            Dictionary of statistics
        """
        try:
            # Get stats from Soliton
            stats = await self.client.get_user_stats(tenant_id)
            
            if not stats:
                logger.warning(f"⚠️ No stats available for tenant {tenant_id}")
                return {
                    "totalMemories": 0,
                    "activeWaves": 0,
                    "source": "soliton_memory",
                    "error": "No stats available"
                }
            
            # Add source information
            stats["source"] = "soliton_memory"
            
            # Add cached relationship count
            if tenant_id in self.relationship_cache:
                stats["cached_relationships"] = len(self.relationship_cache[tenant_id])
            
            return stats
                
        except Exception as e:
            logger.error(f"❌ Error getting stats for tenant {tenant_id}: {str(e)}")
            return {
                "totalMemories": 0,
                "activeWaves": 0,
                "source": "soliton_memory",
                "error": str(e)
            }
    
    @async_retry_with_backoff(max_retries=3)  # Read operation, fewer retries
    async def find_related_concepts(self, 
                                   tenant_id: str, 
                                   query: str,
                                   limit: int = 5,
                                   min_strength: float = 0.3,
                                   tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Find concepts related to a query in a tenant's memory space
        
        Args:
            tenant_id: ID of the tenant
            query: The query text to search for
            limit: Maximum number of results
            min_strength: Minimum memory strength to consider
            tags: Optional tags to filter by
            
        Returns:
            List of related memories
        """
        # Default tags - search only PDF concepts if not specified
        tags = tags or ["pdf"]
        
        try:
            # Find related memories in Soliton
            memories = await self.client.find_related_memories(
                user_id=tenant_id,
                query=query,
                limit=limit,
                min_strength=min_strength,
                tags=tags
            )
            
            if not memories:
                logger.info(f"No related concepts found for query '{query}' for tenant {tenant_id}")
                return []
            
            logger.info(f"✅ Found {len(memories)} related concepts for tenant {tenant_id}")
            return memories
                
        except Exception as e:
            logger.error(f"❌ Error finding related concepts for tenant {tenant_id}: {str(e)}")
            # Do NOT fall back to legacy storage - return empty list
            return []
    
    @async_retry_with_backoff()
    async def store_document_metadata(self,
                                     tenant_id: str,
                                     document_id: str,
                                     metadata: Dict[str, Any]) -> bool:
        """
        Store metadata about a document in a tenant's memory space
        
        Args:
            tenant_id: ID of the tenant
            document_id: ID of the document
            metadata: Document metadata
            
        Returns:
            True if successful, False otherwise
        """
        # Ensure tenant is initialized
        if tenant_id not in self.tenant_cache:
            await self.initialize_tenant(tenant_id)
        
        # Create a special memory for document metadata
        memory_id = f"doc_meta:{document_id}"
        content = f"Document metadata for {document_id}: {metadata.get('title', 'Untitled')}"
        
        # Add document-specific tags
        tags = ["document_metadata", "pdf"]
        
        try:
            # Store in Soliton Memory
            result = await self.client.store_memory(
                user_id=tenant_id,
                memory_id=memory_id,
                content=content,
                strength=0.9,  # High strength for metadata
                tags=tags,
                metadata=metadata
            )
            
            if result:
                logger.info(f"✅ Stored document metadata for {document_id} for tenant {tenant_id}")
                return True
            else:
                logger.warning(f"⚠️ Failed to store document metadata for {document_id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error storing document metadata for {document_id}: {str(e)}")
            # Do NOT fall back to legacy storage - raise the error
            raise
    
    async def get_documents_for_tenant(self, tenant_id: str) -> List[Dict[str, Any]]:
        """
        Get all documents for a tenant
        
        Args:
            tenant_id: ID of the tenant
            
        Returns:
            List of document metadata
        """
        try:
            # Find memories with document_metadata tag
            memories = await self.client.find_memories_by_tag(
                user_id=tenant_id,
                tag="document_metadata"
            )
            
            if not memories:
                logger.info(f"No documents found for tenant {tenant_id}")
                return []
            
            # Extract document metadata
            documents = []
            for memory in memories:
                if memory.get('metadata'):
                    documents.append({
                        "id": memory.get('id', '').replace('doc_meta:', ''),
                        **memory.get('metadata', {})
                    })
            
            logger.info(f"✅ Found {len(documents)} documents for tenant {tenant_id}")
            return documents
                
        except Exception as e:
            logger.error(f"❌ Error getting documents for tenant {tenant_id}: {str(e)}")
            return []
    
    def get_tenant_cache_info(self) -> Dict[str, Any]:
        """
        Get information about the tenant cache
        
        Returns:
            Dictionary with tenant cache information
        """
        return {
            "tenant_count": len(self.tenant_cache),
            "tenants": list(self.tenant_cache.keys()),
            "relationship_cache_sizes": {
                tenant: len(relationships) 
                for tenant, relationships in self.relationship_cache.items()
            }
        }

# Create singleton instance
soliton_manager = SolitonMultiTenantManager()

# Export for API usage
async def initialize_tenant(tenant_id: str) -> bool:
    """Initialize a tenant in Soliton Memory"""
    return await soliton_manager.initialize_tenant(tenant_id)

async def store_concept(tenant_id: str, 
                       concept_id: str, 
                       content: str,
                       metadata: Optional[Dict[str, Any]] = None,
                       tags: Optional[List[str]] = None,
                       strength: float = 0.7) -> bool:
    """Store a concept in a tenant's Soliton Memory"""
    return await soliton_manager.store_concept(
        tenant_id, concept_id, content, metadata, tags, strength
    )

async def store_concepts_batch(tenant_id: str, 
                             concepts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Store multiple concepts efficiently using batch operations"""
    return await soliton_manager.store_concepts_batch(tenant_id, concepts)

async def search_concepts_advanced(tenant_id: str,
                                 query: str,
                                 filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Advanced search with multiple filters"""
    return await soliton_manager.search_concepts_advanced(tenant_id, query, filters)

async def get_tenant_analytics(tenant_id: str, days: int = 30) -> Dict[str, Any]:
    """Get detailed analytics for tenant's memory usage"""
    return await soliton_manager.get_tenant_analytics(tenant_id, days)

async def link_concepts_across_documents(tenant_id: str,
                                       similarity_threshold: float = 0.7) -> Dict[str, Any]:
    """Find and link related concepts across different documents"""
    return await soliton_manager.link_concepts_across_documents(tenant_id, similarity_threshold)

async def get_tenant_stats(tenant_id: str) -> Dict[str, Any]:
    """Get statistics for a tenant's memory space"""
    return await soliton_manager.get_tenant_stats(tenant_id)

async def find_related_concepts(tenant_id: str, 
                               query: str,
                               limit: int = 5,
                               min_strength: float = 0.3,
                               tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Find concepts related to a query in a tenant's memory space"""
    return await soliton_manager.find_related_concepts(
        tenant_id, query, limit, min_strength, tags
    )

# Test function
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python soliton_multi_tenant_manager.py <tenant_id> [<concept_text>]")
        sys.exit(1)
    
    # Set up basic logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    async def test_manager():
        tenant_id = sys.argv[1]
        
        # Initialize tenant
        print(f"Initializing tenant {tenant_id}...")
        result = await soliton_manager.initialize_tenant(tenant_id)
        print(f"Initialization result: {result}")
        
        # Store a test concept if provided
        if len(sys.argv) > 2:
            concept_text = sys.argv[2]
            
            # Test batch storage
            concepts = [
                {
                    'id': f"test_{i}_{int(time.time())}",
                    'content': f"{concept_text} - variation {i}",
                    'metadata': {'index': i, 'test': True},
                    'tags': ['test', 'batch']
                }
                for i in range(5)
            ]
            
            print(f"Batch storing {len(concepts)} concepts for tenant {tenant_id}...")
            batch_results = await soliton_manager.store_concepts_batch(tenant_id, concepts)
            print(f"Batch results: {json.dumps(batch_results, indent=2)}")
        
        # Get tenant analytics
        print(f"Getting analytics for tenant {tenant_id}...")
        analytics = await soliton_manager.get_tenant_analytics(tenant_id, days=7)
        print(f"Analytics: {json.dumps(analytics, indent=2)}")
        
        # Test advanced search
        if len(sys.argv) > 2:
            print(f"Testing advanced search...")
            search_results = await soliton_manager.search_concepts_advanced(
                tenant_id,
                query=sys.argv[2],
                filters={
                    'min_score': 0.5,
                    'limit': 5
                }
            )
            print(f"Search results: {len(search_results)} found")
        
        # Show tenant cache info
        print(f"Tenant cache info: {soliton_manager.get_tenant_cache_info()}")
    
    asyncio.run(test_manager())
