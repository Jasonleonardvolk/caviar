# TORI SYSTEM INTEGRATION GUIDE
## Enhanced Pipeline + Memory Sculptor + Multi-Tenant Manager

### 🚀 Complete System Overview

The TORI system consists of three powerful components that work together to create an intelligent PDF knowledge extraction and storage system:

1. **Enhanced Pipeline** (`pipeline.py`) - Extracts concepts with OCR, quality scoring, and academic structure detection
2. **Memory Sculptor** (`memory_sculptor.py`) - Enriches concepts with NLP, detects relationships, and prepares for storage
3. **Multi-Tenant Manager** (`soliton_multi_tenant_manager.py`) - Orchestrates storage, search, and analytics across tenants

### 📋 Integration Example

```python
import asyncio
from pipeline import ingest_pdf_clean
from memory_sculptor import sculpt_and_store_batch
from soliton_multi_tenant_manager import (
    initialize_tenant, 
    get_tenant_analytics,
    link_concepts_across_documents,
    search_concepts_advanced
)

async def process_pdf_for_tenant(pdf_path: str, tenant_id: str, document_id: str):
    """
    Complete PDF processing pipeline with all enhancements
    """
    # Step 1: Extract concepts using enhanced pipeline
    print(f"📄 Extracting concepts from {pdf_path}...")
    extraction_result = ingest_pdf_clean(
        pdf_path=pdf_path,
        doc_id=document_id,
        admin_mode=False,
        use_ocr=True  # Enable OCR for scanned PDFs
    )
    
    if extraction_result['status'] != 'success':
        print(f"❌ Extraction failed: {extraction_result.get('error_message', 'Unknown error')}")
        return
    
    print(f"✅ Extracted {extraction_result['concept_count']} concepts")
    print(f"📊 Section distribution: {extraction_result['section_distribution']}")
    
    # Step 2: Initialize tenant
    await initialize_tenant(tenant_id)
    
    # Step 3: Prepare document metadata
    doc_metadata = {
        'document_id': document_id,
        'filename': extraction_result['filename'],
        'title': extraction_result.get('title_found', 'Untitled'),
        'extraction_date': extraction_result['processing_time_seconds'],
        'concept_count': extraction_result['concept_count'],
        'section_distribution': extraction_result['section_distribution'],
        'average_concept_score': extraction_result['average_concept_score'],
        'high_quality_concepts': extraction_result['high_quality_concepts']
    }
    
    # Step 4: Sculpt and store concepts with relationship detection
    print(f"🎨 Sculpting {len(extraction_result['concepts'])} concepts...")
    sculpt_results = await sculpt_and_store_batch(
        user_id=tenant_id,
        concepts=extraction_result['concepts'],
        doc_metadata=doc_metadata
    )
    
    print(f"✅ Created {len(sculpt_results['memories_created'])} memories")
    print(f"🔗 Detected {len(sculpt_results['relationships_detected'])} concept relationships")
    
    # Step 5: Link concepts across documents
    print(f"🔗 Linking concepts across documents...")
    link_results = await link_concepts_across_documents(
        tenant_id=tenant_id,
        similarity_threshold=0.7
    )
    
    print(f"✅ Found {link_results['relationships_found']} cross-document relationships")
    
    # Step 6: Get analytics
    analytics = await get_tenant_analytics(tenant_id, days=30)
    print(f"\n📊 Tenant Analytics:")
    print(f"  - Total memories: {analytics.get('totalMemories', 0)}")
    print(f"  - Documents: {analytics['analytics'].get('document_count', 0)}")
    print(f"  - Avg quality score: {analytics['analytics'].get('avg_quality_score', 0):.3f}")
    print(f"  - Growth rate: {analytics['analytics'].get('growth_rate', 0):.1f}%")
    
    return {
        'extraction': extraction_result,
        'sculpting': sculpt_results,
        'linking': link_results,
        'analytics': analytics
    }

async def search_tenant_knowledge(tenant_id: str, query: str, filters: dict = None):
    """
    Advanced search across tenant's knowledge base
    """
    # Default filters
    if filters is None:
        filters = {
            'min_score': 0.6,
            'limit': 20,
            'sections': ['abstract', 'introduction', 'conclusion'],  # Focus on key sections
        }
    
    # Perform advanced search
    results = await search_concepts_advanced(
        tenant_id=tenant_id,
        query=query,
        filters=filters
    )
    
    print(f"\n🔍 Search Results for '{query}':")
    print(f"Found {len(results)} relevant concepts")
    
    for i, result in enumerate(results[:5]):  # Show top 5
        metadata = result.get('metadata', {})
        print(f"\n{i+1}. {result.get('content', '')[:100]}...")
        print(f"   Score: {result.get('strength', 0):.3f}")
        print(f"   Quality: {metadata.get('quality_score', 0):.3f}")
        print(f"   Section: {metadata.get('section', 'unknown')}")
        print(f"   Document: {metadata.get('document_id', 'unknown')}")
        
        # Show entities if available
        entities = metadata.get('entities', [])
        if entities:
            entity_types = set(e['type'] for e in entities)
            print(f"   Entities: {', '.join(entity_types)}")
    
    return results

# Example usage
async def main():
    tenant_id = "research_lab_001"
    
    # Process multiple PDFs
    pdfs = [
        ("paper1.pdf", "doc_001"),
        ("paper2.pdf", "doc_002"),
        ("paper3.pdf", "doc_003")
    ]
    
    for pdf_path, doc_id in pdfs:
        print(f"\n{'='*60}")
        print(f"Processing {pdf_path} for tenant {tenant_id}")
        print(f"{'='*60}")
        
        try:
            await process_pdf_for_tenant(pdf_path, tenant_id, doc_id)
        except Exception as e:
            print(f"❌ Error processing {pdf_path}: {str(e)}")
    
    # Search across all documents
    print(f"\n{'='*60}")
    print(f"KNOWLEDGE SEARCH")
    print(f"{'='*60}")
    
    await search_tenant_knowledge(
        tenant_id=tenant_id,
        query="machine learning neural networks",
        filters={
            'min_score': 0.7,
            'limit': 10,
            'sections': ['methodology', 'results']
        }
    )

if __name__ == "__main__":
    asyncio.run(main())
```

### 🔧 Key Integration Points

#### 1. **Pipeline → Memory Sculptor**
```python
# Pipeline extracts concepts with rich metadata
concepts = extraction_result['concepts']
# Each concept includes:
# - name, score, quality_score
# - metadata with section, frequency, in_title, in_abstract
# - method (how it was extracted)

# Memory sculptor enriches and segments
memories = await sculpt_and_store_batch(
    user_id=tenant_id,
    concepts=concepts,
    doc_metadata=doc_metadata
)
```

#### 2. **Memory Sculptor → Multi-Tenant Manager**
```python
# Memory sculptor creates enriched memories
# Multi-tenant manager handles storage and retrieval
# Automatic relationship detection between concepts
```

#### 3. **Cross-Component Features**

**Quality-Based Storage:**
- Pipeline calculates quality scores
- Memory sculptor uses quality for strength calculation
- Multi-tenant manager enables quality-based search

**Relationship Mapping:**
- Pipeline groups related concepts
- Memory sculptor detects co-occurrence
- Multi-tenant manager links across documents

**Section Awareness:**
- Pipeline detects academic sections
- Memory sculptor preserves section metadata
- Multi-tenant manager enables section-based filtering

### 📊 Analytics Dashboard Example

```python
async def generate_tenant_dashboard(tenant_id: str):
    """Generate a comprehensive analytics dashboard"""
    
    analytics = await get_tenant_analytics(tenant_id, days=30)
    
    print(f"\n📊 TENANT DASHBOARD: {tenant_id}")
    print(f"{'='*50}")
    
    # Memory stats
    print(f"\n💾 MEMORY STATISTICS")
    print(f"Total Memories: {analytics['totalMemories']}")
    print(f"Active Waves: {analytics['activeWaves']}")
    
    # Document stats
    print(f"\n📄 DOCUMENT STATISTICS")
    print(f"Total Documents: {analytics['analytics']['document_count']}")
    print(f"Memories This Period: {analytics['analytics']['total_memories_period']}")
    
    # Quality metrics
    print(f"\n⭐ QUALITY METRICS")
    print(f"Avg Concept Strength: {analytics['analytics']['avg_strength']:.3f}")
    print(f"Avg Quality Score: {analytics['analytics']['avg_quality_score']:.3f}")
    
    # Growth metrics
    print(f"\n📈 GROWTH METRICS")
    print(f"Growth Rate: {analytics['analytics']['growth_rate']:.1f}%")
    print(f"Most Active Day: {analytics['analytics']['most_active_day']}")
    
    # Topic distribution
    print(f"\n🏷️ TOP TOPICS")
    topics = analytics['analytics']['topic_distribution']
    for topic, count in list(topics.items())[:5]:
        print(f"  {topic}: {count} memories")
    
    # Section distribution
    print(f"\n📑 SECTION DISTRIBUTION")
    sections = analytics['analytics']['section_distribution']
    for section, count in sections.items():
        print(f"  {section}: {count} concepts")
```

### 🚀 Performance Tips

1. **Batch Processing**
   - Use `store_concepts_batch()` for multiple concepts
   - Process PDFs in parallel when possible
   - Set appropriate `max_concurrent_ops` for your system

2. **Caching**
   - Tenant initialization is cached
   - Relationship cache reduces repeated searches
   - Consider implementing result caching for frequent queries

3. **Filtering**
   - Use section filters to focus on relevant parts
   - Set appropriate quality/score thresholds
   - Limit date ranges for time-based queries

### 🔒 Best Practices

1. **Tenant Isolation**
   - Each tenant's data is completely isolated
   - No cross-tenant data leakage
   - Proper error handling prevents fallback to shared storage

2. **Error Handling**
   - All components raise errors instead of silent failures
   - Comprehensive logging at each step
   - Batch operations track individual failures

3. **Metadata Preservation**
   - Document metadata flows through all components
   - Section information preserved for academic use
   - Quality scores enable intelligent retrieval

### 🎯 Use Cases

1. **Research Lab Knowledge Base**
   - Process hundreds of papers per tenant
   - Cross-reference findings across documents
   - Track research trends over time

2. **Academic Department Archive**
   - Store student papers and theses
   - Find similar work across years
   - Generate department-wide analytics

3. **Corporate R&D Repository**
   - Extract insights from technical documents
   - Link related concepts across projects
   - Track knowledge evolution

The enhanced TORI system provides a complete solution for intelligent PDF processing, from extraction through storage to advanced retrieval and analytics!
