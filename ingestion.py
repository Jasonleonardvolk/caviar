"""
Prajna Ingestion Orchestrator - PACKAGE-AWARE INTEGRATION
- Handles PDF upload (file or bytes)  
- Routes through package-aware integration (solves relative import issues)
- Calls injection.py for mesh/ledger injection
- Returns summary/result for dashboard

PHILOSOPHY:
- Use package-aware integration to handle relative imports properly
- Always try your 4000-hour system first
- Provide rich analytics when advanced system works
- Graceful fallback with enhanced extraction
"""

from utils.logging import logger
from injection import inject_concepts_into_mesh
from package_integration import extract_concepts_package_aware, get_package_stats

def ingest_document(pdf_file_path_or_bytes, user_id=None, doc_id=None):
    """
    Entry point: orchestrates full pipeline using package-aware integration.
    
    FLOW:
    1. Route through package-aware integration (handles relative imports)
    2. Get back sophisticated concept extraction with full analytics
    3. Inject into Prajna mesh
    4. Return comprehensive results with all advanced features
    
    Args:
        pdf_file_path_or_bytes: path or bytes of PDF
        user_id: who uploaded (optional, for provenance)
        doc_id: document identifier (optional)
    Returns:
        dict: comprehensive summary with advanced analytics
    """
    logger.info("🌉 PRAJNA INGESTION: Using package-aware integration")
    logger.info({"event": "ingestion_start", "doc_id": doc_id, "user": user_id})
    
    # Route through package-aware integration
    extraction_result = extract_concepts_package_aware(
        pdf_file_path_or_bytes, 
        user_id=user_id, 
        doc_id=doc_id
    )
    
    # Extract concepts and metadata from the result
    concepts = extraction_result.get("concepts", [])
    num_concepts = extraction_result.get("num_concepts", len(concepts))
    method = extraction_result.get("method", "unknown")
    status = extraction_result.get("status", "unknown")
    
    # Check if we successfully used the advanced system
    advanced_success = "package_aware_advanced_4000h" in method
    
    logger.info({
        "event": "extraction_complete", 
        "doc_id": doc_id, 
        "num_concepts": num_concepts,
        "method": method,
        "status": status,
        "advanced_system_used": advanced_success,
        "package_aware_integration": True
    })
    
    # Get advanced analytics if available
    advanced_analytics = extraction_result.get("advanced_analytics", {})
    summary = extraction_result.get("summary", {})
    
    # Log advanced system results with rich details
    if advanced_success:
        logger.info("🎉 SUCCESS: Used Jason's 4000-hour sophisticated system via package integration!")
        
        # Log rich analytics if available
        purity_analysis = advanced_analytics.get("purity_analysis", {})
        if purity_analysis:
            logger.info({
                "event": "advanced_purity_analysis",
                "purity_efficiency": purity_analysis.get("purity_efficiency", "N/A"),
                "pure_concepts": purity_analysis.get("pure_concepts", 0),
                "raw_concepts": purity_analysis.get("raw_concepts", 0),
                "consensus_concepts": purity_analysis.get("distribution", {}).get("consensus", 0),
                "high_confidence": purity_analysis.get("distribution", {}).get("high_confidence", 0)
            })
        
        context_extraction = advanced_analytics.get("context_extraction", {})
        if context_extraction:
            logger.info({
                "event": "advanced_context_analysis",
                "title_extracted": context_extraction.get("title_extracted", False),
                "abstract_extracted": context_extraction.get("abstract_extracted", False),
                "sections_identified": context_extraction.get("sections_identified", []),
                "avg_concept_frequency": context_extraction.get("avg_concept_frequency", 0)
            })
            
        # Log performance metrics
        processing_time = advanced_analytics.get("processing_time", 0)
        auto_prefilled = advanced_analytics.get("auto_prefilled_concepts", 0)
        semantic_concepts = advanced_analytics.get("semantic_concepts", 0)
        boosted_concepts = advanced_analytics.get("boosted_concepts", 0)
        
        logger.info({
            "event": "advanced_performance_metrics",
            "processing_time_seconds": processing_time,
            "auto_prefilled_concepts": auto_prefilled,
            "semantic_concepts": semantic_concepts,
            "boosted_concepts": boosted_concepts,
            "file_storage_size": advanced_analytics.get("file_storage_size", 0)
        })
        
    else:
        logger.info(f"🔄 Used fallback extraction: {method}")
        fallback_reason = summary.get("fallback_reason", "unknown")
        logger.info(f"   Fallback reason: {fallback_reason}")
    
    # Inject concepts into Prajna mesh
    inject_result = inject_concepts_into_mesh(concepts, user_id=user_id, doc_id=doc_id)
    
    logger.info({
        "event": "ingestion_complete", 
        "doc_id": doc_id, 
        "injected": inject_result.get("injected", 0),
        "advanced_system_used": advanced_success,
        "package_aware_integration": True
    })
    
    # Build comprehensive result preserving all advanced features
    result = {
        "doc_id": doc_id,
        "num_concepts": num_concepts,
        "concepts": concepts,
        "injection_result": inject_result,
        
        # Preserve ALL advanced analytics (full sophistication when available)
        "advanced_analytics": advanced_analytics,
        
        # Enhanced summary for dashboard
        "summary": {
            **summary,
            "advanced_system_used": advanced_success,
            "method": method,
            "status": status,
            "package_aware": True
        },
        
        # Integration metadata
        "integration_info": {
            "method": "package_aware_integration", 
            "extraction_method": method,
            "extraction_successful": status == "success",
            "advanced_pipeline_used": advanced_success,
            "fallback_used": not advanced_success,
            "package_integration": True,
            "relative_imports_handled": True
        }
    }
    
    # Log final comprehensive summary
    logger.info("🌉 PRAJNA INGESTION COMPLETE - Package-Aware Integration")
    logger.info(f"   📊 Total concepts: {num_concepts}")
    logger.info(f"   🔧 Method: {method}")
    logger.info(f"   🎯 Advanced system: {'✅ SUCCESS' if advanced_success else '🔄 FALLBACK'}")
    logger.info(f"   💉 Injected to mesh: {inject_result.get('injected', 0)}")
    logger.info(f"   ✅ Status: {status}")
    
    # Log advanced metrics if available
    if advanced_success and advanced_analytics:
        processing_time = advanced_analytics.get("processing_time", 0)
        auto_prefilled = advanced_analytics.get("auto_prefilled_concepts", 0)
        pure_concepts = summary.get("pure_concepts", 0)
        consensus_concepts = summary.get("consensus_concepts", 0)
        purity_efficiency = summary.get("purity_efficiency", "N/A")
        
        logger.info(f"   ⚡ Processing time: {processing_time:.1f}s")
        logger.info(f"   📥 Auto-prefilled: {auto_prefilled}")
        logger.info(f"   🏆 Pure concepts: {pure_concepts}")
        logger.info(f"   🤝 Consensus concepts: {consensus_concepts}")
        logger.info(f"   📊 Purity efficiency: {purity_efficiency}")
    
    return result

def get_ingestion_statistics():
    """
    Get comprehensive ingestion statistics from package integration.
    """
    package_stats = get_package_stats()
    
    return {
        "ingestion_stats": package_stats,
        "method": "package_aware_integration",
        "advanced_available": package_stats.get("advanced_available", False),
        "total_documents_processed": package_stats.get("total_documents_processed", 0),
        "total_concepts_extracted": package_stats.get("total_concepts_extracted", 0),
        "total_advanced_successes": package_stats.get("total_advanced_successes", 0),
        "total_fallback_uses": package_stats.get("total_fallback_uses", 0),
        "success_rate": package_stats.get("success_rate", 0),
        "fallback_rate": package_stats.get("fallback_rate", 0),
        "package_integration": True
    }

# Legacy function for backward compatibility
def ingest_document_legacy(pdf_file_path_or_bytes, user_id=None, doc_id=None):
    """
    Legacy function that routes to the package-aware integration system.
    """
    logger.info("🔄 Legacy ingestion function called - routing to package-aware integration")
    return ingest_document(pdf_file_path_or_bytes, user_id, doc_id)
