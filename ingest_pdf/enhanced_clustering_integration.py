"""
TORI Enhanced Clustering - Drop-in Replacement
==============================================

This module provides a drop-in replacement for existing TORI clustering functions
with enhanced capabilities including benchmarking, monitoring, and quality assessment.

QUICK INTEGRATION:
1. Import this module: from enhanced_clustering_integration import enhanced_cluster_concepts
2. Replace your existing clustering calls
3. Enjoy improved clustering with detailed metrics

COMPATIBILITY:
- Maintains backward compatibility with existing cluster label outputs
- Adds optional enhanced features when requested
- Graceful fallback to original clustering if enhanced features fail
"""

import os
import sys
import warnings
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from clustering import run_oscillator_clustering, run_oscillator_clustering_with_metrics
    from clustering_pipeline import TORIClusteringPipeline, ConceptData, PipelineResult
    from clustering_config import ConfigManager
    from clustering_monitor import create_production_monitor
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Enhanced clustering features not available: {e}")
    ENHANCED_FEATURES_AVAILABLE = False

def enhanced_cluster_concepts(embeddings: np.ndarray, 
                            texts: Optional[List[str]] = None,
                            concept_ids: Optional[List[str]] = None,
                            metadata: Optional[List[Dict[str, Any]]] = None,
                            method: str = 'oscillator',
                            return_enhanced_results: bool = False,
                            enable_monitoring: bool = False,
                            enable_benchmarking: bool = False,
                            **clustering_kwargs) -> Union[List[int], PipelineResult]:
    """
    Enhanced clustering function that serves as a drop-in replacement.
    
    Args:
        embeddings: Concept embeddings (N x D numpy array)
        texts: Optional concept texts (for enhanced results)
        concept_ids: Optional concept IDs (auto-generated if not provided)
        metadata: Optional metadata for each concept
        method: Clustering method ('oscillator', 'kmeans', 'hdbscan', 'auto')
        return_enhanced_results: If True, returns full PipelineResult; if False, returns labels only
        enable_monitoring: Enable performance monitoring
        enable_benchmarking: Run method comparison
        **clustering_kwargs: Additional clustering parameters
        
    Returns:
        List[int] of cluster labels (compatibility mode) OR PipelineResult (enhanced mode)
    """
    
    # Input validation
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.array(embeddings)
    
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array (n_concepts x n_features)")
    
    n_concepts = len(embeddings)
    
    # Generate default values for optional parameters
    if concept_ids is None:
        concept_ids = [f"concept_{i}" for i in range(n_concepts)]
    
    if texts is None:
        texts = [f"Concept {i}" for i in range(n_concepts)]
    
    if metadata is None:
        metadata = [{} for _ in range(n_concepts)]
    
    # Enhanced clustering path
    if ENHANCED_FEATURES_AVAILABLE and return_enhanced_results:
        try:
            return _run_enhanced_clustering(
                embeddings, texts, concept_ids, metadata, method,
                enable_monitoring, enable_benchmarking, clustering_kwargs
            )
        except Exception as e:
            warnings.warn(f"Enhanced clustering failed, falling back to basic: {e}")
    
    # Basic clustering path (compatibility mode)
    return _run_basic_clustering(embeddings, method, clustering_kwargs)

def _run_enhanced_clustering(embeddings: np.ndarray,
                           texts: List[str],
                           concept_ids: List[str], 
                           metadata: List[Dict[str, Any]],
                           method: str,
                           enable_monitoring: bool,
                           enable_benchmarking: bool,
                           clustering_kwargs: Dict[str, Any]) -> PipelineResult:
    """Run enhanced clustering with full pipeline features."""
    
    # Create concept data
    concepts = []
    for i in range(len(embeddings)):
        concept = ConceptData(
            concept_id=concept_ids[i],
            text=texts[i],
            embedding=embeddings[i].tolist(),
            metadata=metadata[i],
            timestamp=None,
            confidence=metadata[i].get('confidence'),
            source_document=metadata[i].get('source_document'),
            page_number=metadata[i].get('page_number')
        )
        concepts.append(concept)
    
    # Create pipeline with monitoring if requested
    pipeline = TORIClusteringPipeline(enable_monitoring=enable_monitoring)
    
    # Run clustering
    result = pipeline.process_concepts(
        concepts=concepts,
        method=method if method != 'auto' else None,
        enable_benchmarking=enable_benchmarking
    )
    
    return result

def _run_basic_clustering(embeddings: np.ndarray, 
                        method: str, 
                        clustering_kwargs: Dict[str, Any]) -> List[int]:
    """Run basic clustering for backward compatibility."""
    
    if method == 'oscillator' or method == 'auto':
        # Use enhanced oscillator if available, otherwise fallback
        if ENHANCED_FEATURES_AVAILABLE:
            try:
                result = run_oscillator_clustering_with_metrics(
                    embeddings, 
                    enable_logging=False,
                    **clustering_kwargs
                )
                return result['labels']
            except:
                pass
        
        # Final fallback - try to use original clustering if available
        try:
            return run_oscillator_clustering(embeddings, **clustering_kwargs)
        except Exception as e:
            warnings.warn(f"Oscillator clustering failed: {e}")
            return _fallback_clustering(embeddings)
    
    elif method == 'kmeans':
        return _kmeans_fallback(embeddings, clustering_kwargs)
    
    else:
        warnings.warn(f"Unknown method {method}, using oscillator")
        return _run_basic_clustering(embeddings, 'oscillator', clustering_kwargs)

def _fallback_clustering(embeddings: np.ndarray) -> List[int]:
    """Ultra-simple fallback clustering when everything else fails."""
    try:
        from sklearn.cluster import KMeans
        k = max(2, min(10, int(np.sqrt(len(embeddings) / 2))))
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        return labels.tolist()
    except ImportError:
        # Ultimate fallback - simple distance-based clustering
        return _distance_based_clustering(embeddings)

def _distance_based_clustering(embeddings: np.ndarray, threshold: float = 0.5) -> List[int]:
    """Simple distance-based clustering as ultimate fallback."""
    n = len(embeddings)
    labels = [-1] * n
    cluster_id = 0
    
    for i in range(n):
        if labels[i] == -1:  # Unassigned
            labels[i] = cluster_id
            
            # Find similar concepts
            for j in range(i + 1, n):
                if labels[j] == -1:
                    # Cosine similarity
                    sim = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-8
                    )
                    if sim > threshold:
                        labels[j] = cluster_id
            
            cluster_id += 1
    
    return labels

def _kmeans_fallback(embeddings: np.ndarray, kwargs: Dict[str, Any]) -> List[int]:
    """K-means fallback implementation."""
    try:
        from sklearn.cluster import KMeans
        k = kwargs.get('k', max(2, min(10, int(np.sqrt(len(embeddings) / 2)))))
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        return labels.tolist()
    except ImportError:
        return _fallback_clustering(embeddings)

# Convenience functions for common use cases

def quick_cluster(embeddings: np.ndarray, method: str = 'oscillator') -> List[int]:
    """Quick clustering with minimal setup - direct replacement for existing functions."""
    return enhanced_cluster_concepts(embeddings, method=method, return_enhanced_results=False)

def advanced_cluster(embeddings: np.ndarray, 
                   texts: List[str],
                   method: str = 'auto',
                   enable_monitoring: bool = True) -> PipelineResult:
    """Advanced clustering with full enhanced features."""
    return enhanced_cluster_concepts(
        embeddings=embeddings,
        texts=texts,
        method=method,
        return_enhanced_results=True,
        enable_monitoring=enable_monitoring,
        enable_benchmarking=True
    )

def benchmark_clustering_methods(embeddings: np.ndarray, 
                               texts: Optional[List[str]] = None) -> Dict[str, Any]:
    """Benchmark different clustering methods on your data."""
    if not ENHANCED_FEATURES_AVAILABLE:
        raise RuntimeError("Enhanced features not available for benchmarking")
    
    result = enhanced_cluster_concepts(
        embeddings=embeddings,
        texts=texts,
        method='auto',
        return_enhanced_results=True,
        enable_benchmarking=True
    )
    
    return {
        'method_used': result.pipeline_metadata.get('method_used'),
        'quality_metrics': result.quality_assessment,
        'performance_metrics': result.performance_metrics,
        'cluster_summary': result.cluster_summary
    }

# Integration helpers for existing TORI pipeline

def integrate_with_existing_function(original_function):
    """
    Decorator to enhance existing clustering functions.
    
    Usage:
        @integrate_with_existing_function
        def my_clustering_function(embeddings):
            return some_clustering_algorithm(embeddings)
    """
    def wrapper(*args, **kwargs):
        # Try enhanced clustering first
        try:
            if len(args) > 0 and isinstance(args[0], np.ndarray):
                embeddings = args[0]
                return enhanced_cluster_concepts(embeddings, **kwargs)
        except Exception as e:
            warnings.warn(f"Enhanced clustering failed: {e}")
        
        # Fall back to original function
        return original_function(*args, **kwargs)
    
    return wrapper

def replace_clustering_calls_in_module(module, function_names: List[str]):
    """
    Replace clustering function calls in an existing module.
    
    Args:
        module: The module to modify
        function_names: List of function names to replace
    """
    for func_name in function_names:
        if hasattr(module, func_name):
            original_func = getattr(module, func_name)
            enhanced_func = integrate_with_existing_function(original_func)
            setattr(module, func_name, enhanced_func)
            print(f"✅ Enhanced {func_name} in {module.__name__}")

# Example usage and migration guide
if __name__ == "__main__":
    print("🔗 TORI Enhanced Clustering - Integration Test")
    print("=============================================")
    
    # Generate sample data
    np.random.seed(42)
    embeddings = np.random.randn(50, 100)
    texts = [f"Sample concept {i} about various topics" for i in range(50)]
    
    print(f"📊 Test data: {embeddings.shape[0]} concepts, {embeddings.shape[1]} dimensions")
    
    # Test 1: Drop-in replacement (compatibility mode)
    print("\n🔄 Test 1: Drop-in replacement mode")
    labels_basic = enhanced_cluster_concepts(embeddings, return_enhanced_results=False)
    print(f"   Result: {len(set(labels_basic))} clusters")
    print(f"   Type: {type(labels_basic)}")
    
    # Test 2: Enhanced mode
    print("\n🚀 Test 2: Enhanced mode")
    if ENHANCED_FEATURES_AVAILABLE:
        result_enhanced = enhanced_cluster_concepts(
            embeddings, texts, 
            return_enhanced_results=True,
            enable_monitoring=True
        )
        print(f"   Result: {len(result_enhanced.cluster_summary)} clusters")
        print(f"   Quality: {result_enhanced.quality_assessment['overall_cohesion']:.3f}")
        print(f"   Method: {result_enhanced.pipeline_metadata['method_used']}")
    else:
        print("   Enhanced features not available")
    
    # Test 3: Benchmarking
    print("\n📈 Test 3: Benchmarking")
    if ENHANCED_FEATURES_AVAILABLE:
        try:
            benchmark_results = benchmark_clustering_methods(embeddings, texts)
            print(f"   Best method: {benchmark_results['method_used']}")
            print(f"   Quality score: {benchmark_results['quality_metrics']['overall_cohesion']:.3f}")
        except Exception as e:
            print(f"   Benchmarking failed: {e}")
    else:
        print("   Benchmarking requires enhanced features")
    
    print("\n✅ Integration test completed!")
    print("\n🎯 To integrate with your existing TORI pipeline:")
    print("   1. Import: from enhanced_clustering_integration import enhanced_cluster_concepts")
    print("   2. Replace: labels = your_clustering_function(embeddings)")
    print("   3. With: labels = enhanced_cluster_concepts(embeddings)")
    print("   4. For full features: result = enhanced_cluster_concepts(embeddings, texts, return_enhanced_results=True)")
