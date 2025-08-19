"""
TORI Clustering Pipeline Integration
Seamless integration of enhanced clustering system with existing TORI pipeline.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import numpy as np

# Add TORI modules to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from clustering import run_oscillator_clustering_with_metrics, run_oscillator_clustering
from clustering_enhanced import benchmark_all_clustering_methods
from clustering_config import ConfigManager, TORIClusteringConfig
from clustering_monitor import ClusteringMonitor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('TORIPipelineIntegration')

@dataclass
class ConceptData:
    """Standardized concept data structure for pipeline integration."""
    concept_id: str
    text: str
    embedding: List[float]
    metadata: Dict[str, Any]
    source_document: Optional[str] = None
    page_number: Optional[int] = None
    confidence: Optional[float] = None
    timestamp: Optional[float] = None

@dataclass
class ClusteringResult:
    """Standardized clustering result for pipeline integration."""
    concept_id: str
    cluster_id: int
    cluster_confidence: float
    method_used: str
    processing_time: float
    quality_metrics: Dict[str, float]
    trace_info: Dict[str, Any]

@dataclass
class PipelineResult:
    """Complete pipeline result including all clustering information."""
    concepts: List[ConceptData]
    clustering_results: List[ClusteringResult]
    cluster_summary: Dict[int, Dict[str, Any]]
    pipeline_metadata: Dict[str, Any]
    performance_metrics: Dict[str, float]
    quality_assessment: Dict[str, Any]

class TORIClusteringPipeline:
    """
    Enhanced clustering pipeline that integrates seamlessly with existing TORI workflows.
    """
    
    def __init__(self, 
                 config_manager: Optional[ConfigManager] = None,
                 monitor: Optional[ClusteringMonitor] = None,
                 enable_monitoring: bool = True):
        
        self.config_manager = config_manager or ConfigManager()
        self.monitor = monitor if monitor else (ClusteringMonitor() if enable_monitoring else None)
        self.config = self.config_manager.get_config()
        
        # Pipeline state
        self.processing_history: List[Dict[str, Any]] = []
        self.cached_results: Dict[str, Any] = {}
        
        logger.info("TORI Clustering Pipeline initialized")
    
    def process_concepts(self, 
                        concepts: List[ConceptData],
                        method: Optional[str] = None,
                        enable_benchmarking: bool = False) -> PipelineResult:
        """
        Process concepts through the enhanced clustering pipeline.
        
        Args:
            concepts: List of concept data to cluster
            method: Specific clustering method to use (or None for auto-selection)
            enable_benchmarking: Whether to run benchmark comparison
            
        Returns:
            Complete pipeline result with clustering information
        """
        start_time = time.time()
        
        try:
            # Validate inputs
            if not concepts:
                raise ValueError("No concepts provided for clustering")
            
            # Extract embeddings
            embeddings = np.array([concept.embedding for concept in concepts])
            
            # Determine clustering method
            if method is None:
                method = self._select_optimal_method(embeddings)
            
            # Configure clustering based on dataset characteristics
            config = self._get_optimized_config(embeddings, method)
            
            # Run clustering
            if enable_benchmarking:
                clustering_results = self._run_benchmark_clustering(embeddings, config)
                best_method = self._select_best_method(clustering_results)
                clustering_result = clustering_results[best_method]
            else:
                clustering_result = self._run_single_clustering(embeddings, method, config)
            
            # Process results
            pipeline_result = self._process_clustering_results(
                concepts, clustering_result, method, start_time
            )
            
            # Monitor performance
            if self.monitor:
                self.monitor.record_clustering_result(clustering_result, embeddings, start_time)
            
            # Cache results
            self._cache_results(pipeline_result)
            
            # Update history
            self._update_processing_history(pipeline_result, start_time)
            
            logger.info(f"Pipeline processed {len(concepts)} concepts into "
                       f"{len(pipeline_result.cluster_summary)} clusters using {method}")
            
            return pipeline_result
            
        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}")
            # Return empty result with error information
            return PipelineResult(
                concepts=concepts,
                clustering_results=[],
                cluster_summary={},
                pipeline_metadata={
                    'error': str(e),
                    'processing_time': time.time() - start_time
                },
                performance_metrics={},
                quality_assessment={}
            )
    
    def _select_optimal_method(self, embeddings: np.ndarray) -> str:
        """Select optimal clustering method based on dataset characteristics."""
        n_concepts, n_dimensions = embeddings.shape
        
        # Decision logic based on dataset characteristics
        if n_concepts < 50:
            return 'kmeans'  # Fast for small datasets
        elif n_concepts < 500:
            return 'oscillator'  # Optimal for medium datasets
        elif n_concepts < 2000:
            return 'oscillator'  # Still good with proper batching
        else:
            return 'kmeans'  # Scalable for very large datasets
    
    def _get_optimized_config(self, embeddings: np.ndarray, method: str) -> TORIClusteringConfig:
        """Get optimized configuration for the specific dataset and method."""
        n_concepts, n_dimensions = embeddings.shape
        
        # Use existing config as base
        config = self.config_manager.get_config()
        
        # Adjust based on dataset size
        if n_concepts > 1000:
            config.performance.batch_size = min(1000, n_concepts // 4)
            config.performance.use_multiprocessing = True
        else:
            config.performance.batch_size = n_concepts
            config.performance.use_multiprocessing = False
        
        # Adjust based on method
        if method == 'oscillator':
            if n_concepts > 500:
                config.oscillator.steps = min(100, config.oscillator.steps)
        
        return config
    
    def _run_single_clustering(self, embeddings: np.ndarray, method: str, 
                             config: TORIClusteringConfig) -> Dict[str, Any]:
        """Run clustering with a single method."""
        
        if method == 'oscillator':
            return run_oscillator_clustering_with_metrics(
                embeddings,
                steps=config.oscillator.steps,
                tol=config.oscillator.tolerance,
                cohesion_threshold=config.oscillator.cohesion_threshold,
                enable_logging=False
            )
        else:
            # Use benchmark system to run single method
            try:
                results = benchmark_all_clustering_methods(
                    embeddings, 
                    methods=[method], 
                    enable_logging=False
                )
                return results.get(method, {})
            except Exception as e:
                logger.warning(f"Method {method} failed, falling back to oscillator: {e}")
                return run_oscillator_clustering_with_metrics(embeddings, enable_logging=False)
    
    def _run_benchmark_clustering(self, embeddings: np.ndarray, 
                                config: TORIClusteringConfig) -> Dict[str, Any]:
        """Run benchmark clustering with multiple methods."""
        return benchmark_all_clustering_methods(
            embeddings,
            methods=config.benchmark.enabled_methods,
            enable_logging=config.benchmark.enable_logging
        )
    
    def _select_best_method(self, clustering_results: Dict[str, Any]) -> str:
        """Select the best clustering method from benchmark results."""
        valid_results = {k: v for k, v in clustering_results.items() if 'error' not in v}
        
        if not valid_results:
            return 'oscillator'  # Default fallback
        
        # Composite scoring based on quality and performance
        best_method = None
        best_score = -1
        
        for method, result in valid_results.items():
            cohesion = result.get('avg_cohesion', 0)
            runtime = result.get('runtime', float('inf'))
            n_clusters = result.get('n_clusters', 0)
            
            # Scoring formula (prioritize quality over speed)
            score = cohesion * 0.7 + (1 / max(runtime, 0.1)) * 0.2 + (n_clusters > 1) * 0.1
            
            if score > best_score:
                best_score = score
                best_method = method
        
        return best_method or 'oscillator'
    
    def _process_clustering_results(self, concepts: List[ConceptData], 
                                  clustering_result: Dict[str, Any],
                                  method: str, start_time: float) -> PipelineResult:
        """Process clustering results into standardized pipeline result."""
        
        # Extract clustering information
        labels = clustering_result.get('labels', [])
        clusters = clustering_result.get('clusters', {})
        cohesion_scores = clustering_result.get('cohesion_scores', {})
        
        # Create clustering results for each concept
        clustering_results = []
        for i, concept in enumerate(concepts):
            if i < len(labels):
                cluster_id = labels[i]
                clustering_results.append(ClusteringResult(
                    concept_id=concept.concept_id,
                    cluster_id=cluster_id,
                    cluster_confidence=cohesion_scores.get(cluster_id, 0.0),
                    method_used=method,
                    processing_time=time.time() - start_time,
                    quality_metrics={
                        'cohesion': cohesion_scores.get(cluster_id, 0.0),
                        'overall_quality': clustering_result.get('avg_cohesion', 0.0)
                    },
                    trace_info={
                        'convergence_step': clustering_result.get('convergence_step'),
                        'total_steps': clustering_result.get('total_steps'),
                        'singleton_merges': clustering_result.get('singleton_merges', 0),
                        'orphan_reassignments': clustering_result.get('orphan_reassignments', 0)
                    }
                ))
        
        # Create cluster summary
        cluster_summary = {}
        for cluster_id, member_indices in clusters.items():
            cluster_concepts = [concepts[i] for i in member_indices if i < len(concepts)]
            cluster_summary[cluster_id] = {
                'size': len(cluster_concepts),
                'cohesion': cohesion_scores.get(cluster_id, 0.0),
                'representative_texts': [c.text[:100] + '...' for c in cluster_concepts[:3]],
                'source_documents': list(set(c.source_document for c in cluster_concepts if c.source_document)),
                'avg_confidence': np.mean([c.confidence for c in cluster_concepts if c.confidence is not None])
            }
        
        # Performance metrics
        total_time = time.time() - start_time
        performance_metrics = {
            'total_processing_time': total_time,
            'clustering_runtime': clustering_result.get('runtime', 0.0),
            'concepts_per_second': len(concepts) / max(total_time, 0.001),
            'method_used': method,
            'convergence_efficiency': clustering_result.get('convergence_step', 0) / max(clustering_result.get('total_steps', 1), 1)
        }
        
        # Quality assessment
        quality_assessment = {
            'overall_cohesion': clustering_result.get('avg_cohesion', 0.0),
            'num_clusters': len(clusters),
            'singleton_clusters': sum(1 for size in [len(members) for members in clusters.values()] if size == 1),
            'largest_cluster_size': max([len(members) for members in clusters.values()], default=0),
            'cluster_size_distribution': {
                'min': min([len(members) for members in clusters.values()], default=0),
                'max': max([len(members) for members in clusters.values()], default=0),
                'avg': np.mean([len(members) for members in clusters.values()]) if clusters else 0
            }
        }
        
        # Pipeline metadata
        pipeline_metadata = {
            'processed_at': time.time(),
            'method_used': method,
            'config_environment': self.config_manager.current_env,
            'total_concepts': len(concepts),
            'successful_clusters': len(clusters),
            'processing_success': True
        }
        
        return PipelineResult(
            concepts=concepts,
            clustering_results=clustering_results,
            cluster_summary=cluster_summary,
            pipeline_metadata=pipeline_metadata,
            performance_metrics=performance_metrics,
            quality_assessment=quality_assessment
        )
    
    def _cache_results(self, result: PipelineResult):
        """Cache results for future use."""
        if not self.config.performance.enable_caching:
            return
        
        # Create cache key based on concept IDs and method
        concept_ids = [c.concept_id for c in result.concepts]
        method = result.pipeline_metadata.get('method_used', 'unknown')
        cache_key = f"{hash(str(sorted(concept_ids)))}_{method}"
        
        # Store with timestamp
        self.cached_results[cache_key] = {
            'result': result,
            'timestamp': time.time()
        }
        
        # Cleanup old cache entries
        cutoff_time = time.time() - (self.config.performance.cache_ttl_hours * 3600)
        self.cached_results = {
            k: v for k, v in self.cached_results.items() 
            if v['timestamp'] > cutoff_time
        }
    
    def _update_processing_history(self, result: PipelineResult, start_time: float):
        """Update processing history for monitoring and analysis."""
        history_entry = {
            'timestamp': time.time(),
            'processing_time': time.time() - start_time,
            'num_concepts': len(result.concepts),
            'num_clusters': len(result.cluster_summary),
            'method_used': result.pipeline_metadata.get('method_used'),
            'avg_cohesion': result.quality_assessment.get('overall_cohesion', 0.0),
            'success': result.pipeline_metadata.get('processing_success', False)
        }
        
        self.processing_history.append(history_entry)
        
        # Keep only recent history
        max_history = 1000
        if len(self.processing_history) > max_history:
            self.processing_history = self.processing_history[-max_history:]
    
    def get_processing_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get processing statistics for the specified time period."""
        cutoff_time = time.time() - (hours * 3600)
        recent_history = [h for h in self.processing_history if h['timestamp'] > cutoff_time]
        
        if not recent_history:
            return {'error': 'No recent processing history available'}
        
        return {
            'total_runs': len(recent_history),
            'successful_runs': sum(1 for h in recent_history if h['success']),
            'avg_processing_time': np.mean([h['processing_time'] for h in recent_history]),
            'avg_concepts_per_run': np.mean([h['num_concepts'] for h in recent_history]),
            'avg_clusters_per_run': np.mean([h['num_clusters'] for h in recent_history]),
            'avg_cohesion': np.mean([h['avg_cohesion'] for h in recent_history]),
            'method_usage': {
                method: sum(1 for h in recent_history if h['method_used'] == method)
                for method in set(h['method_used'] for h in recent_history)
            }
        }
    
    def export_pipeline_results(self, result: PipelineResult, 
                               format: str = 'json', 
                               filepath: Optional[str] = None) -> str:
        """Export pipeline results in specified format."""
        
        if format.lower() == 'json':
            export_data = {
                'concepts': [asdict(c) for c in result.concepts],
                'clustering_results': [asdict(r) for r in result.clustering_results],
                'cluster_summary': result.cluster_summary,
                'pipeline_metadata': result.pipeline_metadata,
                'performance_metrics': result.performance_metrics,
                'quality_assessment': result.quality_assessment
            }
            
            if filepath:
                with open(filepath, 'w') as f:
                    json.dump(export_data, f, indent=2)
                return filepath
            else:
                return json.dumps(export_data, indent=2)
        
        elif format.lower() == 'csv':
            # Export concepts with cluster assignments
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Header
            writer.writerow(['concept_id', 'cluster_id', 'cluster_confidence', 'method_used', 'text_preview'])
            
            # Data
            for concept, cluster_result in zip(result.concepts, result.clustering_results):
                writer.writerow([
                    concept.concept_id,
                    cluster_result.cluster_id,
                    cluster_result.cluster_confidence,
                    cluster_result.method_used,
                    concept.text[:100] + '...' if len(concept.text) > 100 else concept.text
                ])
            
            csv_content = output.getvalue()
            
            if filepath:
                with open(filepath, 'w', newline='') as f:
                    f.write(csv_content)
                return filepath
            else:
                return csv_content
        
        else:
            raise ValueError(f"Unsupported export format: {format}")

# Utility functions for pipeline integration

def integrate_with_existing_pipeline(existing_clustering_function: callable) -> callable:
    """
    Decorator to integrate enhanced clustering with existing TORI pipeline functions.
    """
    def enhanced_clustering_wrapper(*args, **kwargs):
        # Create pipeline instance
        pipeline = TORIClusteringPipeline()
        
        # Try to extract embeddings from existing function arguments
        try:
            # Call original function to get embeddings
            original_result = existing_clustering_function(*args, **kwargs)
            
            # If original function returns embeddings, use them
            if isinstance(original_result, np.ndarray):
                embeddings = original_result
                
                # Create concept data from embeddings
                concepts = [
                    ConceptData(
                        concept_id=f"concept_{i}",
                        text=f"Concept {i}",
                        embedding=embeddings[i].tolist(),
                        metadata={}
                    )
                    for i in range(len(embeddings))
                ]
                
                # Process through enhanced pipeline
                result = pipeline.process_concepts(concepts)
                return result
            else:
                return original_result
                
        except Exception as e:
            logger.error(f"Enhanced clustering integration failed: {e}")
            return existing_clustering_function(*args, **kwargs)
    
    return enhanced_clustering_wrapper

def convert_legacy_results(legacy_labels: List[int], 
                         concepts: List[ConceptData]) -> PipelineResult:
    """Convert legacy clustering results to new pipeline format."""
    
    # Create clustering results
    clustering_results = []
    for i, (concept, label) in enumerate(zip(concepts, legacy_labels)):
        clustering_results.append(ClusteringResult(
            concept_id=concept.concept_id,
            cluster_id=label,
            cluster_confidence=0.5,  # Default confidence
            method_used='legacy',
            processing_time=0.0,
            quality_metrics={'cohesion': 0.0},
            trace_info={}
        ))
    
    # Create cluster summary
    cluster_summary = {}
    clusters = {}
    for i, label in enumerate(legacy_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(i)
    
    for cluster_id, member_indices in clusters.items():
        cluster_concepts = [concepts[i] for i in member_indices]
        cluster_summary[cluster_id] = {
            'size': len(cluster_concepts),
            'cohesion': 0.0,
            'representative_texts': [c.text[:100] + '...' for c in cluster_concepts[:3]],
            'source_documents': [],
            'avg_confidence': 0.5
        }
    
    return PipelineResult(
        concepts=concepts,
        clustering_results=clustering_results,
        cluster_summary=cluster_summary,
        pipeline_metadata={'method_used': 'legacy', 'processed_at': time.time()},
        performance_metrics={'total_processing_time': 0.0},
        quality_assessment={'overall_cohesion': 0.0, 'num_clusters': len(cluster_summary)}
    )

# Example usage
if __name__ == "__main__":
    print("🔗 TORI Clustering Pipeline Integration Demo")
    print("===========================================")
    
    # Create sample concepts
    sample_concepts = [
        ConceptData(
            concept_id=f"concept_{i}",
            text=f"Sample concept {i} about advanced topics in AI and machine learning",
            embedding=np.random.randn(384).tolist(),
            metadata={'importance': np.random.random()},
            source_document=f"doc_{i//3}.pdf",
            confidence=0.8 + np.random.random() * 0.2
        )
        for i in range(20)
    ]
    
    # Create pipeline
    pipeline = TORIClusteringPipeline()
    
    # Process concepts
    result = pipeline.process_concepts(sample_concepts, enable_benchmarking=True)
    
    # Display results
    print(f"\n📊 Pipeline Results:")
    print(f"   Processed: {len(result.concepts)} concepts")
    print(f"   Clusters: {len(result.cluster_summary)}")
    print(f"   Method: {result.pipeline_metadata['method_used']}")
    print(f"   Quality: {result.quality_assessment['overall_cohesion']:.3f}")
    print(f"   Processing time: {result.performance_metrics['total_processing_time']:.3f}s")
    
    # Export results
    json_export = pipeline.export_pipeline_results(result, 'json')
    print(f"\n💾 Results exported to JSON ({len(json_export)} characters)")
    
    # Show processing statistics
    stats = pipeline.get_processing_statistics()
    print(f"\n📈 Processing Statistics:")
    print(f"   Total runs: {stats['total_runs']}")
    print(f"   Success rate: {stats['successful_runs']}/{stats['total_runs']}")
    print(f"   Avg processing time: {stats['avg_processing_time']:.3f}s")
    
    print(f"\n✅ Pipeline integration ready for production use!")
