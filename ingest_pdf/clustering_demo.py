#!/usr/bin/env python3
"""
TORI Clustering Integration Demo
Demonstrates the enhanced clustering system with benchmarking and concept scoring integration.
"""

import numpy as np
import json
import time
from typing import List, Dict, Any
import os
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from clustering import run_oscillator_clustering_with_metrics, benchmark_all_clustering_methods
from clustering_enhanced import (
    kmeans_clustering, 
    hdbscan_clustering, 
    affinity_propagation_clustering,
    compute_silhouette_score
)

def generate_sample_concept_embeddings(n_concepts: int = 100, n_dimensions: int = 384, n_clusters: int = 5) -> tuple:
    """
    Generate sample concept embeddings with known ground truth clusters.
    """
    np.random.seed(42)  # For reproducibility
    
    embeddings = []
    true_labels = []
    concept_texts = []
    
    # Generate cluster centers
    cluster_centers = np.random.randn(n_clusters, n_dimensions)
    cluster_centers = cluster_centers / np.linalg.norm(cluster_centers, axis=1, keepdims=True)
    
    concepts_per_cluster = n_concepts // n_clusters
    remaining_concepts = n_concepts % n_clusters
    
    concept_themes = [
        "machine learning algorithms and neural networks",
        "quantum physics and particle interactions", 
        "climate change and environmental sustainability",
        "biomedical research and drug discovery",
        "artificial intelligence and robotics"
    ]
    
    for cluster_id in range(n_clusters):
        # Number of concepts for this cluster
        n_cluster_concepts = concepts_per_cluster + (1 if cluster_id < remaining_concepts else 0)
        
        # Generate embeddings around cluster center
        cluster_embeddings = []
        for i in range(n_cluster_concepts):
            # Add noise to cluster center
            noise = np.random.normal(0, 0.3, n_dimensions)
            embedding = cluster_centers[cluster_id] + noise
            embedding = embedding / np.linalg.norm(embedding)  # Normalize
            cluster_embeddings.append(embedding)
            
            # Generate concept text
            theme = concept_themes[cluster_id % len(concept_themes)]
            concept_text = f"Concept {len(embeddings)+1}: Advanced research in {theme} with focus on {['theoretical foundations', 'practical applications', 'empirical analysis', 'computational methods', 'experimental validation'][i % 5]}"
            concept_texts.append(concept_text)
            
            true_labels.append(cluster_id)
        
        embeddings.extend(cluster_embeddings)
    
    return np.array(embeddings), true_labels, concept_texts

def evaluate_clustering_against_ground_truth(predicted_labels: List[int], true_labels: List[int]) -> Dict[str, float]:
    """
    Evaluate clustering performance against ground truth using various metrics.
    """
    try:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score
        
        metrics = {
            'adjusted_rand_score': adjusted_rand_score(true_labels, predicted_labels),
            'normalized_mutual_info': normalized_mutual_info_score(true_labels, predicted_labels),
            'fowlkes_mallows_score': fowlkes_mallows_score(true_labels, predicted_labels)
        }
    except ImportError:
        # Fallback to simple accuracy if sklearn not available
        # Map predicted clusters to true clusters using Hungarian algorithm approximation
        from collections import Counter, defaultdict
        
        # Create confusion matrix
        confusion = defaultdict(lambda: defaultdict(int))
        for true_label, pred_label in zip(true_labels, predicted_labels):
            confusion[pred_label][true_label] += 1
        
        # Simple accuracy: best case alignment
        total_correct = 0
        used_true_labels = set()
        
        # Greedy assignment of predicted clusters to true clusters
        for pred_cluster in sorted(confusion.keys(), key=lambda x: len(confusion[x]), reverse=True):
            best_true_cluster = max(confusion[pred_cluster].keys(), 
                                   key=lambda x: confusion[pred_cluster][x] if x not in used_true_labels else 0)
            if best_true_cluster not in used_true_labels:
                total_correct += confusion[pred_cluster][best_true_cluster]
                used_true_labels.add(best_true_cluster)
        
        accuracy = total_correct / len(true_labels)
        
        metrics = {
            'clustering_accuracy': accuracy,
            'perfect_clusters': sum(1 for cluster_assigns in confusion.values() 
                                   if len(cluster_assigns) == 1 and list(cluster_assigns.values())[0] > 1)
        }
    
    return metrics

def run_comprehensive_clustering_demo():
    """
    Run a comprehensive demonstration of the enhanced clustering system.
    """
    print("🚀 TORI Enhanced Clustering System Demo")
    print("=" * 60)
    
    # Generate sample data
    print("\n📊 Generating sample concept embeddings...")
    embeddings, true_labels, concept_texts = generate_sample_concept_embeddings(
        n_concepts=150, n_dimensions=384, n_clusters=6
    )
    
    print(f"Generated {len(embeddings)} concept embeddings with {embeddings.shape[1]} dimensions")
    print(f"Ground truth: {len(set(true_labels))} clusters")
    
    # Benchmark all clustering methods
    print("\n🔬 Running comprehensive clustering benchmark...")
    methods = ['oscillator', 'kmeans', 'hdbscan', 'affinity_propagation']
    
    try:
        results = benchmark_all_clustering_methods(embeddings, methods, enable_logging=True)
        
        print("\n📈 CLUSTERING BENCHMARK RESULTS")
        print("=" * 60)
        
        # Create detailed comparison table
        print(f"{'Method':<20} {'Clusters':<10} {'Cohesion':<12} {'Silhouette':<12} {'Runtime(s)':<12} {'Accuracy':<12}")
        print("-" * 88)
        
        best_method = None
        best_score = -1
        
        for method_name, result in results.items():
            if 'error' in result:
                print(f"{method_name:<20} ERROR: {result['error']}")
                continue
            
            # Calculate silhouette score
            try:
                silhouette = compute_silhouette_score(embeddings, result['labels'])
            except:
                silhouette = 0.0
            
            # Evaluate against ground truth
            ground_truth_metrics = evaluate_clustering_against_ground_truth(result['labels'], true_labels)
            accuracy = list(ground_truth_metrics.values())[0] if ground_truth_metrics else 0.0
            
            # Composite score for ranking
            composite_score = (result['avg_cohesion'] * 0.4 + 
                             silhouette * 0.3 + 
                             accuracy * 0.3)
            
            if composite_score > best_score:
                best_score = composite_score
                best_method = method_name
            
            print(f"{method_name:<20} {result['n_clusters']:<10} {result['avg_cohesion']:.3f}{'':8} "
                  f"{silhouette:.3f}{'':8} {result['runtime']:.3f}{'':8} {accuracy:.3f}")
        
        print("\n🏆 BEST PERFORMING METHOD:", best_method.upper())
        
        # Detailed analysis of best method
        if best_method and best_method in results:
            best_result = results[best_method]
            print(f"\n🔍 DETAILED ANALYSIS - {best_method.upper()}")
            print("-" * 40)
            
            if best_method == 'oscillator':
                print(f"Convergence: {best_result.get('convergence_step', 'N/A')}/{best_result.get('total_steps', 'N/A')} steps")
                print(f"Phase Variance: {best_result.get('phase_variance', 0):.3f}")
                print(f"Singleton Merges: {best_result.get('singleton_merges', 0)}")
                print(f"Orphan Reassignments: {best_result.get('orphan_reassignments', 0)}")
                print(f"Low Cohesion Removed: {best_result.get('removed_low_cohesion', 0)}")
            
            # Cluster size distribution
            cluster_sizes = [len(members) for members in best_result['clusters'].values()]
            if cluster_sizes:
                print(f"Cluster sizes: min={min(cluster_sizes)}, max={max(cluster_sizes)}, "
                      f"avg={np.mean(cluster_sizes):.1f}")
            
            # Show sample concepts from each cluster
            print(f"\n📝 SAMPLE CONCEPTS BY CLUSTER")
            print("-" * 40)
            for cluster_id, member_indices in list(best_result['clusters'].items())[:3]:  # Show first 3 clusters
                print(f"\nCluster {cluster_id} ({len(member_indices)} concepts):")
                sample_indices = member_indices[:2]  # Show 2 samples per cluster
                for idx in sample_indices:
                    if idx < len(concept_texts):
                        text_preview = concept_texts[idx][:80] + "..." if len(concept_texts[idx]) > 80 else concept_texts[idx]
                        print(f"  • {text_preview}")
    
    except Exception as e:
        print(f"❌ Error running benchmark: {e}")
        print("Falling back to oscillator clustering only...")
        
        # Fallback to oscillator clustering only
        result = run_oscillator_clustering_with_metrics(embeddings, enable_logging=True)
        
        print(f"\n🎯 OSCILLATOR CLUSTERING RESULTS")
        print("-" * 40)
        print(f"Clusters found: {result['n_clusters']}")
        print(f"Average cohesion: {result['avg_cohesion']:.3f}")
        print(f"Runtime: {result['runtime']:.3f}s")
        print(f"Convergence: {result['convergence_step']}/{result['total_steps']} steps")
    
    # Performance analysis
    print(f"\n⚡ PERFORMANCE INSIGHTS")
    print("-" * 40)
    
    # Memory usage estimate
    memory_usage_mb = (embeddings.nbytes + len(concept_texts) * 100) / (1024 * 1024)  # Rough estimate
    print(f"Memory usage: ~{memory_usage_mb:.1f} MB")
    print(f"Scalability: {len(embeddings)} concepts processed")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    print("-" * 40)
    
    if len(embeddings) < 50:
        print("• Consider K-means for small datasets (< 50 concepts)")
    elif len(embeddings) < 500:
        print("• Oscillator clustering recommended for medium datasets (50-500 concepts)")
        print("• HDBSCAN good alternative for noise tolerance")
    else:
        print("• For large datasets (> 500 concepts), consider:")
        print("  - Hierarchical clustering for better scalability")
        print("  - Dimensionality reduction before clustering")
        print("  - Batch processing for memory efficiency")
    
    print(f"\n✅ Demo completed successfully!")
    return results if 'results' in locals() else {}

def save_demo_results(results: Dict[str, Any], filename: str = "clustering_demo_results.json"):
    """
    Save demo results to file for analysis.
    """
    try:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for method, result in results.items():
            if 'error' not in result:
                serializable_result = {}
                for key, value in result.items():
                    if hasattr(value, 'tolist'):
                        serializable_result[key] = value.tolist()
                    else:
                        serializable_result[key] = value
                serializable_results[method] = serializable_result
            else:
                serializable_results[method] = result
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"💾 Results saved to {filename}")
        
    except Exception as e:
        print(f"❌ Error saving results: {e}")

def create_typescript_integration_example():
    """
    Create an example TypeScript file showing how to integrate with the clustering system.
    """
    typescript_example = '''// Example TypeScript integration with TORI Clustering System
import { ConceptClusteringEngine, createConceptTuples } from './conceptScoring';
import { benchmarkClustering } from './clusterBenchmark';

async function runConceptClusteringExample() {
  // Create sample concept data
  const conceptTexts = [
    "Machine learning algorithms for natural language processing",
    "Deep neural networks and backpropagation optimization",
    "Quantum computing and qubit entanglement principles",
    "Climate change impacts on global weather patterns",
    "Renewable energy sources and sustainability metrics"
  ];
  
  // Mock embeddings (in real usage, these would come from your embedding model)
  const mockEmbeddings = conceptTexts.map(() => 
    Array.from({length: 384}, () => Math.random() - 0.5).map(x => x / Math.sqrt(384))
  );
  
  // Create concept tuples
  const concepts = createConceptTuples(conceptTexts, mockEmbeddings, 
    conceptTexts.map((_, i) => ({ importance: Math.random(), sourceDocument: `doc_${i}.pdf` }))
  );
  
  // Initialize clustering engine
  const engine = new ConceptClusteringEngine();
  engine.addConcepts(concepts);
  
  // Run comprehensive clustering analysis
  try {
    const result = await engine.runClusteringAnalysis(
      ['oscillator', 'kmeans', 'hdbscan'],
      {
        enableBenchmarking: true,
        selectBestMethod: true,
        generateTraces: true,
        saveHistory: true
      }
    );
    
    console.log('Clustering Analysis Results:');
    console.log('Recommended Method:', result.recommendedMethod);
    console.log('Benchmark Report:');
    console.log(result.benchmarkReport);
    
    // Get cluster insights
    const insights = engine.generateInsights();
    console.log('Clustering Insights:');
    console.log(insights.summary);
    console.log('Recommendations:', insights.recommendations);
    
    // Export results for further analysis
    const exportData = engine.exportResults();
    console.log('Cluster Summary:', exportData.clusterSummary);
    
  } catch (error) {
    console.error('Clustering analysis failed:', error);
  }
}

// Run the example
runConceptClusteringExample();
'''
    
    with open('clustering_integration_example.ts', 'w') as f:
        f.write(typescript_example)
    
    print("💻 TypeScript integration example created: clustering_integration_example.ts")

if __name__ == "__main__":
    print("🎬 Starting TORI Clustering Integration Demo...")
    
    # Run main demo
    results = run_comprehensive_clustering_demo()
    
    # Save results
    if results:
        save_demo_results(results)
    
    # Create TypeScript integration example
    create_typescript_integration_example()
    
    print("\n🎉 All demonstrations completed!")
    print("\nNext steps:")
    print("1. Review clustering_demo_results.json for detailed analysis")
    print("2. Check clustering_integration_example.ts for TypeScript usage")
    print("3. Integrate the enhanced clustering system into your TORI pipeline")
    print("4. Tune clustering parameters based on your specific use case")
