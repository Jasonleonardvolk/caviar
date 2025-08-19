# TORI Concept Scoring Integration with Enhanced Clustering 
# Integrates advanced clustering with existing TORI concept scoring system 
 
from clustering import run_oscillator_clustering_with_metrics 
import numpy as np 
 
def enhanced_cluster_concepts(vectors, k_estimate): 
    """Drop-in replacement for clusterConcepts in conceptScoring.ts""" 
    result = run_oscillator_clustering_with_metrics(np.array(vectors), enable_logging=False) 
    return result['labels'] 
 
print("Integration helper created. Use enhanced_cluster_concepts() in your TypeScript bridge.") 
