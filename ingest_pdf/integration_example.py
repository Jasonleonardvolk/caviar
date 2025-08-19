# Simple TORI Clustering Integration Example 
from clustering_pipeline import TORIClusteringPipeline, ConceptData 
import numpy as np 
 
# Replace your existing clustering call: 
# labels = your_clustering_function(embeddings) 
 
# With enhanced clustering: 
pipeline = TORIClusteringPipeline() 
concepts = [ConceptData(f'concept_{i}', f'Text {i}', embedding.tolist(), {}) for i, embedding in enumerate(embeddings)] 
result = pipeline.process_concepts(concepts) 
enhanced_labels = [r.cluster_id for r in result.clustering_results] 
