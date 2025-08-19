// ingest_pdf/conceptScoring.ts
import { benchmarkClustering, generateBenchmarkReport } from './clusterBenchmark';

export interface ConceptTuple {
  conceptId: string;
  text: string;
  embedding: number[];
  clusterId?: number;
  clusterTrace?: ClusterTrace;
  coherenceScore?: number;
  importance?: number;
  timestamp?: number;
  sourceDocument?: string;
  pageNumber?: number;
}

export interface ClusterTrace {
  method: string;
  originalClusterId: number;
  finalClusterId: number;
  cohesionScore: number;
  mergedFrom?: number[];
  reassignedFrom?: number;
  convergenceStep?: number;
  phaseVariance?: number;
  similarityScore?: number;
  timestamp: number;
}

export interface ConceptClusteringResult {
  concepts: ConceptTuple[];
  clusteringResults: { [method: string]: any };
  benchmarkReport: string;
  recommendedMethod: string;
  clusterQualityMetrics: ClusterQualityMetrics;
}

export interface ClusterQualityMetrics {
  silhouetteScore: number;
  avgCohesion: number;
  clusterStability: number;
  convergenceEfficiency: number;
  clusterDistribution: {
    minSize: number;
    maxSize: number;
    avgSize: number;
    singletons: number;
  };
}

export class ConceptClusteringEngine {
  private concepts: ConceptTuple[] = [];
  private clusteringHistory: { [timestamp: number]: any } = {};
  
  constructor() {}
  
  /**
   * Add concepts to the clustering engine
   */
  addConcepts(concepts: ConceptTuple[]): void {
    this.concepts.push(...concepts);
  }
  
  /**
   * Clear all concepts
   */
  clearConcepts(): void {
    this.concepts = [];
  }
  
  /**
   * Get current concepts
   */
  getConcepts(): ConceptTuple[] {
    return [...this.concepts];
  }
  
  /**
   * Run comprehensive clustering analysis with benchmarking
   */
  async runClusteringAnalysis(
    methods: string[] = ['oscillator', 'kmeans', 'hdbscan'],
    options: {
      enableBenchmarking?: boolean;
      selectBestMethod?: boolean;
      generateTraces?: boolean;
      saveHistory?: boolean;
    } = {}
  ): Promise<ConceptClusteringResult> {
    
    if (this.concepts.length === 0) {
      throw new Error('No concepts available for clustering');
    }
    
    const {
      enableBenchmarking = true,
      selectBestMethod = true,
      generateTraces = true,
      saveHistory = true
    } = options;
    
    // Extract embeddings
    const embeddings = this.concepts.map(concept => concept.embedding);
    
    // Run clustering benchmark
    const clusteringResults = await benchmarkClustering(embeddings, methods);
    
    // Generate benchmark report
    const benchmarkReport = enableBenchmarking ? generateBenchmarkReport(clusteringResults) : '';
    
    // Select best method based on combined metrics
    const recommendedMethod = selectBestMethod ? this.selectBestClusteringMethod(clusteringResults) : methods[0];
    
    // Apply clustering results to concepts
    const bestResult = clusteringResults[recommendedMethod];
    if (bestResult && !('error' in bestResult)) {
      this.applyClustering(bestResult, generateTraces);
    }
    
    // Calculate cluster quality metrics
    const clusterQualityMetrics = this.calculateClusterQualityMetrics(clusteringResults);
    
    // Save to history if requested
    if (saveHistory) {
      const timestamp = Date.now();
      this.clusteringHistory[timestamp] = {
        methods,
        results: clusteringResults,
        recommendedMethod,
        conceptCount: this.concepts.length,
        qualityMetrics: clusterQualityMetrics
      };
    }
    
    return {
      concepts: this.getConcepts(),
      clusteringResults,
      benchmarkReport,
      recommendedMethod,
      clusterQualityMetrics
    };
  }
  
  /**
   * Select the best clustering method based on multiple criteria
   */
  private selectBestClusteringMethod(results: { [method: string]: any }): string {
    const validResults = Object.entries(results).filter(([, result]) => !('error' in result));
    
    if (validResults.length === 0) {
      return 'oscillator'; // Default fallback
    }
    
    let bestMethod = validResults[0][0];
    let bestScore = -Infinity;
    
    for (const [method, result] of validResults) {
      // Composite score based on multiple factors
      const cohesionWeight = 0.4;
      const silhouetteWeight = 0.3;
      const efficiencyWeight = 0.2;
      const stabilityWeight = 0.1;
      
      const cohesionScore = result.avgCohesion || 0;
      const silhouetteScore = result.silhouetteScore || 0;
      const efficiencyScore = 1 / Math.max(result.runtime, 1); // Inverse of runtime for efficiency
      
      // Stability bonus for oscillator method
      const stabilityScore = method === 'oscillator' ? 
        (result.convergenceStep < result.totalSteps ? 1 : 0.5) : 0.5;
      
      const compositeScore = 
        cohesionScore * cohesionWeight +
        silhouetteScore * silhouetteWeight +
        efficiencyScore * efficiencyWeight +
        stabilityScore * stabilityWeight;
      
      if (compositeScore > bestScore) {
        bestScore = compositeScore;
        bestMethod = method;
      }
    }
    
    return bestMethod;
  }
  
  /**
   * Apply clustering results to concepts
   */
  private applyClustering(result: any, generateTraces: boolean): void {
    const timestamp = Date.now();
    
    for (let i = 0; i < this.concepts.length; i++) {
      const concept = this.concepts[i];
      const clusterId = result.labels[i];
      
      concept.clusterId = clusterId;
      concept.coherenceScore = result.cohesionScores[clusterId] || 0;
      
      if (generateTraces) {
        const clusterTrace: ClusterTrace = {
          method: result.method,
          originalClusterId: clusterId,
          finalClusterId: clusterId,
          cohesionScore: concept.coherenceScore,
          timestamp
        };
        
        // Add method-specific trace information
        if (result.method === 'oscillator') {
          clusterTrace.convergenceStep = result.convergenceStep;
          clusterTrace.phaseVariance = result.phaseVariance;
        }
        
        concept.clusterTrace = clusterTrace;
      }
    }
  }
  
  /**
   * Calculate comprehensive cluster quality metrics
   */
  private calculateClusterQualityMetrics(results: { [method: string]: any }): ClusterQualityMetrics {
    const validResults = Object.values(results).filter(result => !('error' in result));
    
    if (validResults.length === 0) {
      return {
        silhouetteScore: 0,
        avgCohesion: 0,
        clusterStability: 0,
        convergenceEfficiency: 0,
        clusterDistribution: {
          minSize: 0,
          maxSize: 0,
          avgSize: 0,
          singletons: 0
        }
      };
    }
    
    // Aggregate metrics across all methods
    const avgSilhouette = validResults.reduce((sum, r) => sum + (r.silhouetteScore || 0), 0) / validResults.length;
    const avgCohesion = validResults.reduce((sum, r) => sum + (r.avgCohesion || 0), 0) / validResults.length;
    
    // Calculate stability (based on oscillator convergence if available)
    const oscillatorResult = validResults.find(r => r.method === 'oscillator');
    const clusterStability = oscillatorResult ? 
      (oscillatorResult.convergenceStep < oscillatorResult.totalSteps ? 1 : 0.5) : 0.5;
    
    // Calculate convergence efficiency
    const convergenceEfficiency = oscillatorResult ? 
      1 - (oscillatorResult.convergenceStep / oscillatorResult.totalSteps) : 0.5;
    
    // Calculate cluster distribution metrics
    const allClusterSizes = validResults.flatMap(r => 
      Object.values(r.clusters).map((cluster: any) => cluster.length)
    );
    
    const clusterDistribution = {
      minSize: allClusterSizes.length > 0 ? Math.min(...allClusterSizes) : 0,
      maxSize: allClusterSizes.length > 0 ? Math.max(...allClusterSizes) : 0,
      avgSize: allClusterSizes.length > 0 ? 
        allClusterSizes.reduce((sum, size) => sum + size, 0) / allClusterSizes.length : 0,
      singletons: allClusterSizes.filter(size => size === 1).length
    };
    
    return {
      silhouetteScore: avgSilhouette,
      avgCohesion,
      clusterStability,
      convergenceEfficiency,
      clusterDistribution
    };
  }
  
  /**
   * Get concepts by cluster
   */
  getConceptsByCluster(clusterId: number): ConceptTuple[] {
    return this.concepts.filter(concept => concept.clusterId === clusterId);
  }
  
  /**
   * Get cluster summary statistics
   */
  getClusterSummary(): { [clusterId: number]: any } {
    const clusterMap: { [clusterId: number]: ConceptTuple[] } = {};
    
    for (const concept of this.concepts) {
      if (concept.clusterId !== undefined) {
        if (!clusterMap[concept.clusterId]) {
          clusterMap[concept.clusterId] = [];
        }
        clusterMap[concept.clusterId].push(concept);
      }
    }
    
    const summary: { [clusterId: number]: any } = {};
    
    for (const [clusterId, concepts] of Object.entries(clusterMap)) {
      const numericClusterId = parseInt(clusterId);
      summary[numericClusterId] = {
        count: concepts.length,
        avgCoherence: concepts.reduce((sum, c) => sum + (c.coherenceScore || 0), 0) / concepts.length,
        avgImportance: concepts.reduce((sum, c) => sum + (c.importance || 0), 0) / concepts.length,
        method: concepts[0]?.clusterTrace?.method || 'unknown',
        representatives: concepts
          .sort((a, b) => (b.coherenceScore || 0) - (a.coherenceScore || 0))
          .slice(0, 3)
          .map(c => ({ id: c.conceptId, text: c.text.substring(0, 100) + '...' }))
      };
    }
    
    return summary;
  }
  
  /**
   * Export clustering results for analysis
   */
  exportResults(): {
    concepts: ConceptTuple[];
    clusterSummary: { [clusterId: number]: any };
    history: { [timestamp: number]: any };
  } {
    return {
      concepts: this.getConcepts(),
      clusterSummary: this.getClusterSummary(),
      history: this.clusteringHistory
    };
  }
  
  /**
   * Generate clustering insights and recommendations
   */
  generateInsights(): {
    summary: string;
    recommendations: string[];
    qualityAssessment: string;
  } {
    const clusterSummary = this.getClusterSummary();
    const totalClusters = Object.keys(clusterSummary).length;
    const totalConcepts = this.concepts.length;
    
    // Generate summary
    const summary = `
Clustering Analysis Summary:
- Total Concepts: ${totalConcepts}
- Total Clusters: ${totalClusters}
- Average Cluster Size: ${(totalConcepts / totalClusters).toFixed(1)}
- Clustering Coverage: ${(this.concepts.filter(c => c.clusterId !== undefined).length / totalConcepts * 100).toFixed(1)}%
    `.trim();
    
    // Generate recommendations
    const recommendations: string[] = [];
    
    // Check for singleton clusters
    const singletonCount = Object.values(clusterSummary).filter(cluster => cluster.count === 1).length;
    if (singletonCount > totalClusters * 0.3) {
      recommendations.push('Consider lowering cohesion threshold to merge more singleton clusters');
    }
    
    // Check for oversized clusters
    const maxClusterSize = Math.max(...Object.values(clusterSummary).map(cluster => cluster.count));
    if (maxClusterSize > totalConcepts * 0.5) {
      recommendations.push('Large clusters detected - consider increasing cohesion threshold for better separation');
    }
    
    // Check clustering efficiency
    const avgCoherence = Object.values(clusterSummary).reduce((sum, cluster) => sum + cluster.avgCoherence, 0) / totalClusters;
    if (avgCoherence < 0.3) {
      recommendations.push('Low average cohesion - consider using oscillator clustering or adjusting parameters');
    }
    
    if (recommendations.length === 0) {
      recommendations.push('Clustering quality appears optimal for current parameters');
    }
    
    // Quality assessment
    let qualityAssessment = 'Good';
    if (avgCoherence < 0.2 || singletonCount > totalClusters * 0.5) {
      qualityAssessment = 'Needs Improvement';
    } else if (avgCoherence > 0.5 && singletonCount < totalClusters * 0.2) {
      qualityAssessment = 'Excellent';
    }
    
    return {
      summary,
      recommendations,
      qualityAssessment: `Overall Quality: ${qualityAssessment} (Avg Cohesion: ${avgCoherence.toFixed(3)})`
    };
  }
}

// Utility functions for concept management

/**
 * Create concept tuples from raw text and embeddings
 */
export function createConceptTuples(
  texts: string[],
  embeddings: number[][],
  metadata: Partial<ConceptTuple>[] = []
): ConceptTuple[] {
  if (texts.length !== embeddings.length) {
    throw new Error('Texts and embeddings arrays must have the same length');
  }
  
  return texts.map((text, index) => ({
    conceptId: `concept_${index}_${Date.now()}`,
    text,
    embedding: embeddings[index],
    timestamp: Date.now(),
    ...metadata[index]
  }));
}

/**
 * Filter concepts by importance threshold
 */
export function filterConceptsByImportance(
  concepts: ConceptTuple[],
  threshold: number
): ConceptTuple[] {
  return concepts.filter(concept => (concept.importance || 0) >= threshold);
}

/**
 * Sort concepts by cluster and importance
 */
export function sortConceptsByClusterAndImportance(concepts: ConceptTuple[]): ConceptTuple[] {
  return concepts.sort((a, b) => {
    // First sort by cluster
    const clusterDiff = (a.clusterId || 0) - (b.clusterId || 0);
    if (clusterDiff !== 0) return clusterDiff;
    
    // Then by importance (descending)
    return (b.importance || 0) - (a.importance || 0);
  });
}

/**
 * Get cluster representatives (most important concepts from each cluster)
 */
export function getClusterRepresentatives(
  concepts: ConceptTuple[],
  representativesPerCluster: number = 3
): { [clusterId: number]: ConceptTuple[] } {
  const clusterMap: { [clusterId: number]: ConceptTuple[] } = {};
  
  // Group by cluster
  for (const concept of concepts) {
    if (concept.clusterId !== undefined) {
      if (!clusterMap[concept.clusterId]) {
        clusterMap[concept.clusterId] = [];
      }
      clusterMap[concept.clusterId].push(concept);
    }
  }
  
  // Get top representatives from each cluster
  const representatives: { [clusterId: number]: ConceptTuple[] } = {};
  
  for (const [clusterId, clusterConcepts] of Object.entries(clusterMap)) {
    const numericClusterId = parseInt(clusterId);
    representatives[numericClusterId] = clusterConcepts
      .sort((a, b) => (b.importance || 0) - (a.importance || 0))
      .slice(0, representativesPerCluster);
  }
  
  return representatives;
}

export default ConceptClusteringEngine;
