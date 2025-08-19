// src/lib/cognitive/conceptScoring_enhanced.ts
// Enhanced version of conceptScoring.ts with advanced clustering integration
// 
// This version integrates your existing composite scoring system with advanced
// clustering capabilities while preserving all your ConceptTuple structure,
// lineage tracking, and composite scoring logic.

import * as fs from 'fs';
import { ConceptClusteringBridge } from './conceptClusteringBridge.js';

interface DocumentData {
  id: string;
  text: string;
  summary: string;
  concepts: string[];
}

export interface ConceptOrigin {
  docId: string;
  occurrences: number;
  sample: string;
}

export interface ConceptTuple {
  name: string;
  clusterId: number;
  score: number;
  frequency: number;
  centrality: number;
  confidence: number;
  domainSalience: number;
  originDocs: ConceptOrigin[];
  mergedFrom: string[];
  scoreEvolution?: Array<{ stage: string; score: number }>;
  clusterTrace?: { 
    algorithm: string; 
    clusterId: number;
    qualityMetrics?: {
      cohesion?: number;
      silhouette?: number;
      convergence?: number;
    };
    performanceMetrics?: {
      runtime?: number;
      method?: string;
    };
  };
}

// Enhanced clustering configuration
interface ClusteringConfig {
  method: 'auto' | 'oscillator' | 'kmeans' | 'hdbscan';
  enableBenchmarking: boolean;
  enableMonitoring: boolean;
  debug: boolean;
}

// Performance tracking for clustering operations
interface ClusteringPerformance {
  startTime: number;
  endTime: number;
  method: string;
  conceptCount: number;
  clusterCount: number;
  qualityScore: number;
}

// Placeholder for concept embedding - enhanced with caching
const embeddingCache = new Map<string, number[]>();

function embedConcept(name: string): number[] {
  // Check cache first
  if (embeddingCache.has(name)) {
    return embeddingCache.get(name)!;
  }
  
  // In practice, call external API or model to get embedding vector
  // For demonstration, return a more sophisticated dummy vector
  const vec = new Array(10).fill(0).map((_, i) => {
    const charCode = name.charCodeAt(i % name.length) || 0;
    return Math.sin(charCode * (i + 1) / 100) * 0.5 + 0.5;
  });
  
  // Cache the result
  embeddingCache.set(name, vec);
  return vec;
}

// Enhanced clustering function with advanced algorithms
async function enhancedClusterConcepts(
  vectors: number[][], 
  k: number, 
  config: ClusteringConfig = { 
    method: 'auto', 
    enableBenchmarking: false, 
    enableMonitoring: true, 
    debug: false 
  }
): Promise<{
  labels: number[];
  clusteringInfo: {
    method: string;
    qualityMetrics?: any;
    performanceMetrics?: any;
    benchmarkResults?: any;
  };
}> {
  
  const bridge = new ConceptClusteringBridge();
  
  try {
    if (config.enableBenchmarking) {
      // Run comprehensive benchmark
      const benchmarkResult = await bridge.benchmarkClustering(vectors, k);
      
      return {
        labels: benchmarkResult.labels,
        clusteringInfo: {
          method: benchmarkResult.method_used || 'oscillator',
          qualityMetrics: benchmarkResult.quality_metrics,
          performanceMetrics: benchmarkResult.performance_stats,
          benchmarkResults: benchmarkResult
        }
      };
    } else {
      // Run single method
      const labels = await bridge.enhancedClusterConcepts(vectors, k, {
        method: config.method
      });
      
      return {
        labels,
        clusteringInfo: {
          method: config.method === 'auto' ? 'oscillator' : config.method,
          performanceMetrics: bridge.getPerformanceStats()
        }
      };
    }
  } catch (error) {
    console.warn(`Enhanced clustering failed: ${error.message}, falling back to K-means`);
    
    // Fallback to your original K-means implementation
    const labels = fallbackKMeansCluster(vectors, k);
    
    return {
      labels,
      clusteringInfo: {
        method: 'kmeans_fallback',
        performanceMetrics: { error: error.message }
      }
    };
  }
}

// Fallback K-means (your original implementation)
function fallbackKMeansCluster(vectors: number[][], k: number): number[] {
  const n = vectors.length;
  const dim = vectors[0].length;
  
  // Initialize centroids as first k vectors (or random selection)
  const centroids = vectors.slice(0, k).map(v => v.slice());
  const labels = new Array(n).fill(-1);
  
  for (let iter = 0; iter < 100; iter++) {
    let changed = false;
    
    // Assign each vector to nearest centroid
    for (let i = 0; i < n; i++) {
      let bestDist = Infinity;
      let bestCluster = -1;
      
      for (let j = 0; j < centroids.length; j++) {
        // Euclidean distance
        let dist = 0;
        for (let d = 0; d < dim; d++) {
          const diff = vectors[i][d] - centroids[j][d];
          dist += diff * diff;
        }
        
        if (dist < bestDist) {
          bestDist = dist;
          bestCluster = j;
        }
      }
      
      if (labels[i] !== bestCluster) {
        labels[i] = bestCluster;
        changed = true;
      }
    }
    
    if (!changed) break;
    
    // Recompute centroids
    const newCentroids = Array.from({ length: k }, () => new Array(dim).fill(0));
    const counts = new Array(k).fill(0);
    
    for (let i = 0; i < n; i++) {
      const c = labels[i];
      counts[c] += 1;
      for (let d = 0; d < dim; d++) {
        newCentroids[c][d] += vectors[i][d];
      }
    }
    
    for (let j = 0; j < k; j++) {
      if (counts[j] > 0) {
        for (let d = 0; d < dim; d++) {
          newCentroids[j][d] /= counts[j];
        }
        centroids[j] = newCentroids[j];
      }
    }
  }
  
  return labels;
}

// Enhanced refineConcepts function with advanced clustering
export async function refineConcepts(
  docs: DocumentData[], 
  config: ClusteringConfig = { 
    method: 'auto', 
    enableBenchmarking: false, 
    enableMonitoring: true, 
    debug: false 
  }
): Promise<ConceptTuple[]> {
  
  const performanceStart = Date.now();
  
  const conceptDocMap: { [concept: string]: { [docId: string]: number } } = {};
  const conceptTotalFreq: { [concept: string]: number } = {};
  
  // Map for quick doc lookup
  const docMap: { [id: string]: DocumentData } = {};
  docs.forEach(doc => { docMap[doc.id] = doc; });
  const totalDocs = docs.length;

  // 1. Count frequency of each concept in each document (unchanged)
  for (const doc of docs) {
    const textLower = doc.text.toLowerCase();
    doc.concepts.forEach(conceptName => {
      const nameLower = conceptName.toLowerCase();
      
      // Count occurrences of conceptName in text
      let count = 0;
      let idx = textLower.indexOf(nameLower);
      while (idx !== -1) {
        count += 1;
        idx = textLower.indexOf(nameLower, idx + 1);
      }
      
      if (!conceptDocMap[conceptName]) {
        conceptDocMap[conceptName] = {};
        conceptTotalFreq[conceptName] = 0;
      }
      
      conceptDocMap[conceptName][doc.id] = count;
      conceptTotalFreq[conceptName] += count;
    });
  }

  // 2. Prepare unique concepts and their embeddings (enhanced with caching)
  const concepts = Object.keys(conceptDocMap);
  const vectors: number[][] = concepts.map(name => embedConcept(name));
  
  // 3. Enhanced clustering with advanced algorithms
  const estimatedK = Math.max(1, Math.round(Math.sqrt(concepts.length / 2)));
  
  const clusteringResult = await enhancedClusterConcepts(vectors, estimatedK, config);
  const labels = clusteringResult.labels;
  const clusteringInfo = clusteringResult.clusteringInfo;

  // 4. Group concepts by cluster (unchanged logic)
  const clusters: { [id: number]: string[] } = {};
  labels.forEach((clusterId, i) => {
    const conceptName = concepts[i];
    if (!clusters[clusterId]) clusters[clusterId] = [];
    clusters[clusterId].push(conceptName);
  });

  const conceptList: ConceptTuple[] = [];
  
  // Precompute maximum frequency for normalization
  const maxFrequency = Math.max(...Object.values(conceptTotalFreq));
  
  // 5. Build ConceptTuples with enhanced cluster information
  Object.entries(clusters).forEach(([clusterIdStr, members]) => {
    const clusterId = Number(clusterIdStr);
    
    // Determine representative name (unchanged logic)
    let repName = members[0];
    let maxFreq = 0;
    for (const name of members) {
      if (conceptTotalFreq[name] > maxFreq) {
        maxFreq = conceptTotalFreq[name];
        repName = name;
      }
    }
    
    // Calculate aggregated metrics for this cluster (unchanged logic)
    const clusterDocs: { [docId: string]: number } = {};
    members.forEach(name => {
      for (const [docId, freq] of Object.entries(conceptDocMap[name])) {
        clusterDocs[docId] = (clusterDocs[docId] || 0) + freq;
      }
    });
    
    const clusterTotalFreq = Object.values(clusterDocs).reduce((a, b) => a + b, 0);
    const clusterDocCount = Object.keys(clusterDocs).length;
    
    // Domain salience: fraction of documents that contain this concept cluster
    const domainSalience = clusterDocCount / totalDocs;
    
    // Narrative centrality: fraction of those docs where concept appears in summary
    let summaryHitCount = 0;
    for (const docId of Object.keys(clusterDocs)) {
      const summaryLower = docMap[docId].summary.toLowerCase();
      
      // If any synonym appears in the summary text, count as hit
      for (const synonym of members) {
        if (summaryLower.includes(synonym.toLowerCase())) {
          summaryHitCount += 1;
          break;
        }
      }
    }
    
    const centrality = clusterDocCount > 0 ? (summaryHitCount / clusterDocCount) : 0;
    
    // Model confidence: (enhanced - could incorporate clustering quality)
    const baseConfidence = 1.0;
    const clusteringQuality = clusteringInfo.qualityMetrics?.cohesion || 0.5;
    const confidence = baseConfidence * (0.5 + clusteringQuality * 0.5);
    
    // Frequency salience: normalized total frequency
    const freqNorm = maxFrequency > 0 ? (clusterTotalFreq / maxFrequency) : 0;
    const centNorm = centrality; // already 0-1
    const confNorm = confidence; // 0-1
    const domainNorm = domainSalience; // 0-1
    
    // Composite score (your original weighted formula)
    const w = { freq: 0.4, centrality: 0.3, confidence: 0.2, domain: 0.1 };
    const compositeScore = w.freq * freqNorm + w.centrality * centNorm + w.confidence * confNorm + w.domain * domainNorm;
    
    // Build enhanced ConceptTuple
    const concept: ConceptTuple = {
      name: repName,
      clusterId: clusterId,
      score: compositeScore,
      frequency: clusterTotalFreq,
      centrality: centrality,
      confidence: confidence,
      domainSalience: domainSalience,
      originDocs: [],
      mergedFrom: [...members],
      clusterTrace: { 
        algorithm: clusteringInfo.method,
        clusterId: clusterId,
        qualityMetrics: clusteringInfo.qualityMetrics,
        performanceMetrics: clusteringInfo.performanceMetrics
      }
    };
    
    // Populate originDocs with lineage info (unchanged logic)
    for (const [docId, occ] of Object.entries(clusterDocs)) {
      // Find a representative synonym that appears in this doc (for snippet)
      let bestSynonym = members[0];
      let bestCount = 0;
      for (const name of members) {
        const freq = conceptDocMap[name][docId] || 0;
        if (freq > bestCount) {
          bestCount = freq;
          bestSynonym = name;
        }
      }
      
      const docText = docMap[docId].text;
      const idx = docText.toLowerCase().indexOf(bestSynonym.toLowerCase());
      let snippet = '';
      
      if (idx !== -1) {
        const start = Math.max(0, idx - 30);
        const end = Math.min(docText.length, idx + bestSynonym.length + 30);
        snippet = (start > 0 ? '...' : '') + docText.substring(start, end) + (end < docText.length ? '...' : '');
      }
      
      concept.originDocs.push({
        docId: docId,
        occurrences: occ,
        sample: snippet || docMap[docId].summary.slice(0, 60) + '...'
      });
    }
    
    // Enhanced score evolution tracking
    if (config.debug) {
      const freqComponent = w.freq * freqNorm;
      const centComponent = w.centrality * centNorm;
      const confComponent = w.confidence * confNorm;
      const domainComponent = w.domain * domainNorm;
      
      concept.scoreEvolution = [
        { stage: 'frequency', score: freqComponent },
        { stage: ' + centrality', score: freqComponent + centComponent },
        { stage: ' + confidence', score: freqComponent + centComponent + confComponent },
        { stage: ' + domain', score: compositeScore }
      ];
    }
    
    conceptList.push(concept);
  });

  // Sort concepts by descending score for final ranking (unchanged)
  conceptList.sort((a, b) => b.score - a.score);

  // Enhanced debug output with clustering information
  if (config.debug) {
    const debugOutput = {
      concepts: conceptList,
      clusteringInfo: clusteringInfo,
      performance: {
        totalTime: Date.now() - performanceStart,
        conceptCount: concepts.length,
        clusterCount: Object.keys(clusters).length,
        method: clusteringInfo.method
      },
      config: config
    };
    
    fs.writeFileSync('concept_lineage_debug_enhanced.json', JSON.stringify(debugOutput, null, 2));
    
    console.log(`🔍 Enhanced clustering completed:`);
    console.log(`   Method: ${clusteringInfo.method}`);
    console.log(`   Concepts: ${concepts.length} → ${Object.keys(clusters).length} clusters`);
    console.log(`   Total time: ${Date.now() - performanceStart}ms`);
    
    if (clusteringInfo.qualityMetrics) {
      console.log(`   Quality metrics:`, clusteringInfo.qualityMetrics);
    }
  }
  
  return conceptList;
}

// Utility function for clustering health check
export async function checkClusteringHealth(): Promise<{
  status: 'healthy' | 'degraded' | 'unhealthy';
  details: any;
}> {
  try {
    const bridge = new ConceptClusteringBridge();
    const healthResult = await bridge.healthCheck();
    
    return {
      status: healthResult.status === 'healthy' ? 'healthy' : 'unhealthy',
      details: healthResult
    };
  } catch (error) {
    return {
      status: 'unhealthy',
      details: { error: error.message }
    };
  }
}

// Utility function to get clustering performance insights
export async function getClusteringInsights(): Promise<any> {
  try {
    const bridge = new ConceptClusteringBridge();
    return bridge.getPerformanceStats();
  } catch (error) {
    return { error: error.message };
  }
}

// Backward compatibility - export the original function name
export { refineConcepts as refineConceptsEnhanced };

// For drop-in replacement, also export as default
export default refineConcepts;
