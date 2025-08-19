// ingest_pdf/clusterBenchmark.ts
import { performance } from 'perf_hooks';
import { spawn } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';

// Distance function (Euclidean)
function euclideanDist(a: number[], b: number[]): number {
  let sumSq = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = a[i] - b[i];
    sumSq += diff * diff;
  }
  return Math.sqrt(sumSq);
}

// Cosine similarity function
function cosineSimilarity(a: number[], b: number[]): number {
  const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0);
  const normA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
  const normB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
  return dotProduct / (normA * normB + 1e-8);
}

// K-Means clustering
interface ClusteringResult {
  labels: number[];
  clusters: { [clusterId: number]: number[] };
  cohesionScores: { [clusterId: number]: number };
  runtime: number;
  method: string;
  nClusters: number;
  avgCohesion: number;
  centroids?: number[][];
  iterations?: number;
  convergenceStep?: number;
  totalSteps?: number;
  phaseVariance?: number;
  singletonMerges?: number;
  orphanReassignments?: number;
  removedLowCohesion?: number;
  silhouetteScore?: number;
}

function kMeansCluster(data: number[][], k: number, maxIter: number = 100): ClusteringResult {
  const start = performance.now();
  const n = data.length;
  
  // Randomly initialize centroids from data points
  const centroids: number[][] = [];
  const usedIndices = new Set<number>();
  while (centroids.length < k) {
    const idx = Math.floor(Math.random() * n);
    if (!usedIndices.has(idx)) {
      centroids.push([...data[idx]]);
      usedIndices.add(idx);
    }
  }

  let labels = new Array(n).fill(-1);
  let iterations = 0;
  
  for (let iter = 0; iter < maxIter; iter++) {
    iterations = iter + 1;
    let changed = false;
    // Assign points to nearest centroid
    for (let i = 0; i < n; i++) {
      let minDist = Infinity;
      let bestCluster = -1;
      for (let j = 0; j < centroids.length; j++) {
        const dist = euclideanDist(data[i], centroids[j]);
        if (dist < minDist) {
          minDist = dist;
          bestCluster = j;
        }
      }
      if (labels[i] !== bestCluster) {
        labels[i] = bestCluster;
        changed = true;
      }
    }
    if (!changed) break;
    
    // Recompute centroids as mean of points in each cluster
    const newCentroids: number[][] = Array.from({ length: k }, () => new Array(data[0].length).fill(0));
    const counts = new Array(k).fill(0);
    for (let i = 0; i < n; i++) {
      const cluster = labels[i];
      counts[cluster] += 1;
      for (let d = 0; d < data[i].length; d++) {
        newCentroids[cluster][d] += data[i][d];
      }
    }
    for (let j = 0; j < k; j++) {
      if (counts[j] > 0) {
        for (let d = 0; d < newCentroids[j].length; d++) {
          newCentroids[j][d] /= counts[j];
        }
        centroids[j] = newCentroids[j];
      } else {
        // If a cluster lost all points, reinitialize it to a random point
        const idx = Math.floor(Math.random() * n);
        centroids[j] = [...data[idx]];
      }
    }
  }
  
  // Build clusters and calculate cohesion scores
  const clusters: { [clusterId: number]: number[] } = {};
  const cohesionScores: { [clusterId: number]: number } = {};
  
  for (let i = 0; i < k; i++) {
    clusters[i] = [];
  }
  
  for (let i = 0; i < n; i++) {
    clusters[labels[i]].push(i);
  }
  
  for (const [clusterId, members] of Object.entries(clusters)) {
    if (members.length > 0) {
      cohesionScores[parseInt(clusterId)] = calculateClusterCohesion(data, members);
    }
  }
  
  const runtime = performance.now() - start;
  const avgCohesion = Object.values(cohesionScores).reduce((sum, val) => sum + val, 0) / Object.keys(cohesionScores).length;
  
  return {
    labels,
    clusters,
    cohesionScores,
    runtime,
    method: 'kmeans',
    nClusters: k,
    avgCohesion,
    centroids,
    iterations
  };
}

function calculateClusterCohesion(data: number[][], memberIndices: number[]): number {
  if (memberIndices.length < 2) {
    return 0.0;
  }
  
  let totalSimilarity = 0;
  let pairCount = 0;
  
  for (let i = 0; i < memberIndices.length; i++) {
    for (let j = i + 1; j < memberIndices.length; j++) {
      const sim = cosineSimilarity(data[memberIndices[i]], data[memberIndices[j]]);
      totalSimilarity += sim;
      pairCount++;
    }
  }
  
  return pairCount > 0 ? totalSimilarity / pairCount : 0.0;
}

// Compute silhouette score and cohesion for given labels
function computeClusteringMetrics(data: number[][], labels: number[]): { silhouette: number, cohesion: number } {
  const n = data.length;
  const clusterMap: { [cluster: number]: number[] } = {};
  
  for (let i = 0; i < n; i++) {
    const cluster = labels[i];
    if (!clusterMap[cluster]) clusterMap[cluster] = [];
    clusterMap[cluster].push(i);
  }
  
  let totalSilhouette = 0;
  let totalA = 0;
  
  for (let i = 0; i < n; i++) {
    const cluster = labels[i];
    const pointsInCluster = clusterMap[cluster];
    
    // Compute a = mean distance to other points in same cluster
    let a = 0;
    if (pointsInCluster.length > 1) {
      for (const j of pointsInCluster) {
        if (j === i) continue;
        a += euclideanDist(data[i], data[j]);
      }
      a /= (pointsInCluster.length - 1);
    } else {
      a = 0;
    }
    
    // Compute b = minimum mean distance to points in another cluster
    let b = Infinity;
    for (const otherCluster in clusterMap) {
      if (Number(otherCluster) === cluster) continue;
      const indices = clusterMap[otherCluster];
      let distSum = 0;
      for (const j of indices) {
        distSum += euclideanDist(data[i], data[j]);
      }
      const meanDist = distSum / indices.length;
      if (meanDist < b) {
        b = meanDist;
      }
    }
    
    const s = (b - a) / (a < b ? b : a);  // silhouette for point i
    totalSilhouette += s;
    totalA += a;
  }
  
  const avgSilhouette = totalSilhouette / n;
  const avgA = totalA / n;  // average intra-cluster distance (cohesion measure)
  return { silhouette: avgSilhouette, cohesion: avgA };
}

// Python bridge for oscillator clustering
async function runOscillatorClustering(vectors: number[][], options: any = {}): Promise<ClusteringResult> {
  return new Promise((resolve, reject) => {
    const start = performance.now();
    
    // Create temporary files for data exchange
    const tempDir = path.join(__dirname, 'temp');
    if (!fs.existsSync(tempDir)) {
      fs.mkdirSync(tempDir, { recursive: true });
    }
    
    const inputFile = path.join(tempDir, `clustering_input_${Date.now()}.json`);
    const outputFile = path.join(tempDir, `clustering_output_${Date.now()}.json`);
    
    const inputData = {
      vectors,
      options: {
        steps: options.steps || 60,
        tol: options.tol || 1e-3,
        cohesion_threshold: options.cohesion_threshold || 0.15,
        enable_logging: options.enable_logging || false
      }
    };
    
    fs.writeFileSync(inputFile, JSON.stringify(inputData));
    
    // Python script to run oscillator clustering
    const pythonScript = `
import json
import sys
import numpy as np
sys.path.append('${__dirname}')
try:
    from clustering import run_oscillator_clustering_with_metrics
    
    with open('${inputFile}', 'r') as f:
        data = json.load(f)
    
    vectors = np.array(data['vectors'])
    options = data['options']
    
    result = run_oscillator_clustering_with_metrics(
        vectors,
        steps=options['steps'],
        tol=options['tol'],
        cohesion_threshold=options['cohesion_threshold'],
        enable_logging=options['enable_logging']
    )
    
    # Convert numpy arrays to lists for JSON serialization
    for key, value in result.items():
        if hasattr(value, 'tolist'):
            result[key] = value.tolist()
    
    with open('${outputFile}', 'w') as f:
        json.dump(result, f)
        
except Exception as e:
    error_result = {'error': str(e)}
    with open('${outputFile}', 'w') as f:
        json.dump(error_result, f)
`;

    // Write and execute Python script
    const scriptFile = path.join(tempDir, `clustering_script_${Date.now()}.py`);
    fs.writeFileSync(scriptFile, pythonScript);
    
    const pythonProcess = spawn('python', [scriptFile], { 
      stdio: ['pipe', 'pipe', 'pipe'],
      cwd: __dirname 
    });
    
    let output = '';
    let error = '';
    
    pythonProcess.stdout.on('data', (data) => {
      output += data.toString();
    });
    
    pythonProcess.stderr.on('data', (data) => {
      error += data.toString();
    });
    
    pythonProcess.on('close', (code) => {
      try {
        // Clean up temporary files
        [inputFile, outputFile, scriptFile].forEach(file => {
          if (fs.existsSync(file)) {
            fs.unlinkSync(file);
          }
        });
        
        if (code !== 0) {
          reject(new Error(`Python process exited with code ${code}: ${error}`));
          return;
        }
        
        if (!fs.existsSync(outputFile)) {
          reject(new Error('No output file generated'));
          return;
        }
        
        const result = JSON.parse(fs.readFileSync(outputFile, 'utf-8'));
        
        if (result.error) {
          reject(new Error(result.error));
          return;
        }
        
        // Format result to match ClusteringResult interface
        const formattedResult: ClusteringResult = {
          labels: result.labels,
          clusters: result.clusters,
          cohesionScores: result.cohesion_scores,
          runtime: result.runtime,
          method: 'oscillator',
          nClusters: result.n_clusters,
          avgCohesion: result.avg_cohesion,
          convergenceStep: result.convergence_step,
          totalSteps: result.total_steps,
          phaseVariance: result.phase_variance,
          singletonMerges: result.singleton_merges,
          orphanReassignments: result.orphan_reassignments,
          removedLowCohesion: result.removed_low_cohesion
        };
        
        resolve(formattedResult);
        
      } catch (e) {
        reject(new Error(`Failed to parse Python output: ${e}`));
      }
    });
    
    pythonProcess.on('error', (err) => {
      reject(new Error(`Failed to start Python process: ${err}`));
    });
  });
}

// Python bridge for all clustering methods
async function runPythonClusteringBenchmark(vectors: number[][], methods: string[] = ['oscillator', 'kmeans', 'hdbscan']): Promise<{ [method: string]: ClusteringResult }> {
  return new Promise((resolve, reject) => {
    const tempDir = path.join(__dirname, 'temp');
    if (!fs.existsSync(tempDir)) {
      fs.mkdirSync(tempDir, { recursive: true });
    }
    
    const inputFile = path.join(tempDir, `benchmark_input_${Date.now()}.json`);
    const outputFile = path.join(tempDir, `benchmark_output_${Date.now()}.json`);
    
    const inputData = { vectors, methods };
    fs.writeFileSync(inputFile, JSON.stringify(inputData));
    
    const pythonScript = `
import json
import sys
import numpy as np
sys.path.append('${__dirname}')
try:
    from clustering_enhanced import benchmark_all_clustering_methods
    
    with open('${inputFile}', 'r') as f:
        data = json.load(f)
    
    vectors = np.array(data['vectors'])
    methods = data['methods']
    
    results = benchmark_all_clustering_methods(vectors, methods, enable_logging=False)
    
    # Convert numpy arrays to lists for JSON serialization
    for method_name, result in results.items():
        if 'error' not in result:
            for key, value in result.items():
                if hasattr(value, 'tolist'):
                    result[key] = value.tolist()
    
    with open('${outputFile}', 'w') as f:
        json.dump(results, f)
        
except Exception as e:
    error_result = {'error': str(e)}
    with open('${outputFile}', 'w') as f:
        json.dump(error_result, f)
`;

    const scriptFile = path.join(tempDir, `benchmark_script_${Date.now()}.py`);
    fs.writeFileSync(scriptFile, pythonScript);
    
    const pythonProcess = spawn('python', [scriptFile], { 
      stdio: ['pipe', 'pipe', 'pipe'],
      cwd: __dirname 
    });
    
    pythonProcess.on('close', (code) => {
      try {
        // Clean up temporary files
        [inputFile, outputFile, scriptFile].forEach(file => {
          if (fs.existsSync(file)) {
            fs.unlinkSync(file);
          }
        });
        
        if (code !== 0) {
          reject(new Error(`Python process exited with code ${code}`));
          return;
        }
        
        const results = JSON.parse(fs.readFileSync(outputFile, 'utf-8'));
        
        if (results.error) {
          reject(new Error(results.error));
          return;
        }
        
        // Format results to match ClusteringResult interface
        const formattedResults: { [method: string]: ClusteringResult } = {};
        
        for (const [method, result] of Object.entries(results)) {
          const typedResult = result as any;
          if (typedResult.error) {
            console.error(`Error in ${method}: ${typedResult.error}`);
            continue;
          }
          
          formattedResults[method] = {
            labels: typedResult.labels,
            clusters: typedResult.clusters,
            cohesionScores: typedResult.cohesion_scores,
            runtime: typedResult.runtime,
            method: method,
            nClusters: typedResult.n_clusters,
            avgCohesion: typedResult.avg_cohesion,
            convergenceStep: typedResult.convergence_step,
            totalSteps: typedResult.total_steps,
            phaseVariance: typedResult.phase_variance,
            singletonMerges: typedResult.singleton_merges,
            orphanReassignments: typedResult.orphan_reassignments,
            removedLowCohesion: typedResult.removed_low_cohesion
          };
        }
        
        resolve(formattedResults);
        
      } catch (e) {
        reject(new Error(`Failed to parse Python output: ${e}`));
      }
    });
  });
}

// Enhanced benchmarking function
export async function benchmarkClustering(vectors: number[][], methods: string[] = ['oscillator', 'kmeans']): Promise<{ [method: string]: ClusteringResult }> {
  const results: { [method: string]: ClusteringResult } = {};
  
  for (const method of methods) {
    try {
      const start = performance.now();
      let result: ClusteringResult;
      
      if (method === 'oscillator') {
        result = await runOscillatorClustering(vectors);
      } else if (method === 'kmeans') {
        const k = Math.round(Math.sqrt(vectors.length / 2)) || 1;
        result = kMeansCluster(vectors, k);
      } else if (method === 'python_benchmark') {
        // Run all Python methods
        const pythonResults = await runPythonClusteringBenchmark(vectors, ['oscillator', 'kmeans', 'hdbscan']);
        Object.assign(results, pythonResults);
        continue;
      } else {
        console.log(`Unknown clustering method: ${method}`);
        continue;
      }
      
      // Add silhouette score
      const metrics = computeClusteringMetrics(vectors, result.labels);
      result.silhouetteScore = metrics.silhouette;
      
      results[method] = result;
      
    } catch (error) {
      console.error(`Error running ${method}: ${error}`);
      results[method] = {
        labels: [],
        clusters: {},
        cohesionScores: {},
        runtime: 0,
        method: method,
        nClusters: 0,
        avgCohesion: 0,
        error: error.toString()
      } as any;
    }
  }
  
  return results;
}

// Detailed benchmark report
export function generateBenchmarkReport(results: { [method: string]: ClusteringResult }): string {
  let report = '\n=== CLUSTERING BENCHMARK REPORT ===\n\n';
  
  const methods = Object.keys(results);
  if (methods.length === 0) {
    return report + 'No results to report.\n';
  }
  
  // Summary table
  report += 'Method'.padEnd(20) + 'Clusters'.padEnd(12) + 'Cohesion'.padEnd(12) + 'Silhouette'.padEnd(12) + 'Runtime (ms)'.padEnd(15) + 'Notes\n';
  report += '-'.repeat(85) + '\n';
  
  for (const method of methods) {
    const result = results[method];
    if ('error' in result) {
      report += `${method.padEnd(20)}ERROR: ${result.error}\n`;
      continue;
    }
    
    const clusters = result.nClusters.toString().padEnd(12);
    const cohesion = result.avgCohesion.toFixed(3).padEnd(12);
    const silhouette = (result.silhouetteScore || 0).toFixed(3).padEnd(12);
    const runtime = result.runtime.toFixed(1).padEnd(15);
    
    let notes = '';
    if (method === 'oscillator') {
      notes = `conv: ${result.convergenceStep}/${result.totalSteps}`;
      if (result.singletonMerges) notes += `, merges: ${result.singletonMerges}`;
      if (result.orphanReassignments) notes += `, orphans: ${result.orphanReassignments}`;
    } else if (method === 'kmeans' && result.iterations) {
      notes = `iters: ${result.iterations}`;
    }
    
    report += `${method.padEnd(20)}${clusters}${cohesion}${silhouette}${runtime}${notes}\n`;
  }
  
  // Detailed analysis
  report += '\n=== DETAILED ANALYSIS ===\n\n';
  
  // Best performer by metric
  const validResults = Object.entries(results).filter(([, r]) => !('error' in r)) as [string, ClusteringResult][];
  
  if (validResults.length > 0) {
    const bestCohesion = validResults.reduce((a, b) => a[1].avgCohesion > b[1].avgCohesion ? a : b);
    const bestSilhouette = validResults.reduce((a, b) => (a[1].silhouetteScore || 0) > (b[1].silhouetteScore || 0) ? a : b);
    const fastest = validResults.reduce((a, b) => a[1].runtime < b[1].runtime ? a : b);
    
    report += `Best Cohesion: ${bestCohesion[0]} (${bestCohesion[1].avgCohesion.toFixed(3)})\n`;
    report += `Best Silhouette: ${bestSilhouette[0]} (${(bestSilhouette[1].silhouetteScore || 0).toFixed(3)})\n`;
    report += `Fastest: ${fastest[0]} (${fastest[1].runtime.toFixed(1)}ms)\n\n`;
  }
  
  // Method-specific details
  for (const [method, result] of Object.entries(results)) {
    if ('error' in result) continue;
    
    report += `${method.toUpperCase()} Details:\n`;
    report += `  Clusters: ${result.nClusters}\n`;
    report += `  Avg Cohesion: ${result.avgCohesion.toFixed(3)}\n`;
    report += `  Runtime: ${result.runtime.toFixed(1)}ms\n`;
    
    if (method === 'oscillator') {
      report += `  Convergence: ${result.convergenceStep}/${result.totalSteps} steps\n`;
      report += `  Phase Variance: ${(result.phaseVariance || 0).toFixed(3)}\n`;
      report += `  Singleton Merges: ${result.singletonMerges || 0}\n`;
      report += `  Orphan Reassignments: ${result.orphanReassignments || 0}\n`;
      report += `  Low Cohesion Removed: ${result.removedLowCohesion || 0}\n`;
    }
    
    // Cluster size distribution
    const clusterSizes = Object.values(result.clusters).map(cluster => cluster.length);
    if (clusterSizes.length > 0) {
      const minSize = Math.min(...clusterSizes);
      const maxSize = Math.max(...clusterSizes);
      const avgSize = clusterSizes.reduce((sum, size) => sum + size, 0) / clusterSizes.length;
      report += `  Cluster sizes: min=${minSize}, max=${maxSize}, avg=${avgSize.toFixed(1)}\n`;
    }
    
    report += '\n';
  }
  
  return report;
}

// Legacy function for backward compatibility
export function benchmarkClusteringSync(vectors: number[][], methods: string[]): void {
  methods.forEach(method => {
    const start = performance.now();
    let labels: number[] = [];
    
    if (method === 'kmeans') {
      const k = Math.round(Math.sqrt(vectors.length / 2)) || 1;
      const { labels: kmLabels } = kMeansCluster(vectors, k);
      labels = kmLabels;
    } else if (method === 'hdbscan') {
      console.log('HDBSCAN clustering requires Python integration - use benchmarkClustering() instead.');
      return;
    } else if (method === 'affinity') {
      console.log('Affinity Propagation clustering requires Python integration - use benchmarkClustering() instead.');
      return;
    } else if (method === 'oscillator') {
      console.log('Oscillator clustering requires Python integration - use benchmarkClustering() instead.');
      return;
    } else {
      console.log(`Unknown clustering method: ${method}`);
      return;
    }
    
    const end = performance.now();
    const { silhouette, cohesion } = computeClusteringMetrics(vectors, labels);
    const clusterCount = new Set(labels).size;
    console.log(`${method}: clusters=${clusterCount}, silhouette=${silhouette.toFixed(3)}, cohesion=${cohesion.toFixed(3)}, runtime=${(end - start).toFixed(1)} ms`);
  });
}
