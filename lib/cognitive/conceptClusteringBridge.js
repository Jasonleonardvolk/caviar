// conceptClusteringBridge.js
// Bridge between TypeScript and Python enhanced clustering for TORI

const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

class ConceptClusteringBridge {
    constructor(pythonPath = 'python', clusteringDir = null) {
        this.pythonPath = pythonPath;
        this.clusteringDir = clusteringDir || path.join(__dirname, '../ingest_pdf');
        this.tempDir = path.join(__dirname, 'temp');
        
        // Ensure temp directory exists
        if (!fs.existsSync(this.tempDir)) {
            fs.mkdirSync(this.tempDir, { recursive: true });
        }
        
        // Performance tracking
        this.clusteringHistory = [];
    }
    
    /**
     * Enhanced clustering function - drop-in replacement for clusterConcepts()
     * @param {number[][]} vectors - Array of concept embedding vectors
     * @param {number} kEstimate - Estimated number of clusters
     * @param {object} options - Additional options
     * @returns {Promise<number[]>} - Array of cluster labels
     */
    async enhancedClusterConcepts(vectors, kEstimate, options = {}) {
        const startTime = Date.now();
        
        try {
            return await this._runClusteringProcess(vectors, kEstimate, options);
        } catch (error) {
            console.warn(`Enhanced clustering failed: ${error.message}, falling back to basic K-means`);
            return this._fallbackKMeans(vectors, kEstimate);
        } finally {
            const runtime = Date.now() - startTime;
            this._trackPerformance('enhanced_clustering', vectors.length, runtime);
        }
    }
    
    /**
     * Benchmark multiple clustering methods
     * @param {number[][]} vectors - Array of concept embedding vectors
     * @param {number} kEstimate - Estimated number of clusters
     * @returns {Promise<object>} - Detailed benchmark results
     */
    async benchmarkClustering(vectors, kEstimate) {
        const startTime = Date.now();
        
        try {
            const inputFile = path.join(this.tempDir, `benchmark_input_${Date.now()}.json`);
            const outputFile = path.join(this.tempDir, `benchmark_output_${Date.now()}.json`);
            
            const inputData = {
                vectors,
                k_estimate: kEstimate,
                enable_benchmarking: true
            };
            
            fs.writeFileSync(inputFile, JSON.stringify(inputData));
            
            const pythonScript = this._createBenchmarkScript(inputFile, outputFile);
            const scriptFile = path.join(this.tempDir, `benchmark_script_${Date.now()}.py`);
            fs.writeFileSync(scriptFile, pythonScript);
            
            await this._executePythonScript(scriptFile);
            
            const result = JSON.parse(fs.readFileSync(outputFile, 'utf-8'));
            
            // Cleanup
            [inputFile, outputFile, scriptFile].forEach(file => {
                if (fs.existsSync(file)) fs.unlinkSync(file);
            });
            
            const runtime = Date.now() - startTime;
            this._trackPerformance('benchmark_clustering', vectors.length, runtime);
            
            return result;
            
        } catch (error) {
            console.error(`Benchmark clustering failed: ${error.message}`);
            // Return basic result
            const labels = await this.enhancedClusterConcepts(vectors, kEstimate);
            return {
                labels,
                method_used: 'fallback',
                error: error.message
            };
        }
    }
    
    async _runClusteringProcess(vectors, kEstimate, options) {
        const inputFile = path.join(this.tempDir, `clustering_input_${Date.now()}.json`);
        const outputFile = path.join(this.tempDir, `clustering_output_${Date.now()}.json`);
        
        const inputData = {
            vectors,
            k_estimate: kEstimate,
            method: options.method || 'auto',
            enable_benchmarking: options.enableBenchmarking || false
        };
        
        fs.writeFileSync(inputFile, JSON.stringify(inputData));
        
        const pythonScript = this._createClusteringScript(inputFile, outputFile);
        const scriptFile = path.join(this.tempDir, `clustering_script_${Date.now()}.py`);
        fs.writeFileSync(scriptFile, pythonScript);
        
        await this._executePythonScript(scriptFile);
        
        if (!fs.existsSync(outputFile)) {
            throw new Error('Clustering output file not generated');
        }
        
        const result = JSON.parse(fs.readFileSync(outputFile, 'utf-8'));
        
        // Cleanup
        [inputFile, outputFile, scriptFile].forEach(file => {
            if (fs.existsSync(file)) fs.unlinkSync(file);
        });
        
        if (result.error) {
            throw new Error(result.error);
        }
        
        return result.labels;
    }
    
    _createClusteringScript(inputFile, outputFile) {
        return `
import json
import sys
import os
sys.path.append('${this.clusteringDir.replace(/\\/g, '/')}')

try:
    from concept_scoring_integration import enhanced_cluster_concepts
    
    with open('${inputFile.replace(/\\/g, '/')}', 'r') as f:
        data = json.load(f)
    
    labels = enhanced_cluster_concepts(
        data['vectors'], 
        data['k_estimate']
    )
    
    result = {'labels': labels}
    
    with open('${outputFile.replace(/\\/g, '/')}', 'w') as f:
        json.dump(result, f)
        
except Exception as e:
    result = {'error': str(e)}
    with open('${outputFile.replace(/\\/g, '/')}', 'w') as f:
        json.dump(result, f)
`;
    }
    
    _createBenchmarkScript(inputFile, outputFile) {
        return `
import json
import sys
sys.path.append('${this.clusteringDir.replace(/\\/g, '/')}')

try:
    from concept_scoring_integration import benchmark_concept_clustering
    
    with open('${inputFile.replace(/\\/g, '/')}', 'r') as f:
        data = json.load(f)
    
    result = benchmark_concept_clustering(
        data['vectors'], 
        data['k_estimate']
    )
    
    with open('${outputFile.replace(/\\/g, '/')}', 'w') as f:
        json.dump(result, f)
        
except Exception as e:
    result = {'error': str(e)}
    with open('${outputFile.replace(/\\/g, '/')}', 'w') as f:
        json.dump(result, f)
`;
    }
    
    async _executePythonScript(scriptFile) {
        return new Promise((resolve, reject) => {
            const pythonProcess = spawn(this.pythonPath, [scriptFile], {
                stdio: ['pipe', 'pipe', 'pipe'],
                cwd: this.clusteringDir
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
                if (code !== 0) {
                    reject(new Error(`Python process failed with code ${code}: ${error}`));
                } else {
                    resolve(output);
                }
            });
            
            pythonProcess.on('error', (err) => {
                reject(new Error(`Failed to start Python process: ${err.message}`));
            });
        });
    }
    
    _fallbackKMeans(vectors, k) {
        // Simple JavaScript K-means implementation as ultimate fallback
        const n = vectors.length;
        const dim = vectors[0].length;
        
        // Initialize centroids randomly
        const centroids = [];
        for (let i = 0; i < k; i++) {
            centroids.push(vectors[Math.floor(Math.random() * n)].slice());
        }
        
        let labels = new Array(n).fill(0);
        
        // K-means iterations
        for (let iter = 0; iter < 50; iter++) {
            let changed = false;
            
            // Assign points to nearest centroid
            for (let i = 0; i < n; i++) {
                let minDist = Infinity;
                let bestCluster = 0;
                
                for (let j = 0; j < k; j++) {
                    let dist = 0;
                    for (let d = 0; d < dim; d++) {
                        const diff = vectors[i][d] - centroids[j][d];
                        dist += diff * diff;
                    }
                    
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
            
            // Update centroids
            const newCentroids = Array.from({ length: k }, () => new Array(dim).fill(0));
            const counts = new Array(k).fill(0);
            
            for (let i = 0; i < n; i++) {
                const cluster = labels[i];
                counts[cluster]++;
                for (let d = 0; d < dim; d++) {
                    newCentroids[cluster][d] += vectors[i][d];
                }
            }
            
            for (let j = 0; j < k; j++) {
                if (counts[j] > 0) {
                    for (let d = 0; d < dim; d++) {
                        newCentroids[j][d] /= counts[j];
                        centroids[j][d] = newCentroids[j][d];
                    }
                }
            }
        }
        
        return labels;
    }
    
    _trackPerformance(operation, conceptCount, runtime) {
        this.clusteringHistory.push({
            timestamp: Date.now(),
            operation,
            conceptCount,
            runtime
        });
        
        // Keep only recent history
        if (this.clusteringHistory.length > 100) {
            this.clusteringHistory = this.clusteringHistory.slice(-100);
        }
    }
    
    getPerformanceStats() {
        if (this.clusteringHistory.length === 0) {
            return { error: 'No clustering history available' };
        }
        
        const recent = this.clusteringHistory.slice(-10);
        const avgRuntime = recent.reduce((sum, h) => sum + h.runtime, 0) / recent.length;
        const avgConcepts = recent.reduce((sum, h) => sum + h.conceptCount, 0) / recent.length;
        
        return {
            totalRuns: this.clusteringHistory.length,
            recentRuns: recent.length,
            avgRuntime: Math.round(avgRuntime),
            avgConcepts: Math.round(avgConcepts),
            lastRun: recent[recent.length - 1] || null
        };
    }
    
    /**
     * Health check - verify the clustering system is working
     * @returns {Promise<object>} - Health status
     */
    async healthCheck() {
        try {
            // Test with small dataset
            const testVectors = [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
                [0.2, 0.3, 0.4]
            ];
            
            const labels = await this.enhancedClusterConcepts(testVectors, 2);
            
            return {
                status: 'healthy',
                message: 'Enhanced clustering system is operational',
                testResult: {
                    inputSize: testVectors.length,
                    clusters: new Set(labels).size,
                    labels: labels
                },
                performance: this.getPerformanceStats()
            };
            
        } catch (error) {
            return {
                status: 'unhealthy',
                message: `Enhanced clustering system has issues: ${error.message}`,
                error: error.message
            };
        }
    }
}

module.exports = { ConceptClusteringBridge };

// Usage example for integration with conceptScoring.ts:
/*
const { ConceptClusteringBridge } = require('./conceptClusteringBridge');
const bridge = new ConceptClusteringBridge();

// In your refineConcepts function, replace:
// const labels = clusterConcepts(vectors, estimatedK);

// With:
// const labels = await bridge.enhancedClusterConcepts(vectors, estimatedK);

// For benchmarking (optional):
// const benchmarkResult = await bridge.benchmarkClustering(vectors, estimatedK);
// console.log('Best method:', benchmarkResult.method_used);
*/
