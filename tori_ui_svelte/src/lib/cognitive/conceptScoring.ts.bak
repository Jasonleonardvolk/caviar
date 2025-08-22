// src/lib/cognitive/conceptScoring.ts
import * as fs from 'fs';

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
  clusterTrace?: { algorithm: string; clusterId: number };
}

// Placeholder for concept embedding - in practice, use a language model or API
function embedConcept(name: string): number[] {
  // e.g., call external API or model to get embedding vector for concept name
  // For demonstration, return a dummy vector (not meaningful)
  const vec = new Array(10).fill(0).map((_, i) => (name.charCodeAt(i % name.length) || 0) / 255);
  return vec;
}

// Simple KMeans clustering (using Euclidean distance) for concept vectors
function clusterConcepts(vectors: number[][], k: number): number[] {
  const n = vectors.length;
  const dim = vectors[0].length;
  // initialize centroids as first k vectors (or random selection)
  const centroids = vectors.slice(0, k).map(v => v.slice());
  const labels = new Array(n).fill(-1);
  for (let iter = 0; iter < 100; iter++) {
    let changed = false;
    // assign each vector to nearest centroid
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
    // recompute centroids
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

export function refineConcepts(docs: DocumentData[], debug: boolean = false): ConceptTuple[] {
  const conceptDocMap: { [concept: string]: { [docId: string]: number } } = {};
  const conceptTotalFreq: { [concept: string]: number } = {};
  // Map for quick doc lookup
  const docMap: { [id: string]: DocumentData } = {};
  docs.forEach(doc => { docMap[doc.id] = doc; });
  const totalDocs = docs.length;

  // 1. Count frequency of each concept in each document
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

  // 2. Prepare unique concepts and their embeddings
  const concepts = Object.keys(conceptDocMap);
  const vectors: number[][] = concepts.map(name => embedConcept(name));
  // If using a better clustering algorithm like HDBSCAN or Affinity Propagation:
  // integrate here (e.g., call external library to get cluster labels).
  // For now, using KMeans with estimated k.
  const estimatedK = Math.max(1, Math.round(Math.sqrt(concepts.length / 2)));
  const labels = clusterConcepts(vectors, estimatedK);

  // 3. Group concepts by cluster
  const clusters: { [id: number]: string[] } = {};
  labels.forEach((clusterId, i) => {
    const conceptName = concepts[i];
    if (!clusters[clusterId]) clusters[clusterId] = [];
    clusters[clusterId].push(conceptName);
  });

  const conceptList: ConceptTuple[] = [];
  // Precompute maximum frequency for normalization
  const maxFrequency = Math.max(...Object.values(conceptTotalFreq));
  let clusterAlgorithm = 'kmeans';  // chosen clustering method name
  Object.entries(clusters).forEach(([clusterIdStr, members]) => {
    const clusterId = Number(clusterIdStr);
    // Determine representative name (e.g., the synonym with highest total frequency)
    let repName = members[0];
    let maxFreq = 0;
    for (const name of members) {
      if (conceptTotalFreq[name] > maxFreq) {
        maxFreq = conceptTotalFreq[name];
        repName = name;
      }
    }
    // Calculate aggregated metrics for this cluster concept
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
    // Model confidence: (placeholder, e.g., could come from extraction process)
    const confidence = 1.0;
    // Frequency salience: normalized total frequency
    const freqNorm = maxFrequency > 0 ? (clusterTotalFreq / maxFrequency) : 0;
    const centNorm = centrality; // already 0-1
    const confNorm = confidence; // assumed 0-1
    const domainNorm = domainSalience; // 0-1
    // Composite score (weighted sum of factors)
    const w = { freq: 0.4, centrality: 0.3, confidence: 0.2, domain: 0.1 };
    const compositeScore = w.freq * freqNorm + w.centrality * centNorm + w.confidence * confNorm + w.domain * domainNorm;
    // Build ConceptTuple
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
      clusterTrace: { algorithm: clusterAlgorithm, clusterId: clusterId }
    };
    // Populate originDocs with lineage info (documents and snippet)
    for (const [docId, occ] of Object.entries(clusterDocs)) {
      // find a representative synonym that appears in this doc (for snippet)
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
    // Optionally track score evolution for debugging
    if (debug) {
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

  // Sort concepts by descending score for final ranking
  conceptList.sort((a, b) => b.score - a.score);

  // If debug, output lineage details to a JSON file
  if (debug) {
    fs.writeFileSync('concept_lineage_debug.json', JSON.stringify(conceptList, null, 2));
  }
  return conceptList;
}
