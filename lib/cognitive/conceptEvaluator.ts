// src/lib/cognitive/conceptEvaluator.ts
import * as fs from 'fs';

interface EvalOptions {
  groundTruthFile: string;
  predictedFile: string;
}

export function evaluateExtraction(options: EvalOptions): void {
  const { groundTruthFile, predictedFile } = options;
  // Load ground truth annotations (JSON)
  const groundTruthData = JSON.parse(fs.readFileSync(groundTruthFile, 'utf-8'));
  const groundTruthMap: { [docId: string]: Set<string> } = {};
  if (Array.isArray(groundTruthData)) {
    groundTruthData.forEach((entry: any, idx: number) => {
      const docId = entry.docId || entry.id || entry.filename || `doc${idx+1}`;
      const concepts = entry.concepts || entry.labels || entry.keyphrases || [];
      groundTruthMap[docId] = new Set(concepts.map((c: string) => c.toLowerCase()));
    });
  } else {
    for (const [docId, concepts] of Object.entries(groundTruthData)) {
      groundTruthMap[docId] = new Set((concepts as string[]).map(c => c.toLowerCase()));
    }
  }
  // Total number of ground truth concepts across all documents
  let totalGroundTruth = 0;
  for (const set of Object.values(groundTruthMap)) {
    totalGroundTruth += set.size;
  }

  // Load predicted concepts (JSON)
  const predictedData = JSON.parse(fs.readFileSync(predictedFile, 'utf-8'));
  const predictedMap: { [docId: string]: { concepts: string[], conceptObjects?: any[], objectMap?: { [name: string]: any } } } = {};
  if (Array.isArray(predictedData) && predictedData.length > 0 && 'clusterId' in predictedData[0]) {
    // Predicted data is an array of ConceptTuple objects (clustered global list)
    predictedData.forEach((conceptObj: any) => {
      if (!conceptObj.originDocs) return;
      conceptObj.originDocs.forEach((origin: any) => {
        const docId = origin.docId;
        if (!predictedMap[docId]) {
          predictedMap[docId] = { concepts: [], conceptObjects: [], objectMap: {} };
        }
        const nameLower = conceptObj.name.toLowerCase();
        predictedMap[docId].concepts.push(nameLower);
        predictedMap[docId].conceptObjects!.push(conceptObj);
        predictedMap[docId].objectMap![nameLower] = conceptObj;
      });
    });
  } else if (Array.isArray(predictedData) && predictedData.length > 0 && ('docId' in predictedData[0] || 'id' in predictedData[0])) {
    // Predicted data is an array of document entries with concept lists
    predictedData.forEach((entry: any) => {
      const docId = entry.docId || entry.id;
      const conceptsList = entry.concepts || entry.predicted || entry.labels || [];
      if (!predictedMap[docId]) {
        predictedMap[docId] = { concepts: [], conceptObjects: [] };
      }
      predictedMap[docId].concepts.push(...conceptsList.map((c: string) => c.toLowerCase()));
    });
  } else if (predictedData && typeof predictedData === 'object') {
    // Predicted data is an object mapping docId -> concept list
    for (const [docId, concepts] of Object.entries(predictedData)) {
      predictedMap[docId] = { concepts: (concepts as string[]).map(c => c.toLowerCase()), conceptObjects: [] };
    }
  }

  // Evaluation metrics
  let totalTP = 0, totalFP = 0, totalFN = 0;
  // For calibration and threshold analysis
  const predictionResults: Array<{ score: number, correct: boolean }> = [];

  // Evaluate each document
  const allDocs = new Set([...Object.keys(groundTruthMap), ...Object.keys(predictedMap)]);
  for (const docId of allDocs) {
    const gtSet = groundTruthMap[docId] || new Set<string>();
    const predInfo = predictedMap[docId] || { concepts: [], conceptObjects: [], objectMap: {} };
    const predConcepts = predInfo.concepts;
    const predObjects = predInfo.objectMap || {};
    const matchedPred = new Set<string>();

    // Match predicted concepts to ground truth
    for (const predName of predConcepts) {
      let matched = false;
      if (gtSet.has(predName)) {
        gtSet.delete(predName);
        matched = true;
      } else {
        // Check synonyms if available
        const conceptObj = predObjects[predName];
        if (conceptObj && conceptObj.mergedFrom) {
          for (const alt of conceptObj.mergedFrom) {
            const altLower = alt.toLowerCase();
            if (gtSet.has(altLower)) {
              gtSet.delete(altLower);
              matched = true;
              break;
            }
          }
        }
      }
      if (matched) {
        totalTP += 1;
        matchedPred.add(predName);
      } else {
        totalFP += 1;
      }
    }
    // Whatever remains in gtSet was not predicted
    totalFN += gtSet.size;

    // Record data for calibration (if scores available)
    if (predInfo.conceptObjects && predInfo.conceptObjects.length) {
      for (const conceptObj of predInfo.conceptObjects) {
        if (typeof conceptObj.score === 'number') {
          const predName = conceptObj.name.toLowerCase();
          const isCorrect = matchedPred.has(predName);
          predictionResults.push({ score: conceptObj.score, correct: isCorrect });
        }
      }
    }
  }

  // Calculate overall precision, recall, F1
  const precision = totalTP / (totalTP + totalFP || 1);
  const recall = totalTP / (totalTP + totalFN || 1);
  const f1 = (precision + recall) ? (2 * precision * recall / (precision + recall)) : 0;
  console.log(`Overall Precision: ${precision.toFixed(3)}, Recall: ${recall.toFixed(3)}, F1: ${f1.toFixed(3)}`);

  // Prepare calibration curve data if we have scores
  if (predictionResults.length > 0) {
    // Normalize scores to [0,1] range
    const maxScore = Math.max(...predictionResults.map(p => p.score));
    const bins = Array.from({ length: 10 }, () => ({ correct: 0, total: 0 }));
    for (const { score, correct } of predictionResults) {
      const normScore = maxScore > 0 ? score / maxScore : 0;
      const binIndex = Math.min(9, Math.floor(normScore * 10));
      bins[binIndex].total += 1;
      if (correct) bins[binIndex].correct += 1;
    }
    let calibrationCSV = "score_range, count, accuracy\n";
    for (let i = 0; i < bins.length; i++) {
      const rangeStart = i * 0.1;
      const rangeEnd = (i === 9) ? 1.0 : (i+1) * 0.1;
      const accuracy = bins[i].total ? (bins[i].correct / bins[i].total) : 0;
      calibrationCSV += `${rangeStart.toFixed(1)}-${rangeEnd.toFixed(1)}, ${bins[i].total}, ${accuracy.toFixed(3)}\n`;
    }
    fs.writeFileSync('calibration_curve.csv', calibrationCSV);
    // Precision/Recall vs threshold
    let prCSV = "threshold, precision, recall, F1\n";
    for (let t = 0; t <= 10; t++) {
      const threshold = t / 10;
      const cutoff = threshold * maxScore;
      let tp_t = 0, fp_t = 0;
      for (const { score, correct } of predictionResults) {
        if (score >= cutoff) {
          if (correct) tp_t += 1;
          else fp_t += 1;
        }
      }
      const fn_t = totalGroundTruth - tp_t;
      const prec_t = tp_t / (tp_t + fp_t || 1);
      const rec_t = tp_t / (tp_t + fn_t || 1);
      const f1_t = (prec_t + rec_t) ? (2 * prec_t * rec_t / (prec_t + rec_t)) : 0;
      prCSV += `${threshold.toFixed(1)}, ${prec_t.toFixed(3)}, ${rec_t.toFixed(3)}, ${f1_t.toFixed(3)}\n`;
    }
    fs.writeFileSync('threshold_metrics.csv', prCSV);
  } else {
    console.warn("No prediction scores available for calibration analysis.");
  }
}
