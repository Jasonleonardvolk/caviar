// Example TypeScript integration with TORI Clustering System
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
