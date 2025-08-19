# Spectral-Phase Concept Ingestion Pipeline

This pipeline implements ALAN 2.x's symbolic, spectral, and phase-based concept extraction from scientific PDFs. It is modular, interpretable, and produces cognitively meaningful concept nodes for downstream reasoning and visualization.

---

## Advanced Extraction, Scoring, and Validation Plan

### 1. Advanced Keyword Extraction

- **N-gram Extraction:** Extract multi-word noun phrases (e.g., "Koopman Operator Theory") as candidate concept names.
- **Intra-cluster TF-IDF:** Score terms/phrases by their frequency in the cluster vs. the rest of the document to surface unique, topical keywords.
- **Section Header & Formatting Heuristics:** Prefer blocks that are likely section headers, bold, or italicized as candidate titles.
- **Fallback:** If no strong candidate, fall back to the most frequent non-stopword(s) in the cluster.

### 2. Cluster Validation and Merging

- **Singleton Detection:** Identify clusters with only one block (singletons).
- **Merge or Discard:** Optionally merge singleton/small clusters with their nearest larger neighbor, or discard if trivial.
- **Cohesion Threshold:** Discard clusters with low internal similarity/cohesion to avoid trivial or incoherent concepts.

### 3. Resonance and Narrative Scoring

- **Resonance Score:** For each cluster, compute a resonance score based on the self-alignment of its Koopman mode (e.g., autocorrelation or periodicity of the mode signal across the document).
- **Narrative Centrality:** Build a graph of clusters and simulate narrative walks (random or weighted by semantic similarity) to find which concepts are most central to the document's flow.
- **Ranking:** Use a weighted combination of resonance and centrality to rank and select top concepts.

### 4. Pipeline Integration

- **Concept Tuple Enrichment:** Each concept tuple will include:
  - Best candidate title (from advanced extraction)
  - Resonance score
  - Narrative centrality score
  - Cluster membership (list of block indices)
- **Persistence:** Output both `.npz` (for ALAN core) and `.json` (for UI/visualization), including all new fields.

### 5. CLI and Testing

- **Command-Line Entrypoint:** Provide a CLI for batch PDF ingestion and parameter tuning.
- **Unit Tests:** Scaffold and implement pytest-based tests for each module (block extraction, features, clustering, scoring, keywords, persistence).

---

## Summary

This pipeline ensures every concept is:

- Human-interpretable and document-derived (no placeholders)
- Spectrally and phase-grounded (Koopman and oscillator logic)
- Cohesive, central, and persistent (not trivial or fragmented)
- Ready for ALAN's cognitive reasoning and interactive visualization

All logic is modular, unit-testable, and LLM-free.
