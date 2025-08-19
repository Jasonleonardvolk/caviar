<script lang="ts">
  let pdfFile: File | null = null;
  let loading = false;
  let error = "";
  let concepts: any[] = [];
  let resultMeta: any = null;
  let advancedAnalytics: any = null;
  let summary: any = null;

  async function handleFileChange(e: Event) {
    const files = (e.target as HTMLInputElement).files;
    pdfFile = files && files.length > 0 ? files[0] : null;
    concepts = [];
    error = "";
    resultMeta = null;
    advancedAnalytics = null;
    summary = null;
  }

  async function handleUpload() {
    if (!pdfFile) {
      error = "Please select a PDF file!";
      return;
    }
    loading = true;
    error = "";
    concepts = [];
    resultMeta = null;
    advancedAnalytics = null;
    summary = null;
    
    try {
      const form = new FormData();
      form.append("file", pdfFile);
      const res = await fetch("/api/upload", {
        method: "POST",
        body: form
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      if (data.status !== "success") throw new Error(data.result || "Extraction failed.");
      
      concepts = data.result.concepts || [];
      resultMeta = data.result;
      advancedAnalytics = data.advanced_analytics || {};
      summary = data.summary || {};
      
    } catch (e) {
      error = e.message || "Upload failed!";
    } finally {
      loading = false;
    }
  }

  function reset() {
    pdfFile = null;
    concepts = [];
    resultMeta = null;
    advancedAnalytics = null;
    summary = null;
    error = "";
  }

  function formatPercentage(value: any): string {
    if (typeof value === 'string' && value.includes('%')) return value;
    if (typeof value === 'number') return `${(value * 100).toFixed(1)}%`;
    return value?.toString() || 'N/A';
  }
</script>

<style>
  main {
    font-family: system-ui, sans-serif;
    display: flex;
    flex-direction: column;
    align-items: center;
    min-height: 90vh;
    padding: 2rem;
    background: linear-gradient(120deg, #09090b 0%, #23272f 100%);
    color: #eee;
  }
  .upload-card {
    background: #171923;
    border-radius: 2rem;
    padding: 2.5rem 2rem;
    box-shadow: 0 2px 24px #0006;
    max-width: 480px;
    width: 100%;
    margin: 2rem 0;
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
  }
  .file-input {
    padding: 1rem;
    border-radius: 1rem;
    background: #222;
    color: #fff;
    border: 2px dashed #444;
    transition: border-color .2s;
    margin-bottom: 0.75rem;
    width: 100%;
  }
  .file-input:focus {
    border-color: #6ee7b7;
    outline: none;
  }
  button {
    background: linear-gradient(90deg, #1e293b, #6ee7b7);
    color: #111;
    padding: 1rem 2rem;
    border: none;
    border-radius: 1.2rem;
    font-weight: bold;
    font-size: 1.1rem;
    cursor: pointer;
    box-shadow: 0 2px 8px #0003;
    transition: background .15s;
  }
  button:disabled {
    background: #444;
    color: #999;
    cursor: not-allowed;
  }
  .error {
    color: #f87171;
    font-weight: bold;
    margin: 0.5rem 0 0.25rem 0;
  }
  .analytics-panel {
    margin-top: 2rem;
    background: #1a1d29;
    border-radius: 1.5rem;
    padding: 2rem;
    box-shadow: 0 2px 16px #0004;
    max-width: 800px;
    width: 100%;
  }
  .metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin: 1.5rem 0;
  }
  .metric-card {
    background: #23272f;
    border-radius: 1rem;
    padding: 1.5rem;
    text-align: center;
    border: 1px solid #333;
  }
  .metric-value {
    font-size: 2rem;
    font-weight: bold;
    color: #6ee7b7;
    margin-bottom: 0.5rem;
  }
  .metric-label {
    font-size: 0.85rem;
    color: #999;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .concept-list {
    margin-top: 2rem;
    background: #23272f;
    border-radius: 1.5rem;
    padding: 2rem 1.5rem;
    box-shadow: 0 2px 16px #0004;
    max-width: 600px;
    width: 100%;
  }
  .concept {
    padding: 0.7rem 0.2rem;
    border-bottom: 1px solid #3336;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 1.07rem;
    letter-spacing: 0.01em;
  }
  .concept:last-child {
    border-bottom: none;
  }
  .concept-name {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  .badge {
    border-radius: 1em;
    padding: 0.17em 0.8em;
    font-size: 0.8em;
    font-weight: bold;
  }
  .badge.consensus {
    background: #f59e0b;
    color: #1a1a1a;
  }
  .badge.high-confidence {
    background: #6ee7b7;
    color: #1a1a1a;
  }
  .badge.boosted {
    background: #8b5cf6;
    color: white;
  }
  .concept-score {
    font-weight: bold;
    color: #6ee7b7;
  }
  .section-header {
    color: #6ee7b7;
    font-weight: bold;
    margin: 2rem 0 1rem 0;
    font-size: 1.2rem;
    border-bottom: 2px solid #6ee7b7;
    padding-bottom: 0.5rem;
  }
  .advanced-badge {
    background: linear-gradient(45deg, #6ee7b7, #8b5cf6);
    color: #1a1a1a;
    font-weight: bold;
    padding: 0.5rem 1rem;
    border-radius: 1rem;
    font-size: 0.85rem;
    display: inline-block;
    margin-bottom: 1rem;
  }
</style>

<main>
  <h1 style="font-size:2.6rem;letter-spacing:-.05em;font-weight:900;margin-bottom:0.5rem;">
    🚀 Prajna Concept Extractor
  </h1>
  <div class="advanced-badge">
    🔬 Powered by 4000-Hour Advanced Pipeline
  </div>
  
  <div class="upload-card">
    <input class="file-input" type="file" accept="application/pdf" on:change={handleFileChange} />
    <button on:click={handleUpload} disabled={loading || !pdfFile}>
      {loading ? "Processing via Advanced Pipeline..." : "Upload & Extract"}
    </button>
    {#if error}
      <div class="error">{error}</div>
    {/if}
    {#if (concepts.length > 0)}
      <button style="background:#23272f;color:#eee;margin-top:.5em;" on:click={reset}>Reset</button>
    {/if}
  </div>

  {#if summary}
    <div class="analytics-panel">
      <div class="section-header">🏆 Advanced Pipeline Analytics</div>
      
      <div class="metrics-grid">
        <div class="metric-card">
          <div class="metric-value">{summary.total_concepts || 0}</div>
          <div class="metric-label">Total Concepts</div>
        </div>
        <div class="metric-card">
          <div class="metric-value">{summary.pure_concepts || 0}</div>
          <div class="metric-label">Pure Concepts</div>
        </div>
        <div class="metric-card">
          <div class="metric-value">{summary.consensus_concepts || 0}</div>
          <div class="metric-label">Consensus</div>
        </div>
        <div class="metric-card">
          <div class="metric-value">{summary.high_confidence || 0}</div>
          <div class="metric-label">High Confidence</div>
        </div>
        <div class="metric-card">
          <div class="metric-value">{summary.auto_prefilled || 0}</div>
          <div class="metric-label">Auto-Prefilled</div>
        </div>
        <div class="metric-card">
          <div class="metric-value">{formatPercentage(summary.purity_efficiency)}</div>
          <div class="metric-label">Purity Efficiency</div>
        </div>
      </div>

      {#if advancedAnalytics?.context_extraction}
        <div style="margin-top:1.5rem;">
          <strong>📍 Context Analysis:</strong>
          <ul style="margin:0.5rem 0;color:#ccc;">
            <li>Title Extracted: {advancedAnalytics.context_extraction.title_extracted ? '✅' : '❌'}</li>
            <li>Abstract Found: {advancedAnalytics.context_extraction.abstract_extracted ? '✅' : '❌'}</li>
            <li>Sections: {advancedAnalytics.context_extraction.sections_identified?.join(', ') || 'None'}</li>
            <li>Avg Frequency: {advancedAnalytics.context_extraction.avg_concept_frequency?.toFixed(1) || 'N/A'}</li>
          </ul>
        </div>
      {/if}

      {#if advancedAnalytics?.extraction_methods?.universal_methods}
        <div style="margin-top:1rem;">
          <strong>🌍 Extraction Methods:</strong>
          <span style="color:#6ee7b7;margin-left:0.5rem;">
            {advancedAnalytics.extraction_methods.universal_methods.join(' + ')}
          </span>
        </div>
      {/if}

      {#if summary.processing_time}
        <div style="margin-top:1rem;text-align:center;color:#999;">
          ⏱️ Processed in {summary.processing_time.toFixed(2)}s by advanced pipeline
        </div>
      {/if}
    </div>
  {/if}

  {#if concepts.length > 0}
    <div class="concept-list">
      <div class="section-header">🧠 Extracted Concepts</div>
      {#each concepts.slice(0, 20) as c (c.name)}
        <div class="concept">
          <div class="concept-name">
            <strong>{c.name}</strong>
            {#if c.purity_metrics?.decision === 'TRIPLE_CONSENSUS' || c.purity_metrics?.decision === 'DOUBLE_CONSENSUS'}
              <span class="badge consensus">Consensus</span>
            {:else if c.purity_metrics?.decision === 'HIGH_CONF'}
              <span class="badge high-confidence">High Conf</span>
            {:else if c.method?.includes('boost')}
              <span class="badge boosted">Boosted</span>
            {/if}
          </div>
          <div class="concept-score">
            {c.score?.toFixed(3) || c.frequency || '--'}
          </div>
        </div>
      {/each}
      {#if concepts.length > 20}
        <div style="text-align:center;margin-top:1rem;color:#999;">
          ... and {concepts.length - 20} more concepts
        </div>
      {/if}
    </div>
  {/if}
</main>