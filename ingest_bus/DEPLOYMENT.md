# Ingest Bus Deployment Guide

This guide provides instructions for deploying and integrating the ingest-bus service in production environments.

## Helm Deployment

Deploy or upgrade the service using Helm:

```bash
helm upgrade --install ingest-bus ./ingest-bus/helm \
  --namespace data-processing \
  --set image.tag=latest \
  --set service.type=ClusterIP \
  --set persistence.enabled=true \
  --set persistence.size=10Gi \
  --set metrics.enabled=true
```

## Prometheus Configuration

Add this scrape configuration to your Prometheus configuration:

```yaml
scrape_configs:
  - job_name: 'ingest-bus'
    metrics_path: '/metrics'
    scrape_interval: 15s
    static_configs:
      - targets: ['ingest-bus:8081']
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        action: keep
        regex: ingest-bus
```

## Persistent Storage

For production, we recommend using:

1. **Persistent Volume Claim (PVC)** - Default and recommended approach
   - Configure in Helm values: `persistence.enabled=true`
   - Create ticket OPS-STORAGE-001 to implement PVC

2. **SQLite Option** (for smaller deployments)
   - Can be enabled as an alternative to file-based storage
   - Configure in Helm values: `storage.type=sqlite`

## API Authentication

Add API key authentication by enabling the flag:

```bash
# In deployment config
--set security.requireApiKey=true
--set security.apiKey=<your-secret-key>
```

Clients must then include the header `X-API-Key: <your-secret-key>` with requests.

## Metrics Enhancements

Additional chunk size metrics have been implemented:

```
# Chunk size histogram with improved buckets
chunk_size_chars_bucket{le="800"} 123
chunk_size_chars_bucket{le="1200"} 456
chunk_size_chars_bucket{le="1600"} 789
chunk_size_chars_bucket{le="2000"} 1234
chunk_size_chars_bucket{le="+Inf"} 1500
```

Set up alert rule to watch 90-percentile > 2000 chars:

```yaml
- alert: ChunkSizeTooLarge
  expr: histogram_quantile(0.9, sum(rate(chunk_size_chars_bucket[5m])) by (le)) > 2000
  for: 15m
  labels:
    severity: warning
  annotations:
    description: "90th percentile chunk size exceeds 2000 characters"
    summary: "Chunks may be too large for optimal embedding"
```

## Worker Implementation

The document worker should be implemented according to this blueprint:

1. Poll ingest-bus for queued jobs (`GET /api/jobs?status=queued`)
2. Claim job to process (`PATCH /api/jobs/{job_id}` with `status=processing`)
3. Download document from `file_url`
4. Check MIME type and validate document
5. Extract text and images using appropriate library (e.g., PyPDF, Tika)
6. Split content into chunks based on semantic boundaries
7. Create embeddings for chunks
8. Insert into knowledge graph and link concepts
9. Update job with chunk information and progress
10. Set job to completed when done

## Kaizen Integration

Set `auto_ticket=true` for ingest alerts to automatically create Kaizen tickets:

```yaml
# In Kaizen configuration
alerts:
  ingest:
    auto_ticket: true
    assignee: "ingest-team"
    priority: "high"
```

Create Config PR to implement this change.

## Planner Integration

Quick win implementation for the Planner kb.search stub:

```python
@app.get("/api/kb/search")
async def search_kb(query: str):
    # Return empty list + schema_version until implementation is complete
    return {
        "results": [],
        "schema_version": "1.0"
    }
```

## Performance Testing

The WebSocket broadcast system should be tested with 100 concurrent subscribers.
Target p95 broadcast time should be < 5 ms on LAN.

Run the stress test:

```bash
python -m ingest-bus.tests.stress_test --subscribers=100 --duration=60
```

## 48-Hour Sprint Plan

1. Helm deploy to staging & enable Prometheus scraping
2. Create and implement disk-persistence ticket: create PVC claim & patch deployment
3. Add optional X-API-Key middleware with default pass-through behavior
4. Implement chunk_size_chars histogram + associated alert rule
5. Develop worker_extract.py v0 to process sample PDFs; verify delta WebSocket to IDE
6. Update Kaizen ingest alerts configuration with auto_ticket=true
7. Implement Planner stub kb.search that returns an empty list with schema_version
8. Run 100-subscriber WebSocket stress test; verify p95 broadcast < 5 ms

Once all items are green, proceed with uploading the Programming track (50 PDFs)
and open KB-CORPUS-01 Kaizen ticket for recall benchmarking.
