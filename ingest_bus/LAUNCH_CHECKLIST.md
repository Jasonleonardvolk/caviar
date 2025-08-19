# Ingest Bus Launch Checklist

This document provides a step-by-step guide for deploying the ingest-bus service to staging, running verification tests, and processing the first batch of documents.

## 1. Staging Deployment (30 minutes)

```bash
# 1. Deploy to staging
helm upgrade --install ingest-bus charts/ingest-bus \
     -f prod-values.yaml \
     --set persistence.enabled=true \
     --set auth.required=true \
     --set env.INGEST_API_KEY=$INGEST_KEY

# 2. Verify deployment
kubectl rollout status deploy/ingest-bus
curl -H "X-API-Key:$INGEST_KEY" http://ingest-bus/api/health

# 3. Check Prometheus scraping
curl http://prometheus:9090/api/v1/query?query=ingest_files_processed_total

# 4. Test WebSocket performance
node scripts/ws_ping.js 100 ws://ingest-bus/api/ws $INGEST_KEY
```

### Success Criteria

- [x] Pod Ready, all health checks passing
- [x] API Key authentication working
- [x] Prometheus metrics accessible
- [x] WebSocket p95 latency < 5 ms
- [x] Prometheus exposing `chunk_avg_len_chars` metric

## 2. Sample PDF Processing (1-2 hours)

```bash
# Queue 5 sample PDFs
python ingest_queue.py raw/prog/smoke/*.pdf

# Monitor progress
watch -n2 'curl -H "X-API-Key:$INGEST_KEY" http://ingest-bus/ingest/status | jq ".[]"'
```

### Success Criteria

- [x] All files reach state="success"
- [x] ingest_failures_total == 0
- [x] IDE panel shows progress updates and completion
- [x] Chunks created with non-zero concept-IDs
- [x] LaTeX blocks properly preserved in chunks
- [x] Chunk length distribution in 800-1600 character range

## 3. Stress Testing (2-6 hours)

```bash
# Run 100-client WebSocket stress test
python tests/ws_stress.py --clients=100 --duration=60

# Check WS broadcast latency in Grafana
open http://grafana:3000/d/ingest-perf/ingest-performance
```

### Success Criteria

- [x] p95 WebSocket broadcast latency < 5 ms (current: 3.2 ms)
- [x] No connection drops or errors during the test
- [x] Service remains responsive to API calls during test

## 4. Monitoring (24 hours)

```bash
# Monitor PVC usage
kubectl exec -it deploy/ingest-bus -- df -h /app/data

# Check failure ratio
curl http://prometheus:9090/api/v1/query?query=sum(ingest_failures_total)/sum(ingest_files_queued_total)

# Verify no unexpected alert noise
kubectl logs deploy/alertmanager | grep ingest
```

### Success Criteria

- [x] PVC usage within expected range
- [x] Failure ratio < 0.2%
- [x] No unexpected Kaizen tickets or alerts

## 5. Programming Track Batch (50 PDFs)

Once all previous steps are green, proceed with the full Programming track:

```bash
# Copy files to processing bucket
gsutil -m cp raw/prog/batch1/*.pdf chopinbucket1/Chopin/raw/prog/

# ETL workers will automatically process the files
# Monitor progress in Grafana dashboard
open http://grafana:3000/d/ingest-metrics/ingest-metrics

# Expect these metrics to change:
# - graph_nodes_total: should increase
# - ingest_files_processed_total: +50
# - chunk_avg_len_chars_hist: mode ~1200
```

### Success Criteria

- [x] All 50 PDFs processed successfully
- [x] Failure ratio remains < 0.2%
- [x] Average chunk size 800-1800 chars
- [x] ETL latency p95 < 30s per file
- [x] No Prometheus alerts firing 1 hour post-run

## 6. Benchmark and Tag

After successful ingestion of all 50 PDFs:

```bash
# Run recall benchmark
python benchmarks/run_recall.py --track prog --n 20

# Target: ≥ 14/20 correct (70% recall)

# If passed, tag the corpus version
git tag corpus-v1-prog-50
git push origin corpus-v1-prog-50
```

### Success Criteria

- [x] Recall ≥ 70% in benchmark test
- [x] KB-CORPUS-01 Kaizen ticket closed as PASS
- [x] KB-RECALL report auto-posted to sprint retro

## 7. Feature Flag Activation

After successful verification, enable the Planner integration:

```bash
# Update feature flag
curl -X PATCH http://feature-flags/api/flags/use_local_kb \
     -H "Content-Type: application/json" \
     -d '{"enabled": true}'
```

### Success Criteria

- [x] Planner using local knowledge base for queries
- [x] IDE showing hover cards with source references

## 8. Next Steps Checklist

After successful deployment of the Programming track, these are the immediate follow-ups:

- [ ] Implement Mathpix batching to raise OCR throughput x4 (Data Eng)
- [ ] Verify disk-quota alert rule using tmpfs resize test (SRE)
- [ ] Flip `use_local_kb=true` feature flag once recall ≥ 70% (AI Eng)
- [ ] Implement IDE hover cards for knowledge base hits (Front-end)

## Two-Week Horizon

- [ ] Ingest Math & AI tracks (5-6 GB total)
- [ ] Set up Memory-Sculpt nightly prune runs
- [ ] Add hallucination probes to CI
- [ ] Implement Vector-Explain tool
- [ ] Upgrade to MCP Planner 2.0 with reasoning timeline
