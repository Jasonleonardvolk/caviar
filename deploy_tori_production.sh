#!/bin/bash
# deploy_tori_production.sh - Complete production deployment

set -e

echo "🚀 Deploying TORI Production System with Qwen3-Embedding-8B"

# Check GPU availability
if ! nvidia-smi > /dev/null 2>&1; then
    echo "❌ CUDA/GPU not available. Qwen3-8B requires GPU."
    exit 1
fi

# Create virtual environment
python3.11 -m venv venv_tori_prod
source venv_tori_prod/bin/activate

# Install production dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install sentence-transformers[all] fastapi uvicorn[standard] httpx diskcache
pip install spacy scispacy numpy scipy scikit-learn

# Download models
python -c "
from sentence_transformers import SentenceTransformer
print('Downloading Qwen3-Embedding-8B...')
model = SentenceTransformer('Qwen/Qwen3-Embedding-8B')
print('Model downloaded successfully')
"

# Setup directories
mkdir -p /var/tori/emb_cache
mkdir -p /var/tori/logs
mkdir -p /var/tori/psi_archive

# Set environment variables
export TORI_EMBED_MODE=local
export TORI_EMBED_URL=http://localhost:8080
export TORI_COSINE_THRESHOLD=0.65
export CUDA_VISIBLE_DEVICES=0

# Start embedding service
echo "🔥 Starting Qwen3 embedding service..."
CUDA_VISIBLE_DEVICES=0 python serve_embeddings.py > /var/tori/logs/embedding_service.log 2>&1 &
EMBED_PID=$!

# Wait for service to start
sleep 30

# Test service health
if curl -f http://localhost:8080/health > /dev/null 2>&1; then
    echo "✅ Embedding service is healthy"
else
    echo "❌ Embedding service failed to start"
    kill $EMBED_PID 2>/dev/null || true
    exit 1
fi

# Run production tests
echo "🧪 Running production quality tests..."
python -m pytest tests/test_production_ready.py -v

# Test sample ingestion
echo "📄 Testing sample document ingestion..."
python -c "
import asyncio
from core.canonical_ingestion_production import ingest_file_production

async def test_ingestion():
    result = await ingest_file_production('tests/fixtures/sample.pdf')
    print(f'Ingestion result: {result.success}')
    print(f'Quality score: {result.quality_metrics.get(\"overall_quality\", 0):.3f}')
    print(f'Penrose status: {result.penrose_verification.status}')

asyncio.run(test_ingestion())
"

echo "🎯 TORI Production System deployed successfully!"
echo "📊 Embedding service running on port 8080"
echo "🔍 Monitor logs at /var/tori/logs/"
echo "💾 Cache directory: /var/tori/emb_cache"

# Save PID for cleanup
echo $EMBED_PID > /var/tori/embedding_service.pid

echo "✨ Ready for theory-grade concept balloons! 🎈💥"
