#!/bin/bash
# deploy_tori_production_fixed.sh - Production deployment with all fixes

set -e

echo "🚀 Deploying TORI Production System with Qwen3-Embedding-8B (Fixed)"

# Check GPU availability
if ! nvidia-smi > /dev/null 2>&1; then
    echo "❌ CUDA/GPU not available. Qwen3-8B requires GPU."
    exit 1
fi

# Set environment variables
export TORI_EMBED_MODE=local
export TORI_EMBED_URL=http://localhost:8080
export TORI_COSINE_THRESHOLD=0.65
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/var/tori/huggingface_cache  # Fixed: Pin model cache
export DISABLE_AUTH=true  # For development, remove in production

# Create directories with proper permissions
sudo mkdir -p /var/tori/{emb_cache,logs,psi_archive,huggingface_cache}
sudo chown -R $USER:$USER /var/tori
chmod 755 /var/tori

# Setup Python environment in dickbox
cd "C:\Users\jason\Desktop\tori\kha\mcp_metacognitive\agents\dickbox"

# Install production dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install sentence-transformers[all] fastapi uvicorn[standard] httpx diskcache
pip install spacy scispacy numpy scipy scikit-learn
pip install prometheus-fastapi-instrumentator psutil GPUtil PyJWT PyMuPDF

# Download models with proper caching
echo "📦 Downloading Qwen3-Embedding-8B..."
python -c "
import os
os.environ['HF_HOME'] = '/var/tori/huggingface_cache'
from sentence_transformers import SentenceTransformer
print('Downloading Qwen3-Embedding-8B...')
model = SentenceTransformer('Qwen/Qwen3-Embedding-8B')
print('Model downloaded successfully')
"

# Create systemd service for production deployment
sudo tee /etc/systemd/system/tori-embedding.service > /dev/null <<EOF
[Unit]
Description=TORI Embedding Service (Qwen3-8B)
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment="CUDA_VISIBLE_DEVICES=0"
Environment="HF_HOME=/var/tori/huggingface_cache"
Environment="TORI_EMBED_CACHE=/var/tori/emb_cache"
ExecStart=$(which python) serve_embeddings_production.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable tori-embedding.service
sudo systemctl start tori-embedding.service

# Wait for service to be ready with proper health check polling
echo "⏳ Waiting for embedding service to start..."
timeout=120
elapsed=0
while [ $elapsed -lt $timeout ]; do
    if curl -f http://localhost:8080/health > /dev/null 2>&1; then
        echo "✅ Embedding service is healthy"
        break
    fi
    echo "Waiting... ($elapsed/$timeout seconds)"
    sleep 5
    elapsed=$((elapsed + 5))
done

if [ $elapsed -ge $timeout ]; then
    echo "❌ Embedding service failed to start within $timeout seconds"
    sudo systemctl status tori-embedding.service
    exit 1
fi

# Run production tests
echo "🧪 Running production quality tests..."
python -m pytest tests/test_production_ready.py -v

# Test sample ingestion with better error handling
echo "📄 Testing sample document ingestion..."
python -c "
import asyncio
import sys
sys.path.append('.')
from core.canonical_ingestion_production_fixed import ingest_file_production

async def test_ingestion():
    try:
        result = await ingest_file_production('tests/fixtures/sample.pdf')
        print(f'Ingestion result: {result.success}')
        print(f'Quality score: {result.quality_metrics.get(\"overall_quality\", 0):.3f}')
        print(f'Penrose status: {result.penrose_verification.status if result.penrose_verification else \"None\"}')
        print(f'SLO compliance: {result.quality_metrics.get(\"slo_compliance\", 0.0):.1f}')
    except Exception as e:
        print(f'Test ingestion failed: {e}')
        exit(1)

asyncio.run(test_ingestion())
"

echo "🎯 TORI Production System deployed successfully!"
echo "📊 Embedding service running as systemd service: tori-embedding.service"
echo "🔍 Monitor with: sudo systemctl status tori-embedding.service"
echo "📋 View logs with: sudo journalctl -u tori-embedding.service -f"
echo "💾 Cache directory: /var/tori/emb_cache"
echo "🔧 GPU metrics available at: http://localhost:8080/stats"

echo "✨ Ready for theory-grade concept balloons! 🎈💥"
