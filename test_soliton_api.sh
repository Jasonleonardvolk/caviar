#!/bin/bash
# Quick test script for the Soliton API endpoints

echo "Testing Soliton API Endpoints..."
echo "================================"

# Base URL
BASE_URL="http://localhost:8000/api/soliton"

# 1. Initialize
echo -e "\n1. Testing /initialize endpoint..."
curl -X POST "$BASE_URL/initialize" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test_user_123", "lattice_reset": false}'

# 2. Store a memory with embedding
echo -e "\n\n2. Testing /store endpoint..."
curl -X POST "$BASE_URL/store" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user_123",
    "concept_id": "test_concept_001",
    "content": {"text": "This is a test memory", "type": "test"},
    "activation_strength": 0.8,
    "embedding": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
  }'

# 3. Query memories
echo -e "\n\n3. Testing /query endpoint..."
curl -X POST "$BASE_URL/query" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user_123",
    "query_embedding": [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85],
    "k": 5
  }'

# 4. Get stats
echo -e "\n\n4. Testing /stats endpoint..."
curl -X GET "$BASE_URL/stats/test_user_123"

# 5. Check health
echo -e "\n\n5. Testing /health endpoint..."
curl -X GET "$BASE_URL/health"

echo -e "\n\nTest completed!"
