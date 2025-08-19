#!/bin/bash

# Complete working example with proper authentication
echo "🚀 TORI Complete Authentication Example"
echo "======================================="

# Step 1: Login and extract token
echo "1️⃣ Getting authentication token..."
RESPONSE=$(curl -s -X POST "http://localhost:8443/api/auth/login" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{"username":"operator","password":"operator123"}')

echo "Login Response:"
echo "$RESPONSE"

# Extract token (this works on most systems)
TOKEN=$(echo "$RESPONSE" | grep -o '"token":"[^"]*"' | cut -d'"' -f4)

if [[ -z "$TOKEN" ]]; then
    echo "❌ Failed to extract token"
    echo "Please manually copy the token from the response above"
    exit 1
fi

echo ""
echo "✅ Token extracted: ${TOKEN:0:20}..."
echo ""

# Step 2: Upload with proper Authorization header
echo "2️⃣ Uploading PDF with Authorization header..."
echo "Command being executed:"
echo "curl -X POST \"http://localhost:8443/api/upload\" \\"
echo "  -H \"accept: application/json\" \\"
echo "  -H \"Authorization: Bearer $TOKEN\" \\"
echo "  -F \"file=@2407.15527v2.pdf;type=application/pdf\""
echo ""

curl -X POST "http://localhost:8443/api/upload" \
  -H "accept: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@2407.15527v2.pdf;type=application/pdf"

echo ""
echo "🎯 Upload attempt completed!"
