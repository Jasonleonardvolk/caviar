#!/bin/bash

# TORI Authentication & Upload Test Script
# Usage: ./test_upload.sh [pdf_file_path]

echo "🚀 TORI Authentication & Upload Test"
echo "====================================="

# Configuration
TORI_HOST="localhost:8443"
PDF_FILE="${1:-2407.15527v2.pdf}"

# Check if PDF file exists
if [[ ! -f "$PDF_FILE" ]]; then
    echo "❌ Error: PDF file '$PDF_FILE' not found"
    echo "Usage: $0 [pdf_file_path]"
    exit 1
fi

echo "📄 PDF File: $PDF_FILE"
echo "🌐 TORI Host: $TORI_HOST"

# Step 1: Login and get token
echo ""
echo "🔐 Step 1: Authenticating with TORI..."
LOGIN_RESPONSE=$(curl -s -X POST "http://$TORI_HOST/api/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username":"operator","password":"operator123"}')

if [[ $? -ne 0 ]]; then
    echo "❌ Error: Failed to connect to TORI server"
    echo "Make sure TORI is running on http://$TORI_HOST"
    exit 1
fi

# Extract token
TOKEN=$(echo "$LOGIN_RESPONSE" | grep -o '"token":"[^"]*"' | cut -d'"' -f4)

if [[ -z "$TOKEN" ]]; then
    echo "❌ Error: Failed to get authentication token"
    echo "Response: $LOGIN_RESPONSE"
    exit 1
fi

echo "✅ Authentication successful!"
echo "🎫 Token: ${TOKEN:0:20}..."

# Step 2: Upload PDF
echo ""
echo "📤 Step 2: Uploading PDF..."
UPLOAD_RESPONSE=$(curl -s -X POST "http://$TORI_HOST/api/upload" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@$PDF_FILE;type=application/pdf")

if [[ $? -ne 0 ]]; then
    echo "❌ Error: Upload failed"
    exit 1
fi

echo "📋 Upload Response:"
echo "$UPLOAD_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$UPLOAD_RESPONSE"

# Check if upload was successful
if echo "$UPLOAD_RESPONSE" | grep -q '"success":true\|"message":"PDF uploaded successfully"'; then
    echo ""
    echo "✅ Upload successful!"
    
    # Extract file path for concept extraction
    FILE_PATH=$(echo "$UPLOAD_RESPONSE" | grep -o '"file_path":"[^"]*"' | cut -d'"' -f4)
    
    if [[ -n "$FILE_PATH" ]]; then
        echo "📂 File Path: $FILE_PATH"
        
        # Step 3: Extract concepts
        echo ""
        echo "🧬 Step 3: Extracting concepts..."
        EXTRACT_RESPONSE=$(curl -s -X POST "http://$TORI_HOST/api/extract" \
          -H "Content-Type: application/json" \
          -H "Authorization: Bearer $TOKEN" \
          -d "{\"file_path\": \"$FILE_PATH\"}")
        
        echo "📊 Extraction Response:"
        echo "$EXTRACT_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$EXTRACT_RESPONSE"
        
        if echo "$EXTRACT_RESPONSE" | grep -q '"success":true'; then
            echo ""
            echo "🎆 Complete Success! PDF uploaded and concepts extracted!"
        else
            echo ""
            echo "⚠️ Upload successful, but concept extraction may have failed"
        fi
    fi
else
    echo ""
    echo "❌ Upload failed. Check the response above for details."
    exit 1
fi

echo ""
echo "🎯 Test completed!"
