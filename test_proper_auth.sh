#!/bin/bash

# TORI Complete Authentication Test Script
# This script properly handles Bearer token authentication

echo "🔐 TORI Authentication & Upload Test"
echo "===================================="

# Configuration
TORI_HOST="localhost:8443"
PDF_FILE="${1:-2407.15527v2.pdf}"

# Check if we can reach TORI
echo "🌐 Testing connection to TORI..."
if ! curl -s "http://$TORI_HOST/health" > /dev/null; then
    echo "❌ Error: Cannot reach TORI server at http://$TORI_HOST"
    echo "   Make sure TORI is running with:"
    echo "   python phase3_complete_production_system.py --host 0.0.0.0 --port 8443"
    exit 1
fi
echo "✅ TORI server is responding"

# Step 1: Authenticate and get token
echo ""
echo "🔐 Step 1: Getting authentication token..."
echo "Command: curl -X POST \"http://$TORI_HOST/api/auth/login\" -H \"Content-Type: application/json\" -d '{\"username\":\"operator\",\"password\":\"operator123\"}'"

LOGIN_RESPONSE=$(curl -s -X POST "http://$TORI_HOST/api/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username":"operator","password":"operator123"}')

echo "📋 Login Response:"
echo "$LOGIN_RESPONSE"

# Extract token
TOKEN=$(echo "$LOGIN_RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('token', ''))
except:
    pass
")

if [[ -z "$TOKEN" ]]; then
    echo ""
    echo "❌ Failed to extract token from response"
    echo "Please check if TORI is running and credentials are correct"
    exit 1
fi

echo ""
echo "✅ Token extracted successfully: ${TOKEN:0:20}..."

# Check if PDF file exists
if [[ ! -f "$PDF_FILE" ]]; then
    echo ""
    echo "📄 PDF file '$PDF_FILE' not found - creating a test file..."
    echo "This is a test PDF content for TORI authentication testing" > "$PDF_FILE"
    echo "✅ Created test file: $PDF_FILE"
fi

# Step 2: Upload with proper Authorization header
echo ""
echo "📤 Step 2: Uploading PDF with Bearer token..."
echo "Command: curl -X POST \"http://$TORI_HOST/api/upload\" -H \"Authorization: Bearer \$TOKEN\" -F \"file=@$PDF_FILE\""

UPLOAD_RESPONSE=$(curl -s -X POST "http://$TORI_HOST/api/upload" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@$PDF_FILE;type=application/pdf")

echo ""
echo "📋 Upload Response:"
echo "$UPLOAD_RESPONSE"

# Check response
if echo "$UPLOAD_RESPONSE" | grep -q '"success":true\|"message":"PDF uploaded successfully"\|"status":"uploaded"'; then
    echo ""
    echo "🎆 SUCCESS! PDF uploaded successfully!"
    
    # If upload returned a file path, try extraction
    FILE_PATH=$(echo "$UPLOAD_RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('file_path', ''))
except:
    pass
")
    
    if [[ -n "$FILE_PATH" ]]; then
        echo ""
        echo "🧬 Step 3: Extracting concepts..."
        EXTRACT_RESPONSE=$(curl -s -X POST "http://$TORI_HOST/api/extract" \
          -H "Content-Type: application/json" \
          -H "Authorization: Bearer $TOKEN" \
          -d "{\"file_path\": \"$FILE_PATH\"}")
        
        echo "📊 Extraction Response:"
        echo "$EXTRACT_RESPONSE"
        
        if echo "$EXTRACT_RESPONSE" | grep -q '"success":true'; then
            echo ""
            echo "🎆 COMPLETE SUCCESS! PDF uploaded and concepts extracted!"
        fi
    fi
    
elif echo "$UPLOAD_RESPONSE" | grep -q "403\|Forbidden\|Not authenticated"; then
    echo ""
    echo "❌ 403 Authentication Error - Token may be invalid"
    echo "   Check that the login was successful and token was extracted correctly"
    
elif echo "$UPLOAD_RESPONSE" | grep -q "401\|Unauthorized"; then
    echo ""
    echo "❌ 401 Authorization Error - Check credentials or token expiration"
    
else
    echo ""
    echo "⚠️ Unexpected response - check the output above"
fi

echo ""
echo "🎯 Authentication test completed!"
echo ""
echo "💡 Manual commands for testing:"
echo "   1. Login: curl -X POST \"http://$TORI_HOST/api/auth/login\" -H \"Content-Type: application/json\" -d '{\"username\":\"operator\",\"password\":\"operator123\"}'"
echo "   2. Upload: curl -X POST \"http://$TORI_HOST/api/upload\" -H \"Authorization: Bearer \$TOKEN\" -F \"file=@$PDF_FILE\""
echo ""
echo "📋 Available user roles:"
echo "   - observer  / observer123  (read-only)"
echo "   - operator  / operator123  (can upload)"
echo "   - approver  / approver123  (can approve)"
echo "   - admin     / admin123     (full access)"
