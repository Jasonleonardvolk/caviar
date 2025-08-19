#!/bin/bash
# build_capsule.sh - Build TORI deployment capsules with verification
# chmod +x build_capsule.sh

set -euo pipefail

# Configuration
BUILD_DIR="${BUILD_DIR:-/tmp/tori_build_$$}"
OUTPUT_DIR="${OUTPUT_DIR:-./artifacts}"
SIGNING_KEY="${SIGNING_KEY:-/etc/tori/keys/minisign.key}"
PUBLIC_KEY="${PUBLIC_KEY:-/etc/tori/keys/minisign.pub}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

usage() {
    cat << EOF
Usage: $0 [OPTIONS] SERVICE_NAME VERSION SOURCE_DIR

Build a TORI capsule for deployment.

Arguments:
  SERVICE_NAME    Name of the service (e.g., tori-ingest)
  VERSION         Version string (e.g., 1.4.3)
  SOURCE_DIR      Directory containing service files

Options:
  -o, --output DIR      Output directory (default: ./artifacts)
  -k, --key FILE        Minisign private key (default: /etc/tori/keys/minisign.key)
  -p, --public FILE     Minisign public key (default: /etc/tori/keys/minisign.pub)
  -m, --manifest FILE   Custom manifest file (default: SOURCE_DIR/capsule.yml)
  -s, --skip-sign       Skip signature generation
  -v, --verbose         Verbose output
  -h, --help           Show this help

Example:
  $0 tori-ingest 1.4.3 ./services/ingest

EOF
}

# Parse arguments
SKIP_SIGN=false
VERBOSE=false
MANIFEST_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -k|--key)
            SIGNING_KEY="$2"
            shift 2
            ;;
        -p|--public)
            PUBLIC_KEY="$2"
            shift 2
            ;;
        -m|--manifest)
            MANIFEST_FILE="$2"
            shift 2
            ;;
        -s|--skip-sign)
            SKIP_SIGN=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
        *)
            break
            ;;
    esac
done

if [ $# -ne 3 ]; then
    log_error "Missing required arguments"
    usage
    exit 1
fi

SERVICE_NAME="$1"
VERSION="$2"
SOURCE_DIR="$3"

# Validate inputs
if [ ! -d "$SOURCE_DIR" ]; then
    log_error "Source directory does not exist: $SOURCE_DIR"
    exit 1
fi

# Create directories
mkdir -p "$BUILD_DIR" "$OUTPUT_DIR"

log_info "Building capsule for $SERVICE_NAME v$VERSION"

# Copy files to build directory
log_info "Copying files to build directory..."
cp -r "$SOURCE_DIR"/* "$BUILD_DIR/"

# Generate or copy manifest
if [ -n "$MANIFEST_FILE" ] && [ -f "$MANIFEST_FILE" ]; then
    cp "$MANIFEST_FILE" "$BUILD_DIR/capsule.yml"
elif [ ! -f "$BUILD_DIR/capsule.yml" ]; then
    log_info "Generating capsule manifest..."
    cat > "$BUILD_DIR/capsule.yml" << EOF
name: $SERVICE_NAME
version: $VERSION
entrypoint: bin/start.sh
dependencies:
  python: "3.10"
build_info:
  timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)
  builder: $(whoami)@$(hostname)
  git_sha: $(git rev-parse HEAD 2>/dev/null || echo "unknown")
services:
  - name: $SERVICE_NAME
    slice: tori-server.slice
    resource_limits:
      cpu_quota: 200
      memory_max: 4G
EOF
fi

# Add public key to manifest if signing
if [ "$SKIP_SIGN" = false ] && [ -f "$PUBLIC_KEY" ]; then
    PUBLIC_KEY_CONTENT=$(cat "$PUBLIC_KEY" | base64 -w 0)
    # Use a temp file to avoid issues with yq
    cp "$BUILD_DIR/capsule.yml" "$BUILD_DIR/capsule.yml.tmp"
    if command -v yq >/dev/null 2>&1; then
        yq eval ".public_key = \"$PUBLIC_KEY_CONTENT\"" "$BUILD_DIR/capsule.yml.tmp" > "$BUILD_DIR/capsule.yml"
        yq eval '.signature = "capsule.sig"' -i "$BUILD_DIR/capsule.yml"
    else
        # Fallback: append to YAML
        echo "public_key: $PUBLIC_KEY_CONTENT" >> "$BUILD_DIR/capsule.yml"
        echo 'signature: capsule.sig' >> "$BUILD_DIR/capsule.yml"
    fi
    rm -f "$BUILD_DIR/capsule.yml.tmp"
fi

# Create file list
log_info "Generating file list..."
cd "$BUILD_DIR"
find . -type f -print0 | sort -z | xargs -0 sha256sum > files.sha256

# Create tarball
TARBALL_NAME="${SERVICE_NAME}-${VERSION}.tar.gz"
TARBALL_PATH="$OUTPUT_DIR/$TARBALL_NAME"
log_info "Creating tarball: $TARBALL_NAME"
tar czf "$TARBALL_PATH" .

# Calculate capsule hash
CAPSULE_SHA=$(sha256sum "$TARBALL_PATH" | cut -d' ' -f1)
CAPSULE_SIZE=$(stat -c%s "$TARBALL_PATH" 2>/dev/null || stat -f%z "$TARBALL_PATH" 2>/dev/null || echo "0")

log_info "Capsule SHA256: $CAPSULE_SHA"
log_info "Capsule size: $CAPSULE_SIZE bytes"

# Sign capsule if requested
SIGNATURE_FILE=""
if [ "$SKIP_SIGN" = false ]; then
    if command -v minisign >/dev/null 2>&1 && [ -f "$SIGNING_KEY" ]; then
        log_info "Signing capsule with minisign..."
        SIGNATURE_FILE="$OUTPUT_DIR/${SERVICE_NAME}-${VERSION}.tar.gz.sig"
        minisign -Sm "$TARBALL_PATH" -s "$SIGNING_KEY" -x "$SIGNATURE_FILE"
        log_info "Signature created: $SIGNATURE_FILE"
    else
        log_warn "Minisign not available or key not found, skipping signature"
    fi
fi

# Generate results.txt
RESULTS_FILE="$OUTPUT_DIR/results.txt"
log_info "Generating results.txt..."
cat > "$RESULTS_FILE" << EOF
TORI Capsule Build Results
==========================
Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)
Service: $SERVICE_NAME
Version: $VERSION

Capsule Information:
-------------------
File: $TARBALL_NAME
SHA256: $CAPSULE_SHA
Size: $CAPSULE_SIZE bytes

Build Information:
-----------------
Builder: $(whoami)@$(hostname)
Build Directory: $BUILD_DIR
Source Directory: $SOURCE_DIR
Git SHA: $(cd "$SOURCE_DIR" && git rev-parse HEAD 2>/dev/null || echo "unknown")
Git Branch: $(cd "$SOURCE_DIR" && git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")

EOF

if [ -n "$SIGNATURE_FILE" ] && [ -f "$SIGNATURE_FILE" ]; then
    echo "Signature Information:" >> "$RESULTS_FILE"
    echo "---------------------" >> "$RESULTS_FILE"
    echo "Signature File: $(basename "$SIGNATURE_FILE")" >> "$RESULTS_FILE"
    echo "Signature SHA256: $(sha256sum "$SIGNATURE_FILE" | cut -d' ' -f1)" >> "$RESULTS_FILE"
    echo "Public Key: $PUBLIC_KEY" >> "$RESULTS_FILE"
    echo "" >> "$RESULTS_FILE"
fi

echo "File List:" >> "$RESULTS_FILE"
echo "----------" >> "$RESULTS_FILE"
cd "$BUILD_DIR"
find . -type f -printf '%P\n' | sort >> "$RESULTS_FILE"

# Also create a JSON version for CI/CD
JSON_FILE="$OUTPUT_DIR/build_metadata.json"
cat > "$JSON_FILE" << EOF
{
  "service": "$SERVICE_NAME",
  "version": "$VERSION",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "capsule": {
    "file": "$TARBALL_NAME",
    "sha256": "$CAPSULE_SHA",
    "size": $CAPSULE_SIZE
  },
  "signature": {
    "file": "$(basename "$SIGNATURE_FILE" 2>/dev/null || echo "")",
    "public_key": "$PUBLIC_KEY"
  },
  "build": {
    "builder": "$(whoami)@$(hostname)",
    "git_sha": "$(cd "$SOURCE_DIR" && git rev-parse HEAD 2>/dev/null || echo "unknown")",
    "git_branch": "$(cd "$SOURCE_DIR" && git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")"
  }
}
EOF

# Cleanup
rm -rf "$BUILD_DIR"

log_info "Build complete!"
log_info "Capsule: $TARBALL_PATH"
log_info "Results: $RESULTS_FILE"
log_info "Metadata: $JSON_FILE"

if [ "$VERBOSE" = true ]; then
    echo ""
    cat "$RESULTS_FILE"
fi
