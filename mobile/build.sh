#!/bin/bash
# Build script for TORI Hologram Mobile App
# Handles both iOS and Android builds with proper code signing

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
BUILD_TYPE=${1:-"debug"}  # debug or release
PLATFORM=${2:-"all"}      # ios, android, or all

echo -e "${GREEN}Building TORI Hologram Mobile App${NC}"
echo "Build Type: $BUILD_TYPE"
echo "Platform: $PLATFORM"

# Check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}Checking prerequisites...${NC}"
    
    # Check Node.js
    if ! command -v node &> /dev/null; then
        echo -e "${RED}Node.js is not installed${NC}"
        exit 1
    fi
    
    # Check npm
    if ! command -v npm &> /dev/null; then
        echo -e "${RED}npm is not installed${NC}"
        exit 1
    fi
    
    # Check Capacitor CLI
    if ! npm list -g @capacitor/cli &> /dev/null; then
        echo -e "${YELLOW}Installing Capacitor CLI...${NC}"
        npm install -g @capacitor/cli
    fi
    
    # Platform-specific checks
    if [[ "$PLATFORM" == "ios" ]] || [[ "$PLATFORM" == "all" ]]; then
        if ! command -v xcodebuild &> /dev/null; then
            echo -e "${RED}Xcode is not installed${NC}"
            exit 1
        fi
    fi
    
    if [[ "$PLATFORM" == "android" ]] || [[ "$PLATFORM" == "all" ]]; then
        if [ -z "$ANDROID_HOME" ]; then
            echo -e "${RED}ANDROID_HOME is not set${NC}"
            exit 1
        fi
    fi
}

# Install dependencies
install_dependencies() {
    echo -e "${YELLOW}Installing dependencies...${NC}"
    cd mobile
    npm install
    cd ..
}

# Build web assets
build_web_assets() {
    echo -e "${YELLOW}Building web assets...${NC}"
    cd mobile
    
    if [ "$BUILD_TYPE" == "release" ]; then
        npm run build -- --mode production
    else
        npm run build -- --mode development
    fi
    
    # Optimize assets for mobile
    echo -e "${YELLOW}Optimizing assets...${NC}"
    
    # Compress shaders with Brotli
    find dist-mobile/shaders -name "*.wgsl" -exec brotli -9 {} \;
    
    # Generate shader manifest
    node scripts/generate-shader-manifest.js
    
    cd ..
}

# Sync with Capacitor
sync_capacitor() {
    echo -e "${YELLOW}Syncing with Capacitor...${NC}"
    cd mobile
    cap sync
    cd ..
}

# Build iOS
build_ios() {
    echo -e "${GREEN}Building iOS app...${NC}"
    cd mobile/ios/App
    
    if [ "$BUILD_TYPE" == "release" ]; then
        # Production build with code signing
        xcodebuild -workspace App.xcworkspace \
                   -scheme App \
                   -configuration Release \
                   -archivePath build/TORIHologram.xcarchive \
                   archive \
                   CODE_SIGN_IDENTITY="iPhone Distribution" \
                   PROVISIONING_PROFILE_SPECIFIER="TORI Hologram Distribution"
        
        # Export IPA
        xcodebuild -exportArchive \
                   -archivePath build/TORIHologram.xcarchive \
                   -exportPath build \
                   -exportOptionsPlist ExportOptions.plist
        
        echo -e "${GREEN}iOS IPA created at: mobile/ios/App/build/TORIHologram.ipa${NC}"
    else
        # Debug build
        xcodebuild -workspace App.xcworkspace \
                   -scheme App \
                   -configuration Debug \
                   -sdk iphonesimulator \
                   -derivedDataPath build
        
        echo -e "${GREEN}iOS debug build complete${NC}"
    fi
    
    cd ../../..
}

# Build Android
build_android() {
    echo -e "${GREEN}Building Android app...${NC}"
    cd mobile/android
    
    if [ "$BUILD_TYPE" == "release" ]; then
        # Production build with signing
        ./gradlew assembleRelease
        
        # Sign APK (requires keystore)
        if [ -f "$HOME/.android/tori-hologram.keystore" ]; then
            jarsigner -verbose \
                      -sigalg SHA256withRSA \
                      -digestalg SHA-256 \
                      -keystore "$HOME/.android/tori-hologram.keystore" \
                      -storepass "$ANDROID_KEYSTORE_PASSWORD" \
                      app/build/outputs/apk/release/app-release-unsigned.apk \
                      tori-hologram
            
            # Align APK
            zipalign -v 4 \
                     app/build/outputs/apk/release/app-release-unsigned.apk \
                     app/build/outputs/apk/release/TORIHologram.apk
            
            echo -e "${GREEN}Android APK created at: mobile/android/app/build/outputs/apk/release/TORIHologram.apk${NC}"
        else
            echo -e "${YELLOW}Warning: Keystore not found, APK is unsigned${NC}"
        fi
    else
        # Debug build
        ./gradlew assembleDebug
        echo -e "${GREEN}Android debug APK created at: mobile/android/app/build/outputs/apk/debug/app-debug.apk${NC}"
    fi
    
    cd ../..
}

# Generate build info
generate_build_info() {
    echo -e "${YELLOW}Generating build info...${NC}"
    
    BUILD_NUMBER=$(date +%Y%m%d%H%M%S)
    GIT_COMMIT=$(git rev-parse --short HEAD)
    
    cat > mobile/build-info.json << EOF
{
  "buildNumber": "$BUILD_NUMBER",
  "buildType": "$BUILD_TYPE",
  "gitCommit": "$GIT_COMMIT",
  "buildDate": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "platform": "$PLATFORM"
}
EOF
}

# Calculate app size
check_app_size() {
    echo -e "${YELLOW}Checking app size...${NC}"
    
    if [[ "$PLATFORM" == "ios" ]] || [[ "$PLATFORM" == "all" ]]; then
        if [ -f "mobile/ios/App/build/TORIHologram.ipa" ]; then
            IOS_SIZE=$(du -h mobile/ios/App/build/TORIHologram.ipa | cut -f1)
            echo -e "iOS IPA size: ${GREEN}$IOS_SIZE${NC}"
            
            # Extract and analyze
            unzip -l mobile/ios/App/build/TORIHologram.ipa | tail -1
        fi
    fi
    
    if [[ "$PLATFORM" == "android" ]] || [[ "$PLATFORM" == "all" ]]; then
        if [ -f "mobile/android/app/build/outputs/apk/release/TORIHologram.apk" ]; then
            ANDROID_SIZE=$(du -h mobile/android/app/build/outputs/apk/release/TORIHologram.apk | cut -f1)
            echo -e "Android APK size: ${GREEN}$ANDROID_SIZE${NC}"
            
            # Analyze APK contents
            unzip -l mobile/android/app/build/outputs/apk/release/TORIHologram.apk | grep -E "(\.so|\.wgsl|\.js)" | sort -k4 -hr | head -10
        fi
    fi
}

# Main build process
main() {
    check_prerequisites
    install_dependencies
    generate_build_info
    build_web_assets
    sync_capacitor
    
    if [[ "$PLATFORM" == "ios" ]] || [[ "$PLATFORM" == "all" ]]; then
        build_ios
    fi
    
    if [[ "$PLATFORM" == "android" ]] || [[ "$PLATFORM" == "all" ]]; then
        build_android
    fi
    
    check_app_size
    
    echo -e "${GREEN}Build complete!${NC}"
}

# Run main
main
