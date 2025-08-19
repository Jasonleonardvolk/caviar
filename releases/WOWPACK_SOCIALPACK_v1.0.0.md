# WOW Pack + Social Pack Release v1.0.0
**Date:** 2025-08-19

## ✅ Features Implemented

### WOW Pack Pipeline
- **HEVC HDR10**: Full HDR10 support with mastering metadata SEI
- **AV1 HDR**: Modern codec with HDR support
- **H.264 SDR**: Universal compatibility fallback
- **Verification**: `tools\release\Verify-WowPack.ps1` confirms all encoding parameters

### Social Pack Pipeline  
- **TikTok Export**: 1080×1920 9:16 vertical, H.264 High@4.2, 60fps
- **Snapchat Export**: 1080×1920 9:16 vertical, H.264 High@4.2, 60fps
- **Size Management**: Auto-caps at 250MB for platform limits
- **Batch Processing**: Process multiple files with `Batch-Encode-Social.ps1`

## 📁 Directory Structure
```
content/
├── wowpack/
│   ├── input/     # ProRes/MOV masters
│   └── video/     # Encoded outputs
└── socialpack/
    ├── input/     # SDR sources
    ├── out/
    │   ├── tiktok/
    │   └── snap/
    └── thumbs/    # Auto-generated thumbnails
```

## 🔧 Scripts
- `tools/encode/Build-WowPack.ps1` - Single file HDR pipeline
- `tools/encode/Batch-Encode-Simple.ps1` - Batch WOW processing
- `tools/encode/Build-SocialPack.ps1` - Single file social export
- `tools/encode/Batch-Encode-Social.ps1` - Batch social processing
- `tools/release/Verify-WowPack.ps1` - Validate HDR metadata

## 📊 Verification Output
```
Mastering metadata: True (all HEVC variants)
Color primaries: bt2020
Transfer characteristics: smpte2084
Color space: bt2020nc
```

## 🚫 Git Exclusions
Media files excluded by design - only code/scripts committed

## 🎯 Usage
```powershell
# WOW Pack
.\tools\encode\Batch-Encode-Simple.ps1

# Social Pack  
.\tools\encode\Batch-Encode-Social.ps1 -Framerate 60
```

## 📱 Platform Specs
- **TikTok**: 9:16, 1080×1920, 5-60s, <250MB
- **Snapchat**: 9:16, 1080×1920, 5-60s, <250MB
